"""Train a model.

Prior to migration to the high-level Trainer, this was
used to train handrolled transformers.

It will be transformed to train sparse autoencoders.
"""

import os
import numpy as np

import typer
import torch
import math
import wandb
import tempfile

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import islice
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from torch.optim import Optimizer
from models.sparse_autoencoder.model import Autoencoder, AutoencoderConfig
from transformers import AutoConfig
from training.autoencoder.config import TrainingConfig
from transformer_lens import HookedTransformer
from models.sparse_autoencoder.utils import get_model_activations, hooked_model_fixed
from training.autoencoder.data import ActivationDataset, split_dataset
from training.autoencoder.loss import get_reconstruction_loss

# Useful to set all seeds.
from transformers import set_seed


@torch.no_grad()
def get_activation_frequencies(
    model: HookedTransformer,
    encoder: Autoencoder,
    samples: list[str],
    target_layer: int,
    target_activation_name: str,
    device: str = "cuda",
) -> tuple[torch.Tensor, float]:
    frequency_scores: torch.Tensor = torch.zeros(
        encoder.config.hidden_size, dtype=torch.float32
    ).to(device)
    total: int = 0

    for text in samples:
        activations = get_model_activations(
            model=model,
            model_input=text,
            layer=target_layer,
            activation_name=target_activation_name,
        )
        hidden = encoder(activations, use_ghost_gradients=False)[2]
        frequency_scores += (hidden > 0).sum(0)
        total += hidden.shape[0]

    frequency_scores /= total
    dead_percentage = (frequency_scores == 0).float().mean()

    return frequency_scores, dead_percentage


def get_uniform_sample_loader(dataset, sample_size, batch_size=1):
    """
    Create a DataLoader for a uniform sample of the dataset.

    Args:
        dataset: The dataset to sample from.
        sample_size: The number of samples to draw from the dataset.
        batch_size: The batch size for the DataLoader of the sampled subset.

    Returns:
        A DataLoader instance for the sampled subset.
    """
    sample_size = min(sample_size, len(dataset))
    indices = np.random.choice(len(dataset), size=sample_size, replace=False)
    indices = indices.astype(int).tolist()

    subset = Subset(dataset, indices)
    subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    return subset_loader


def train_epoch(
    accelerator: Accelerator,
    model: Autoencoder,
    optimizer: Optimizer,
    target_model: HookedTransformer,
    data_loader: DataLoader,
    data_loader_eval: DataLoader,
    training_config: TrainingConfig,
    forward_passes_since_last_activation: torch.Tensor,
):
    sample_loader = get_uniform_sample_loader(
        data_loader_eval.dataset.text_dataset,
        training_config.reconstruction_loss_sample_amount,
        batch_size=1,
    )
    texts = [
        torch.tensor(sample["input_ids"], device=target_model.cfg.device)
        for sample in sample_loader
    ]

    data_loader = tqdm(data_loader, desc="Training")
    ghost_gradient_neuron_mask: torch.Tensor | None = None

    active_tokens = 0

    for i, batch in enumerate(data_loader):
        active_tokens += batch.size(0) * batch.size(1)

        with accelerator.accumulate(model):
            if training_config.use_ghost_gradients:
                ghost_gradient_neuron_mask = (
                    forward_passes_since_last_activation[i]
                    > training_config.dead_feature_window
                ).bool()

                print("Max dead length:", forward_passes_since_last_activation[i].max())
                print("Ghost gradient mask, any non-0 mask:", ghost_gradient_neuron_mask.any())

            model.train()
            loss, _, latents_pre_act, l2_loss, l1_loss = model(
                batch,
                use_ghost_gradients=training_config.use_ghost_gradients,
                ghost_gradient_neuron_mask=ghost_gradient_neuron_mask,
            )

            if training_config.use_ghost_gradients:
                # Feature activation book-keeping.
                did_fire = ((latents_pre_act > 0).float().sum(-2) > 0).any(dim=0)
                print("Fire mean", did_fire.float().mean())
                forward_passes_since_last_activation += 1
                forward_passes_since_last_activation[did_fire] = 0

            accelerator.backward(loss)

            model.make_decoder_weights_and_gradient_unit_norm()
            # NOTE: This might inhibit training.
            model.remove_gradients_parallel_to_decoder_directions()

            optimizer.step()
            optimizer.zero_grad()

        data_loader.set_description(f"Training - Loss: {loss.item():.4f}")

        if accelerator.sync_gradients:
            if training_config.wandb:
                if i % 100:
                    wandb.log(
                        {
                            "loss": loss.item(),
                            "l2_loss": l2_loss.item(),
                            "l1_loss": l1_loss.item(),
                        }
                    )

                # TODO: Don't hack this. This is just temporary. Tired.
                if i % 30000 and accelerator.is_main_process:# and reconstruction_score > 0.4:
                    with tempfile.TemporaryDirectory() as checkpoint_dir:
                        accelerator.save_state(checkpoint_dir)
                        artifact = wandb.Artifact(
                            f"model-checkpoint-{i}",
                            type="model",
                            description=f"Model checkpoint at step {i}"
                        )
                        artifact.add_dir(checkpoint_dir)
                        wandb.log_artifact(artifact)

                if i % 1000:
                    # TODO: Configurable interval.
                    reconstruction_score, *_ = get_reconstruction_loss(
                        model=target_model,
                        encoder=model,
                        samples=texts,
                        target_activation_name=training_config.data.target_activation_name,
                    )

                    activation_frequencies, dead_percentage = (
                        get_activation_frequencies(
                            model=target_model,
                            encoder=model,
                            samples=texts,
                            target_activation_name=training_config.data.target_activation_name,
                            target_layer=training_config.data.target_layer,
                            device=model.device,
                        )
                    )

                    feature_sparsity = activation_frequencies / active_tokens
                    active_tokens = 0

                    log_feature_sparsity = (
                        torch.log10(feature_sparsity + 1e-10).detach().cpu()
                    )

                    frequency_below_1e_minus_6 = (
                        (activation_frequencies < 1e-6).float().mean().item()
                    )
                    frequency_below_1e_minus_5 = (
                        (activation_frequencies < 1e-5).float().mean().item()
                    )
                    frequency_below_1e_minus_2 = (
                        (activation_frequencies < 1e-2).float().mean().item()
                    )

                    # Explainability.
                    l0_mean = (activation_frequencies > 0).float().sum(-1).detach().mean().item()

                    # Feature sparsity histogram.
                    plt.figure(figsize=(10, 6))  # You can set the size of the figure
                    sns.histplot(log_feature_sparsity.numpy())
                    plt.xlabel("log_10(Feature density)")
                    plt.ylabel("Frequency")
                    plt.title("Feature density")

                    plot_filename = "density_plot.png"
                    plt.savefig(plot_filename)
                    plt.close()

                    wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())

                    wandb.log(
                        {
                            "reconstruction_score": reconstruction_score,
                            "dead_percentage": dead_percentage,
                            "frequency_below_1e-6": frequency_below_1e_minus_6,
                            "frequency_below_1e-5": frequency_below_1e_minus_5,
                            "frequency_below_1e-2": frequency_below_1e_minus_2,
                            "l0_mean": l0_mean,
                            "plots/feature_density_line_chart": wandb_histogram,
                            "plots/feature_kdeplot": wandb.Image(plot_filename),
                        }
                    )

                    print("Reconstruction score:", reconstruction_score)
                    print("Dead percentage:", dead_percentage)
                    print("Frequency below 1e-6:", frequency_below_1e_minus_6)
                    print("Frequency below 1e-5:", frequency_below_1e_minus_5)
                    print("Frequency below 1e-2:", frequency_below_1e_minus_2)


def train(
    accelerator: Accelerator,
    model: Autoencoder,
    target_model: HookedTransformer,
    data_loader: DataLoader,
    data_loader_eval: DataLoader,
    training_config: TrainingConfig,
):
    if training_config.wandb and accelerator.is_main_process:
        wandb.init(
            project=training_config.wandb.project,
            notes=training_config.wandb.notes,
            tags=training_config.wandb.tags,
            config=training_config.model_dump(),
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(training_config.adam_beta1, training_config.adam_beta2),
    )

    (
        model,
        optimizer,
        data_loader,
    ) = accelerator.prepare(
        model,
        optimizer,
        data_loader,
    )

    # Used for ghost gradient mask.
    forward_passes_since_last_activation = torch.zeros(
        model.config.hidden_size,
        device=model.device,
    )

    train_epoch(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        target_model=target_model,
        data_loader=data_loader,
        data_loader_eval=data_loader_eval,
        training_config=training_config,
        forward_passes_since_last_activation=forward_passes_since_last_activation,
    )

    if training_config.wandb and accelerator.is_main_process:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
            torch.save(unwrapped_model.state_dict(), tmp_file.name)
            model_artifact = wandb.Artifact("trained_model", type="model")
            model_artifact.add_file(tmp_file.name)
            wandb.log_artifact(model_artifact)

        wandb.finish()


def train_autoencoder(file_path: str) -> None:
    accelerator = Accelerator()
    training_config = TrainingConfig.from_toml_path(file_path=file_path)
    set_seed(training_config.seed)

    target_model_config = AutoConfig.from_pretrained(
        training_config.data.target_model_name
    )
    target_model = hooked_model_fixed(
        training_config.data.target_model_name, dtype=training_config.model.torch_dtype
    )
    model = Autoencoder(
        AutoencoderConfig(
            hidden_size=int(
                training_config.dictionary_multiplier
                * target_model_config.intermediate_size
            ),
            input_size=int(target_model_config.intermediate_size),
            tied=training_config.model.tied,
            l1_coefficient=training_config.model.l1_coefficient,
            torch_dtype=training_config.model.torch_dtype,
        )
    )

    dataset = ActivationDataset(
        dataset_id=training_config.data.dataset_id,
        dataset_text_key=training_config.data.dataset_text_key,
        dataset_split=training_config.data.dataset_split,
        target_layer=training_config.data.target_layer,
        target_activation_name=training_config.data.target_activation_name,
        model=target_model,
        tokenizer_id=training_config.data.tokenizer_id,
        block_size=training_config.data.block_size,
    )

    model = Autoencoder(
        AutoencoderConfig(
            hidden_size=int(
                training_config.dictionary_multiplier
                * target_model_config.intermediate_size
            ),
            input_size=int(target_model_config.intermediate_size),
            tied=training_config.model.tied,
            l1_coefficient=training_config.model.l1_coefficient,
            torch_dtype=training_config.model.torch_dtype,
        )
    )

    geomatric_median_sample_loader = get_uniform_sample_loader(
        dataset.text_dataset,
        int(len(dataset.text_dataset) * 0.1),  # TODO: Don't hardcode geometric median sample count.
        batch_size=1,
    )
    samples = [
        torch.tensor(sample["input_ids"], device=target_model.cfg.device)
        for sample in geomatric_median_sample_loader
    ]
    all_activations = [
        get_model_activations(
            model=target_model,
            model_input=text,
            layer=training_config.data.target_layer,
            activation_name=training_config.data.target_activation_name
        )
        for text in tqdm(samples, desc="Gathering model activations")
    ]

    model.initialise_decoder_bias_with_geometric_median(
        all_activations=torch.vstack(all_activations)
    )

    dataset_train, dataset_validation = split_dataset(dataset=dataset, validation_percentage=training_config.data.validation_percentage)

    data_loader_train = DataLoader(
        dataset=dataset_train,
        batch_size=training_config.data.batch_size,
        shuffle=training_config.data.shuffle,
    )

    data_loader_eval = DataLoader(
        dataset=dataset_validation,
        batch_size=training_config.data.batch_size,
        shuffle=training_config.data.shuffle,
    )

    train(
        accelerator=accelerator,
        model=model,
        target_model=target_model,
        data_loader=data_loader_train,
        data_loader_eval=data_loader_eval,
        training_config=training_config,
    )


if __name__ == "__main__":
    typer.run(train_autoencoder)
