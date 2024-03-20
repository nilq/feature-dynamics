"""Train a model.

Prior to migration to the high-level Trainer, this was
used to train handrolled transformers.

It will be transformed to train sparse autoencoders.
"""

import os

import typer
import torch
import math
import wandb
import tempfile

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from models.sparse_autoencoder.model import Autoencoder, AutoencoderConfig
from transformers import AutoConfig
from training.autoencoder.config import TrainingConfig
from transformer_lens import HookedTransformer
from models.sparse_autoencoder.utils import get_model_activations, hooked_model_fixed
from training.autoencoder.data import ActivationDataset
from training.autoencoder.loss import get_reconstruction_loss


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
        hidden = encoder(activations)[2]

        frequency_scores = (hidden > 0).sum(0)
        total += hidden.shape[0]

    frequency_scores /= total
    dead_percentage = (frequency_scores == 0).float().mean()

    return frequency_scores, dead_percentage


def train_epoch(
    accelerator: Accelerator,
    model: Autoencoder,
    optimizer: Optimizer,
    target_model: HookedTransformer,
    data_loader: DataLoader,
    training_config: TrainingConfig,
):
    for i, batch in enumerate(data_loader):
        with accelerator.accumulate(model):
            loss, _, _, l2_loss, l1_loss = model(batch)
            accelerator.backward(loss)
            model.make_decoder_weights_and_grad_unit_norm()

            optimizer.step()
            optimizer.zero_grad()

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

                if i % 1000:
                    samples = data_loader

                    reconstruction_score, *_ = get_reconstruction_loss(
                        model=target_model,
                        encoder=model,
                        samples=samples,
                        target_activation_name=training_config.model.target_activation_name
                    )

                    activation_frequencies, dead_percentage = get_activation_frequencies(
                        model=target_model,
                        encoder=model,
                        samples=samples,
                        target_activation_name=training_config.model.target_activation_name,
                        target_layer=training_config.model.target_layer,
                        device=model.device
                    )

                    wandb.log(
                        {
                            "reconstruction_score": reconstruction_score,
                            "dead_percentage": dead_percentage,
                            "frequency_below_1e-6": (activation_frequencies < 1e-6).float().mean().item(),
                            "frequency_below_1e-5": (activation_frequencies < 1e-5).float().mean().item(),
                        }
                    )


def train(
    accelerator: Accelerator,
    model: Autoencoder,
    target_model: HookedTransformer,
    data_loader: DataLoader,
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

    train_epoch(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        target_model=target_model,
        data_loader=data_loader,
        training_config=training_config,
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

    target_model_config = AutoConfig.from_pretrained(
        training_config.model.target_model_name
    )
    target_model = hooked_model_fixed(training_config.model.target_model_name)
    model = Autoencoder(
        AutoencoderConfig(
            hidden_size=training_config.dictionary_multiplier
            * target_model_config.intermediate_size,
            input_size=target_model_config.intermediate_size,
            tied=training_config.model.tied,
            l1_coefficient=training_config.model.l1_coefficient,
            torch_dtype=training_config.model.torch_dtype,
        )
    )

    dataset = ActivationDataset(
        dataset_name=training_config.data.dataset_name,
        dataset_text_column=training_config.data.dataset_text_column,
        dataset_split=training_config.data.dataset_split,
        target_layer=training_config.data.target_layer,
        target_activation_name=training_config.data.target_activation_name,
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=training_config.data.batch_size,
        shuffle=training_config.data.shuffle,
    )

    train(
        accelerator=accelerator,
        model=model,
        target_model=target_model,
        data_loader=data_loader,
        training_config=training_config,
    )


if __name__ == "__main__":
    typer.run(train_autoencoder)
