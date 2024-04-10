"""Evaluation of sparse autoencoder."""

import typer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.sparse_autoencoder.utils import get_model_activations

from training.transformer.data import datasplit_from_dataset_config

from evaluation.autoencoder.config import AutoencoderEvaluationConfig
from evaluation.utils import load_autoencoder_from_wandb
from models.sparse_autoencoder.model import Autoencoder
from transformer_lens import HookedTransformer

from urllib.parse import urlparse
from accelerate import Accelerator


def softmax_cross_entropy_with_logits(
    logits: torch.Tensor, target_logits: torch.Tensor
):
    """Manually computes the cross-entropy between logits and targets.

    Args:
        logits (torch.Tensor): Logits from the model.
        target_logitsn (torch.Tensor): Target logits to compare against.

    Returns:
        cross_entropy (torch.Tensor): The cross-entropy between the logits and targets.
    """
    probs = F.softmax(logits, dim=-1)
    target_probs = F.softmax(target_logits, dim=-1)
    cross_entropy = -torch.sum(target_probs * torch.log(probs + 1e-9), dim=-1).mean()

    return cross_entropy


def evaluate_average_logit_cross_entropy(
    hooked_target_model: HookedTransformer,
    clean_target_model: HookedTransformer,
    validation_loader: DataLoader,
    sample_size: int,
) -> float:
    """Evaluate average logit cross-entropy.

    Args:
        hooked_target_model (HookedTransformer): Model with reconstructed latents.
        clean_target_model (HookedTransformer): Clean model to compare with.
        validation_loader (DataLoader): Validation data loader to sample from.
        sample_size (int): Sample size used to compute cross-entropy average.

    Returns:
        float: Average cross-entropy.
    """
    total_cross_entropy: float = 0.0

    for i, sample in enumerate(validation_loader):
        if i >= sample_size:
            break

        hooked_logits, _ = hooked_target_model.run_with_cache(
            torch.tensor(sample["input_ids"])
        )
        clean_logits, _ = clean_target_model.run_with_cache(
            torch.tensor(sample["input_ids"])
        )

        cross_entropy: torch.Tensor = softmax_cross_entropy_with_logits(
            hooked_logits, clean_logits
        )
        total_cross_entropy += cross_entropy.item()

    return total_cross_entropy / sample_size


# TODO: Take into account activation amount.
def feature_token_log(
    hooked_transformer: HookedTransformer,
    autoencoder: Autoencoder,
    validation_loader: DataLoader,
    sample_size: int,
) -> dict[int, list[str]]:
    """Get log of features and tokens that make it activate.

    Args:
        hooked_transformer (HookedTransformer): Hooked transformer to get activations.
        autoencoder (Autoencoder): Autoencoder to extract feature.
        validation_loader (DataLoader): Validation data loader.
        sample_size (int): Sample size.

    Returns:
        dict[int, list[str]]: Mapping of feature index to list of decoded tokens.
    """
    # Mapping of feature to all token IDs.
    feature_token_log: dict[int, list[str]] = {}
    tokenizer = hooked_transformer.tokenizer

    for i, sample in enumerate(validation_loader):
        if i >= sample_size:
            break

        activations = get_model_activations(
            model=hooked_transformer,
            model_input=torch.tensor(sample["input_ids"]),
            layer=0,
            activation_name="blocks.0.mlp.hook_post",
        )

        _, _, encoding, *_ = autoencoder(activations, use_ghost_gradients=False)

        for i, token_id in enumerate(sample["input_ids"]):
            for feature_index in encoding[i].nonzero():
                index: int = feature_index[0].item()
                feature_token_log[index] = feature_token_log.get(index, []) + [
                    tokenizer.decode([token_id])
                ]

    return feature_token_log


def load_autoencoder_from_config(
    config: AutoencoderEvaluationConfig,
) -> tuple[Autoencoder, HookedTransformer]:
    """Lead autoencoder and hooked target transformer from evaluation configuration.

    Args:
        config (AutoencoderEvaluationConfig): Configuration of autoencoder evaluation.

    Returns:
        tuple[Autoencoder, HookedTransformer]: Models loaded from config.
    """
    path_segments = urlparse(config.wandb_url).path.split("/")

    entity: str = path_segments[1]
    project_name: str = path_segments[2]
    run_id: str = path_segments[4]
    artifact_name: str = (
        f"{entity}/{project_name}/model-checkpoint-0:{config.wandb_model_version}"
    )

    autoencoder, target_model = load_autoencoder_from_wandb(
        artifact_name=artifact_name,
        entity=entity,
        project_name=project_name,
        run_id=run_id,
    )

    return autoencoder, target_model


def target_model_using_reconstructed_latents(
    config: AutoencoderEvaluationConfig,
    model: Autoencoder | None = None,
    target_model: HookedTransformer | None = None,
) -> HookedTransformer:
    """Evaluate generation of target model with substituted reconstructed latents.

    Args:
        config (AutoencoderEvaluationConfig): Configuration of autoencoder evaluation.
        model (Autoencoder | None, optional): Autoencoder model to use. Defaults to None.
        target_model (HookedTransformer | None, optional): Target model of autoencoder to hook here. Defaults to None.

    Returns:
        HookedTransformer: Hooked transformer with an added hook to insert reconstructed latent.
    """
    if not model and not target_model:
        model, target_model = load_autoencoder_from_config(config=config)

    def replace_mlp_output_hook(
        module: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor
    ):
        """Replace MLP activations with SAE reconstructions on the fly.

        Args:
            module (nn.Module): MLP module to which hook is attached.
            inputs (torch.Tensor): Inputs to module.
            outputs (torch.Tensor): Original outputs of module.
        """
        _, reconstruction, *_ = model(outputs, use_ghost_gradients=False)
        return reconstruction

    # Adding forward hook to replace MLP activations with reconstructions.
    target_model.blocks[0].mlp.hook_post.register_forward_hook(replace_mlp_output_hook)
    return target_model


def evaluate(config_path: str) -> None:
    accelerator = Accelerator()
    evaluation_config = AutoencoderEvaluationConfig.from_toml_path(
        file_path=config_path
    )

    autoencoder, clean_target_model = load_autoencoder_from_config(
        config=evaluation_config
    )
    hooked_target_model = target_model_using_reconstructed_latents(
        config=evaluation_config
    )

    _, validation_loader = datasplit_from_dataset_config(
        dataset_config=evaluation_config.dataset_config, training_config=accelerator
    )

    average_logit_cross_entropy = evaluate_average_logit_cross_entropy(
        hooked_target_model=hooked_target_model,
        clean_target_model=clean_target_model,
        validation_loader=validation_loader,
        sample_size=100,
    )

    print("Average logit cross-entropy:", average_logit_cross_entropy)

    feature_log: dict[int, list[str]] = feature_token_log(
        hooked_transformer=clean_target_model,
        autoencoder=autoencoder,
        validation_loader=validation_loader,
        sample_size=1000,
    )

    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    typer.run(evaluate)
