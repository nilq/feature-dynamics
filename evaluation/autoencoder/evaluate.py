"""Evaluation of sparse autoencoder."""

import torch
import torch.nn as nn

from evaluation.autoencoder.config import AutoencoderEvaluationConfig
from evaluation.utils import load_autoencoder_from_wandb
from models.sparse_autoencoder.model import Autoencoder
from transformer_lens import HookedTransformer

from urllib.parse import urlparse, parse_qs


def load_autoencoder_from_config(config: AutoencoderEvaluationConfig) -> tuple[Autoencoder, HookedTransformer]:
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
    artifact_name: str = f"{entity}/{project_name}/model-checkpoint-0:{config.wandb_model_version}"

    autoencoder, target_model = load_autoencoder_from_wandb(
        artifact_name=artifact_name,
        entity=entity,
        project_name=project_name,
        run_id=run_id
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

    def replace_mlp_output_hook(module: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor):
        """Replace MLP activations with SAE reconstruction on the fly.

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
