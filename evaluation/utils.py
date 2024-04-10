"""Evaluation utilities."""

import wandb

from models.sparse_autoencoder.model import Autoencoder, AutoencoderConfig
from models.sparse_autoencoder.utils import hooked_model_fixed
from training.autoencoder.config import TrainingConfig
from transformers import AutoConfig
from accelerate import Accelerator
from transformer_lens import HookedTransformer


def load_autoencoder_from_wandb(
    artifact_name: str,
    entity: str,
    project_name: str,
    run_id: str,
) -> tuple[Autoencoder, HookedTransformer]:
    """Load autoencoder from W&B.

    Args:
        artifact_name (str): Name of artifact to load from.

    Returns:
        tuple[Autoencoder, HookedTransformer]: Loaded autoencoder with target model.
    """
    wandb.init()

    artifact = wandb.use_artifact(artifact_name, type="model")
    artifact_dir = artifact.download()

    run = wandb.Api().run(f"{entity}/{project_name}/{run_id}")
    training_config = TrainingConfig(**run.config)

    target_model_config = AutoConfig.from_pretrained(
        training_config.data.target_model_name
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

    target_model = hooked_model_fixed(
        training_config.data.target_model_name, dtype=training_config.model.torch_dtype
    )

    # Load model via accelerator.
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    accelerator.load_state(artifact_dir)

    return model, target_model
