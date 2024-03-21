"""Autoencoder training configs."""

import tomllib
from typing import Self
from pydantic import BaseModel, Field

from training.transformer.config import DatasetConfig, WandbConfig


class AutoencoderDatasetConfig(BaseModel):
    """Configuration of dataset to train dictionary model."""

    dataset_id: str = Field(..., description="Name of HuggingFace dataset with text.")
    dataset_text_key: str = Field(..., description="Name of text column in dataset.")
    dataset_split: str = Field(..., description="Which split to use.")
    target_layer: int = Field(..., description="Target layer of activations.")
    target_activation_name: str = Field(..., description="Target activation name.")
    target_model_name: str = Field(..., description="Name of model to target.")
    batch_size: int = Field(..., description="Batch size.")
    shuffle: int = Field(..., description="Whether to shuffle data.")
    tokenizer_id: str = Field(
        ..., description="ID of tokenizer for blocked tokenization."
    )
    block_size: int = Field(..., description="Block size of data.")


class ModelConfig(BaseModel):
    l1_coefficient: float = Field(..., description="L1 loss coefficient.")
    tied: bool = Field(..., description="Whether to tie encoder and decoder layers.")
    torch_dtype: str = Field(..., description="Torch data-type of model.")


class TrainingConfig(BaseModel):
    learning_rate: float = Field(..., description="Learning rate.")
    adam_beta1: float = Field(..., description="Adam optimizer beta-1 value.")
    adam_beta2: float = Field(..., description="Adam optimizer beta-2 value.")
    reconstruction_loss_sample_amount: int = Field(
        ..., description="Number of samples to use to compute reconstruction loss."
    )
    use_ghost_gradients: bool = Field(
        ..., description="Whether to use ghost gradients."
    )
    dead_feature_window: int | None = Field(
        ..., description="Dead feature window for ghost gradients."
    )
    dictionary_multiplier: float = Field(
        ...,
        description="Dictionar activation multiplier, how many more dimensions than activations.",
    )
    model: ModelConfig = Field(..., description="Model configuration.")
    data: AutoencoderDatasetConfig = Field(
        ..., description="Dataset and data loading configuration."
    )
    wandb: WandbConfig | None = Field(
        default=None, description="Configuration of Weights & Biases."
    )
    seed: int = Field(..., description="Seed for training.")

    @classmethod
    def from_toml_path(cls, file_path: str) -> Self:
        """Loads the training configuration from a TOML file.

        Args:
            file_path (str): Path to the TOML file.

        Returns:
            TrainingConfig: An instance of the TrainingConfig class with data loaded from the TOML file.
        """
        with open(file_path, "rb") as file:
            toml_data = tomllib.load(file)["training"]
        return cls(**toml_data)
