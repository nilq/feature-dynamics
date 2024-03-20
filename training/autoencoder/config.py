"""Autoencoder training configs."""

from pydantic import BaseModel, Field

from training.transformer.config import WandbConfig


class AutoencoderDatasetConfig(BaseModel):
    """Configuration of dataset to train dictionary model."""

    dataset_name: str = Field(..., description="Name of HuggingFace dataset with text.")
    dataset_text_column: str = Field(..., description="Name of text column in dataset.")
    dataset_split: str = Field(..., description="Which split to use.")
    target_layer: int = Field(..., description="Target layer of activations.")
    target_activation_name: str = Field(..., description="Target activation name.")


class ModelConfig(BaseModel):
    target_model_name: str = Field(..., description="Name of model to target.")
    l1_coefficient: float = Field(..., description="L1 loss coefficient.")
    tied: bool = Field(..., description="Whether to tie encoder and decoder layers.")
    torch_dtype: str = Field(..., description="Torch data-type of model.")


class TrainingConfig(BaseModel):
    learning_rate: float = Field(..., description="Learning rate.")
    adam_beta1: float = Field(..., description="Adam optimizer beta-1 value.")
    adam_beta2: float = Field(..., description="Adam optimizer beta-2 value.")
    dictionary_multiplier: float = Field(
        ...,
        description="Dictionar activation multiplier, how many more dimensions than activations.",
    )
    wandb: WandbConfig | None = Field(
        default=None, description="Configuration of Weights & Biases."
    )
