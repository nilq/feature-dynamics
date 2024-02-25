"""Training configuration models."""

import tomllib

from typing import Self
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    dataset_id: str = Field(..., description="Name of HuggingFace dataset.")
    dataset_config_name: str | None = Field(
        default=None, description="Name of dataset config."
    )
    dataset_text_key: str = Field(..., description="Where to find the text data.")

    tokenizer_id: str = Field(..., description="ID of HuggingFace tokenizer.")
    batch_size: int = Field(..., description="Batch size.")
    block_size: int = Field(
        default=1024, description="Tokenized sequence length of blocks."
    )
    validation_percentage: float = Field(
        default=0.2, description="Validation split percentage."
    )
    test_percentage: float = Field(default=0.1, description="Test split percentage.")
    use_syntaxi: bool = Field(
        default=True, description="Whether to use Syntaxi'ed tokenizer."
    )


class TransformerConfig(BaseModel):
    vocab_dim: int = Field(..., description="Vocabulary size.")
    embedding_dim: int = Field(..., description="Embedding dimension.")
    max_seq_len: int = Field(..., description="Max sequence length.")
    num_layers: int = Field(..., description="Number of transformer block layers.")
    num_heads: int = Field(..., description="Number of attention heads.")


class WandbConfig(BaseModel):
    project: str = Field(..., description="Name of Wandb project.")
    notes: str = Field(..., description="Notes for run.")
    tags: list[str] = Field(..., description="List of tags for run.")


class TrainingConfig(BaseModel):
    epochs: int = Field(..., description="Epochs to train for.")
    learning_rate: float = Field(..., description="Learning rate.")
    learning_rate_scheduler: str = Field(
        ..., description="Name of learning rate scheduler to use."
    )
    gradient_accumulation_steps: int
    warmup_steps: int
    adam_betas: list[float] = Field(default=[0.9, 0.999], description="AdamW betas.")
    weight_decay: float

    wandb: WandbConfig | None = Field(default=None, description="Using Wandb.")
    dataset_config: DatasetConfig
    transformer_config: TransformerConfig

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
