"""Training configuration models."""

import tomllib

from typing import Self
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

from transformers.training_args import TrainingArguments

from typing import Any, Literal


class ModelConfig(BaseModel):
    """Model configurations."""

    model_type: str = Field(
        ...,
        description="Name of autoregressive model supported by HuggingFace transformers ",
    )
    model_config_overrides: dict[str, Any] | None = Field(
        default=None, description="Overriding model configuration settings."
    )
    torch_dtype: Literal["auto", "bfloat16", "float16", "float32"] | None = Field(
        default=None, description="Overriding PyTorch default dtype."
    )


class WandbConfig(BaseModel):
    project: str = Field(..., description="Name of Wandb project.")
    notes: str = Field(..., description="Notes for run.")
    tags: list[str] = Field(..., description="List of tags for run.")


class DatasetConfig(BaseModel):
    """Configuration of datasets and data loading."""

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
    use_syntaxi: bool = Field(
        default=True, description="Whether to use Syntaxi'ed tokenizer."
    )

    test_mode: bool = Field(
        default=False, description="Whether to load a very tiny fraction for dry-run test."
    )


@dataclass
class TrainingConfig(TrainingArguments):
    wandb: WandbConfig | None = Field(
        default=None, description="Configuration of Weights & Biases."
    )
    dataset_config: DatasetConfig = Field(
        ..., description="Dataset and data loading configuration."
    )
    model_config: ModelConfig = Field(
        ..., description="Configuration of model to train."
    )

    gradient_accumulation_steps: int = Field(
        default=1, description="Number of gradient accumulation steps."
    )

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
