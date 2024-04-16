"""Configuration of autoencoder evaluations."""

import tomllib
from typing import Self
from pydantic import BaseModel, Field

from training.transformer.config import DatasetConfig


class CorrelationConfig(BaseModel):
    """Configuration of correlation experiment."""
    sample_size: int = Field(..., description="Sample size for cross-correlation.")
    model_a_path: str = Field(..., description="Path to CSV containing feature data for model A.")
    model_b_path: str = Field(..., description="Path to CSV containing feature data for model B.")
    correlation_threshold: float = Field(..., description="Pearson-correlation threshold.")

    output_path: str = Field(..., description="Path to output correlation CSV file.")

    @classmethod
    def from_toml_path(cls, file_path: str) -> Self:
        """Loads the training configuration from a TOML file.

        Args:
            file_path (str): Path to the TOML file.

        Returns:
            TrainingConfig: An instance of the TrainingConfig class with data loaded from the TOML file.
        """
        with open(file_path, "rb") as file:
            toml_data = tomllib.load(file)["correlation"]
        return cls(**toml_data)


class AutoencoderEvaluationConfig(BaseModel):
    """Configuration of dataset to train dictionary model."""

    wandb_url: str = Field(..., description="URL of W&B experiment.")
    wandb_model_version: str = Field(
        ..., description="Version of model artefact to evaluate."
    )
    dataset_config: DatasetConfig = Field(
        ..., description="Text dataset used for evaluation."
    )
    output_data_path: str = Field(..., description="Where to output data.")
    sample_size: int = Field(
        ..., description="Sample size of dataset to evaluate with."
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
            toml_data = tomllib.load(file)["autoencoder-evaluation"]
        return cls(**toml_data)
