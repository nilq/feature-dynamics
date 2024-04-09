"""Configuration of autoencoder evaluations."""

import tomllib
from typing import Self
from pydantic import BaseModel, Field

class AutoencoderEvaluationConfig(BaseModel):
    """Configuration of dataset to train dictionary model."""
    wandb_url: str = Field(..., description="URL of W&B experiment.")
    wandb_model_version: str = Field(..., description="Version of model artefact to evaluate.")

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
