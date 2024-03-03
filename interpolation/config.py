"""Simple interpolation config."""

from pydantic import BaseModel, Field

from training.transformer.config import DatasetConfig


class InterpolationConfig(BaseModel):
    """Interpolation configuration."""
    base_model: str | None = Field(default=None, description="Base model, otherwise A.")

    model_a: str = Field(..., description="ID of model A.")
    model_b: str = Field(..., description="ID of model B.")

    dataset_a: DatasetConfig = Field(..., description="Dataset configuration for evaluation of model A.")
    dataset_b: DatasetConfig = Field(..., description="Dataset configuration for evaluation of model B.")

    stride: float = Field(..., description="Interpolation step-size.")
    dtype: str = Field(default="float32", description="Merge weight precision.")

    seed: int = Field(default=1337, description="Random seed of merge.")