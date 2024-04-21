"""Activation selection module."""

import numpy as np
import numpy.typing as npt
import pandas as pd

from pydantic.dataclasses import dataclass
from pydantic import Field


@dataclass
class Activation:
    token: str = Field(..., description="Activating token.")
    context: str = Field(..., description="Context of token activation.")
    activation: float = Field(..., description="Activation amount.")
    activation_quantized: float = Field(
        ..., description="Activation amount quantized to be in [0;9]."
    )


def select_activation_samples(
    df: pd.DataFrame, feature_number: int
) -> list[Activation]:
    """Select activation samples for feature.

    Args:
        df (pd.DataFrame): DataFrame containing the columns activation data.
        feature_number (int): Which feature to select samples for.

    Returns:
        list[ActivationSample]: Selected activation samples.
    """
    df_samples: pd.DataFrame = select_activation_samples_df(
        df=df, feature_number=feature_number
    )

    return [
        Activation(
            token=row["token"],
            context=row["context"],
            activation=row["activation"],
            activation_quantized=row["quantized_activation"],
        )
        for _, row in df_samples.iterrows()
    ]


def select_activation_samples_df(df: pd.DataFrame, feature_number: int) -> pd.DataFrame:
    """Select activation samples for feature.

    Args:
        df (pd.DataFrame): DataFrame containing the columns activation data.
        feature_number (int): Which feature to select samples for.

    Returns:
        pd.DataFrame: DataFrame with sample data.
    """

    def quantize_activations(activations: pd.Series) -> pd.Series:
        """Quantize activation series to be in [0;9].

        Args:
            activations (pd.Series): Activations to quantize.

        Returns:
            pd.Series: Quantized activations.
        """
        normalized_activations = (activations - activations.min()) / (
            activations.max() - activations.min()
        )
        return np.floor(10 * normalized_activations).astype(int)

    feature_data = df[df["feature"] == feature_number]

    feature_data["quantized_activation"] = quantize_activations(
        feature_data["activation"]
    )

    top_interval_indices = feature_data[feature_data["quantized_activation"] == 9].index
    other_intervals_indices = feature_data[
        feature_data["quantized_activation"] < 9
    ].index

    top_examples = feature_data.loc[top_interval_indices].nlargest(10, "activation")
    other_examples = (
        feature_data.loc[other_intervals_indices]
        .groupby("quantized_activation")
        .apply(lambda x: x.nlargest(2, "activation"))
        .reset_index(drop=True)
    )
    random_examples = feature_data.sample(5)

    top_tokens = top_examples["token"].unique()
    different_context_examples = (
        feature_data[
            feature_data["token"].isin(top_tokens)
            & ~feature_data.index.isin(top_examples.index)
        ]
        .drop_duplicates("context")
        .head(10)
    )

    selected_examples = pd.concat(
        [top_examples, other_examples, random_examples, different_context_examples]
    )

    return selected_examples
