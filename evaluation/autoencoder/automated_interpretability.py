"""Automated feature interpretability."""

import numpy as np
import numpy.typing as npt
import pandas as pd
import typer


def select_activation_examples(df: pd.DataFrame, feature_number: int) -> pd.DataFrame:
    """Select activation samples for feature.

    Args:
        df (pd.DataFrame): DataFrame containing the columns activation data.
        feature_number (int): Which feature to select samples for.

    Returns:
        pd.DataFrame: DataFrame with sample data.
    """

    def quantize_activations(activations: pd.Series) -> npt.NDArray[np.float_]:
        """Quantize activation series to be in [0;9].

        Args:
            activations (pd.Series): Activations to quantize.

        Returns:
            npt.NDArray[np.float_]: Quantized activations.
        """
        return np.floor(10 * activations / activations.max()).astype(int)

    df["quantized_activation"] = quantize_activations(df["activation"])

    feature_data = df[df["feature"] == feature_number]

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
