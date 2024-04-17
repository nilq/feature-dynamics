"""Assess feature correlations, find matching features."""

import pandas as pd
import itertools

from pathlib import Path
from tqdm import tqdm
from evaluation.autoencoder.config import CorrelationConfig
from scipy.stats import pearsonr

import typer


def correlated_features(
    df_a: pd.DataFrame, df_b: pd.DataFrame, correlation_threshold: float = 0.7
) -> pd.DataFrame:
    """
    Parameters:
        df_a (pd.DataFrame): First data frame containing columns 'id', 'weight', and 'observation'.
        df_b (pd.DataFrame): Second data frame containing similar columns as df_a.
        correlation_threshold (float): The minimum correlation coefficient to consider a pair as correlated.

    Returns:
        pd.DataFrame: A DataFrame containing pairs of IDs with their correlation coefficient.
    """

    # Paranoia, make 200% sure we're computing on clean data.
    df_a = df_a[df_a["feature"].isin(range(16_384))]
    df_b = df_b[df_b["feature"].isin(range(16_384))]
    df_a = df_a[df_a["token"] == df_a["token"]]
    df_b = df_b[df_b["token"] == df_b["token"]]

    # Pivot the data frames
    pivot_a = df_a.pivot_table(index="feature", columns="token", values="activation")
    pivot_b = df_b.pivot_table(index="feature", columns="token", values="activation")

    token_intersection = list(set(pivot_a.columns).intersection(pivot_b.columns))
    pivot_a = pivot_a[token_intersection].fillna(0)
    pivot_b = pivot_b[token_intersection].fillna(0)

    results = []

    progress_wrapped_features = tqdm(
        itertools.product(pivot_a.index, pivot_b.index),
        total=len(pivot_a.index) * len(pivot_b.index),
    )
    for feature_a, feature_b in progress_wrapped_features:
        aligned_a = pivot_a.loc[feature_a]
        aligned_b = pivot_b.loc[feature_b]

        if len(aligned_a) > 1 and len(aligned_b) > 1:
            correlation, _ = pearsonr(aligned_a, aligned_b)
            if correlation >= correlation_threshold:
                results.append(
                    {"A": feature_a, "B": feature_b, "Correlation": correlation}
                )

    return pd.DataFrame(results)


def correlate(file_path: str) -> None:
    correlation_config = CorrelationConfig.from_toml_path(file_path=file_path)
    df_model_a: pd.DataFrame = pd.read_csv(correlation_config.model_a_path).sample(
        n=correlation_config.sample_size
    )
    df_model_b: pd.DataFrame = pd.read_csv(correlation_config.model_b_path).sample(
        n=correlation_config.sample_size
    )

    df_output = correlated_features(
        df_a=df_model_a,
        df_b=df_model_b,
        correlation_threshold=correlation_config.correlation_threshold,
    )

    Path(correlation_config.output_path).parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(correlation_config.output_path)


if __name__ == "__main__":
    typer.run(correlate)
