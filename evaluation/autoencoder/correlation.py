"""Assess feature correlations, find matching features."""

import polars as pl
import pandas as pd
import itertools

from tqdm import tqdm
from scipy.stats import pearsonr



def correlated_features(df_A: pd.DataFrame, df_B: pd.DataFrame, correlation_threshold: float = 0.7) -> pd.DataFrame:
    """
    Parameters:
        df_A (pd.DataFrame): First data frame containing columns 'id', 'weight', and 'observation'.
        df_B (pd.DataFrame): Second data frame containing similar columns as df_A.
        correlation_threshold (float): The minimum correlation coefficient to consider a pair as correlated.

    Returns:
        pd.DataFrame: A DataFrame containing pairs of IDs with their correlation coefficient.
    """
    # Pivot the data frames
    pivot_A = df_A.pivot_table(index='feature', columns='token', values='activation')
    pivot_B = df_B.pivot_table(index='feature', columns='token', values='activation')

    token_intersection = list(set(pivot_A.columns).intersection(pivot_B.columns))
    pivot_A = pivot_A[token_intersection].fillna(0)
    pivot_B = pivot_B[token_intersection].fillna(0)

    results = []

    progress_wrapped_features = tqdm(itertools.product(pivot_A.index, pivot_B.index), total=len(pivot_A.index) * len(pivot_B.index))
    for feature_A, feature_B in progress_wrapped_features:
        aligned_A = pivot_A.loc[feature_A]
        aligned_B = pivot_B.loc[feature_B]

        if len(aligned_A) > 1 and len(aligned_B) > 1:
            correlation, _ = pearsonr(aligned_A, aligned_B)
            if correlation >= correlation_threshold:
                results.append({ "A": feature_A, "B": feature_B, "Correlation": correlation})

    result_df = pd.DataFrame(results)
    return result_df
