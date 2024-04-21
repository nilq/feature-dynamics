"""Evaluation utilities."""

import wandb
import pandas as pd
import numpy as np
import numpy.typing as npt

from models.sparse_autoencoder.model import Autoencoder, AutoencoderConfig
from models.sparse_autoencoder.utils import hooked_model_fixed
from training.autoencoder.config import TrainingConfig
from transformers import AutoConfig
from accelerate import Accelerator
from transformer_lens import HookedTransformer


from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper
from bokeh.transform import transform
from bokeh.palettes import Category10, Viridis256


def load_autoencoder_from_wandb(
    artifact_name: str,
    entity: str,
    project_name: str,
    run_id: str,
) -> tuple[Autoencoder, HookedTransformer]:
    """Load autoencoder from W&B.

    Args:
        artifact_name (str): Name of artifact to load from.

    Returns:
        tuple[Autoencoder, HookedTransformer]: Loaded autoencoder with target model.
    """
    wandb.init()

    artifact = wandb.use_artifact(artifact_name, type="model")
    artifact_dir = artifact.download()

    run = wandb.Api().run(f"{entity}/{project_name}/{run_id}")
    training_config = TrainingConfig(**run.config)

    target_model_config = AutoConfig.from_pretrained(
        training_config.data.target_model_name
    )

    model = Autoencoder(
        AutoencoderConfig(
            hidden_size=int(
                training_config.dictionary_multiplier
                * target_model_config.intermediate_size
            ),
            input_size=int(target_model_config.intermediate_size),
            tied=training_config.model.tied,
            l1_coefficient=training_config.model.l1_coefficient,
            torch_dtype=training_config.model.torch_dtype,
        )
    )

    target_model = hooked_model_fixed(
        training_config.data.target_model_name, dtype=training_config.model.torch_dtype
    )

    # Load model via accelerator.
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    accelerator.load_state(artifact_dir)

    return model, target_model


def preprocess_feature_information(
    df_feature_information: pd.DataFrame,
) -> pd.DataFrame:
    """Preprocess the dataframe to create a quick lookup for the max activation by feature.

    Args:
        df_feature_information (pd.DataFrame): Original dataframe with feature information.

    Returns:
        pd.DataFrame: A dataframe with maximum activations per feature.
    """
    idx = df_feature_information.groupby("feature")["activation"].idxmax()
    max_features = df_feature_information.loc[
        idx, ["feature", "token", "context", "activation"]
    ]
    return max_features.set_index("feature")


def max_activating_token(
    df_feature_information_preprocessed: pd.DataFrame, feature: int
) -> tuple[str, float]:
    """Get max activating token of feature.

    Args:
        df_feature_information_preprocessed (pd.DataFrame): Dataframe containing feature information.
        feature (int): Feature identifier.

    Returns:
        tuple[str, float]: Decoded token and activation of max-activating token.
    """
    try:
        row = df_feature_information_preprocessed.loc[feature]
        return (row["token"], row["activation"])
    except KeyError:
        return ("", 0.0)


def max_activating_context(
    df_feature_information_preprocessed: pd.DataFrame, feature: int
) -> tuple[str, float]:
    """Get max activating context of feature.

    Args:
        df_feature_information_preprocessed (pd.DataFrame): Dataframe containing feature information.
        feature (int): Feature identifier.

    Returns:
        tuple[str, float]: Decoded context and activation of max-activating context.
    """
    try:
        row = df_feature_information_preprocessed.loc[feature]
        return (row["context"], row["activation"])
    except KeyError:
        return ("", 0.0)


def show_plot_of_umap_clusters(
    umap_embeddings: npt.NDArray[np.float_],
    umap_cluster_labels: npt.NDArray[np.int_],
    df_feature_information: pd.DataFrame,
    top_n: int = 10,
) -> None:
    """Plot UMAP with coloured clusters using Bokeh, showing the max activating token on hover.

    Args:
        umap_embeddings (npt.NDArray[np.float_]): 2D UMAP embedding vectors.
        umap_cluster_labels (npt.NDArray[np.int_]): Labels for clusters.
        df_feature_information (pd.DataFrame): DataFrame with feature information.
        top_n (int, optional): Number of largest clusters to plot.
    """
    mask = umap_cluster_labels != -1
    filtered_embeddings = umap_embeddings[mask]
    filtered_labels = umap_cluster_labels[mask]

    unique_labels, counts = np.unique(filtered_labels, return_counts=True)
    top_labels = unique_labels[np.argsort(-counts)][:top_n]

    mask = np.isin(filtered_labels, top_labels)
    filtered_embeddings = filtered_embeddings[mask]
    filtered_labels = filtered_labels[mask]

    df_preprocessed = preprocess_feature_information(
        df_feature_information=df_feature_information
    )
    features: list[int] = np.where(np.isin(umap_cluster_labels, top_labels))[0]
    max_activating_tokens: list[str] = [
        max_activating_token(
            df_feature_information_preprocessed=df_preprocessed, feature=feature
        )
        for feature in features
    ]
    max_activating_contexts: list[str] = [
        max_activating_context(
            df_feature_information_preprocessed=df_preprocessed, feature=feature
        )
        for feature in features
    ]

    source = ColumnDataSource(
        data={
            "x": filtered_embeddings[:, 0],
            "y": filtered_embeddings[:, 1],
            "cluster": filtered_labels.astype(str),
            "max_activating_token": max_activating_tokens,
            "max_activating_context": max_activating_contexts,
            "feature": features,
        }
    )

    # Color mapping and plot configuration
    clusters = np.unique(filtered_labels).astype(str)
    num_clusters = len(clusters)
    palette = (
        Category10[10][:num_clusters]
        if num_clusters <= 10
        else Viridis256[:num_clusters]
    )
    color_mapper = CategoricalColorMapper(factors=clusters, palette=palette)

    plot = figure(
        title="UMAP Clustering",
        tools="pan,wheel_zoom,reset,save",
        width=800,
        height=800,
    )
    plot.scatter(
        "x",
        "y",
        source=source,
        legend_field="cluster",
        fill_alpha=0.6,
        size=7,
        color=transform("cluster", color_mapper),
    )

    plot.add_tools(
        HoverTool(
            tooltips=[
                ("Cluster", "@cluster"),
                ("Max Activating Token", "@max_activating_token"),
                ("Feature ID", "@feature"),
                ("Max Activating Context", "@max_activating_context"),
            ]
        )
    )
    show(plot)


def clean_feature_information_from_csv(csv_path: str) -> pd.DataFrame:
    """Load DataFrame of feature information, removing dirty data.

    Args:
        path (str): Path to CSV.

    Returns:
        pd.DataFrame: DataFrame with feature information.
    """
    df = pd.read_csv(csv_path)
    df = df[df["token"] == df["token"]]
    df = df[df["feature"].isin(range(16382))]

    return df
