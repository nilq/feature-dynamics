"""Evaluation of sparse autoencoder."""

import pandas as pd
import numpy as np
import numpy.typing as npt
import json
import umap.umap_ as umap
from pathlib import Path
import typer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.sparse_autoencoder.utils import get_model_activations
from sklearn.cluster import HDBSCAN
from tqdm import tqdm

from training.transformer.data import datasplit_from_dataset_config

from evaluation.autoencoder.config import AutoencoderEvaluationConfig
from evaluation.utils import load_autoencoder_from_wandb
from models.sparse_autoencoder.model import Autoencoder
from transformer_lens import HookedTransformer

from urllib.parse import urlparse
from accelerate import Accelerator


def autoencoder_feature_clustering(
    autoencoder: Autoencoder,
    n_neighbors: int = 25,
    min_distance: float = 0.1,
    metric: str = "cosine",
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.int_]]:
    """Extract feature UMAP and clusters from autoencoder decoding matrix.

    Args:
        autoencoder (Autoencoder): Autoencoder with feature dictionary.

    Returns:
        tuple[npt.NDArray[np.float_], npt.NDArray[np.int_]]: 2D UMAP embeddings, and HDBSCAN labels.
    """
    # Decoder matrix columns.
    feature_columns: torch.Tensor = autoencoder.decoder.weight.data.T

    # Clustering over 10-dimensional UMAP.
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_distance, metric=metric, n_components=10
    )
    umap_embedding_10D = reducer.fit_transform(feature_columns.float().cpu().numpy())
    hdb = HDBSCAN(min_cluster_size=3).fit(umap_embedding_10D)

    # 2-dimensional UMAP for visualisation.
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_distance, metric=metric, n_components=10
    )
    umap_embedding_2D = reducer.fit_transform(feature_columns.float().cpu().numpy())

    return umap_embedding_2D, hdb.labels_


def softmax_cross_entropy_with_logits(
    logits: torch.Tensor, target_logits: torch.Tensor
):
    """Manually computes the cross-entropy between logits and targets.

    Args:
        logits (torch.Tensor): Logits from the model.
        target_logitsn (torch.Tensor): Target logits to compare against.

    Returns:
        cross_entropy (torch.Tensor): The cross-entropy between the logits and targets.
    """
    probs = F.softmax(logits, dim=-1)
    target_probs = F.softmax(target_logits, dim=-1)
    cross_entropy = -torch.sum(target_probs * torch.log(probs + 1e-9), dim=-1).mean()

    return cross_entropy


def evaluate_average_logit_cross_entropy(
    hooked_target_model: HookedTransformer,
    clean_target_model: HookedTransformer,
    validation_loader: DataLoader,
    sample_size: int,
) -> float:
    """Evaluate average logit cross-entropy.

    Args:
        hooked_target_model (HookedTransformer): Model with reconstructed latents.
        clean_target_model (HookedTransformer): Clean model to compare with.
        validation_loader (DataLoader): Validation data loader to sample from.
        sample_size (int): Sample size used to compute cross-entropy average.

    Returns:
        float: Average cross-entropy.
    """
    total_cross_entropy: float = 0.0

    for i, sample in enumerate(validation_loader):
        if i >= sample_size:
            break

        hooked_logits, _ = hooked_target_model.run_with_cache(
            torch.tensor(sample["input_ids"])
        )
        clean_logits, _ = clean_target_model.run_with_cache(
            torch.tensor(sample["input_ids"])
        )

        cross_entropy: torch.Tensor = softmax_cross_entropy_with_logits(
            hooked_logits, clean_logits
        )
        total_cross_entropy += cross_entropy.item()

    return total_cross_entropy / sample_size


def feature_activation_log(
    hooked_transformer: HookedTransformer,
    autoencoder: Autoencoder,
    validation_loader: DataLoader,
    sample_size: int,
    context_window_size: int,
) -> tuple[pd.DataFrame, dict[int, list[str]], dict[int, list[str]]]:
    """Get log of features and tokens that make them activate.

    Notes:
        Context activations contain the surrounding context (context_window_size - 1) backwards,
        and 1 fowards, i.e. the activating token is at index -2.

    Args:
        hooked_transformer (HookedTransformer): Hooked transformer to get activations.
        autoencoder (Autoencoder): Autoencoder to extract feature.
        validation_loader (DataLoader): Validation data loader.
        sample_size (int): Sample size.

    Returns:
        tuple[pd.DataFrame, dict[int, list[str]], dict[int, list[str]]]:
            DataFrame of feature statistics, feature token activation log, feature context activation log.
    """
    # Mapping of feature to all token IDs.
    feature_token_activation_log: dict[int, list[str]] = {}
    feature_context_activation_log: dict[int, list[str]] = {}
    tokenizer = hooked_transformer.tokenizer

    # DataFrame contents.
    tokens: list[str] = []
    contexts: list[str] = []  # Activating token at index -2.
    positions: list[int] = []
    feature_activations: list[float] = []
    features: list[int] = []

    for i, sample in tqdm(
        enumerate(validation_loader),
        total=len(validation_loader),
        desc="Gathering information",
    ):
        if i >= sample_size:
            break

        input_ids: list[int] = sample["input_ids"]

        activations = get_model_activations(
            model=hooked_transformer,
            model_input=torch.tensor(input_ids),
            layer=0,
            activation_name="blocks.0.mlp.hook_post",
        )
        _, _, encoding, *_ = autoencoder(activations, use_ghost_gradients=False)

        for i, token_id in enumerate(input_ids):
            for feature_index in encoding[i].nonzero():
                # Index of our feature.
                index: int = feature_index[0].item()

                # Trailing context window, with activating feature at -2.
                start_index = max(0, i - context_window_size + 1)
                context_tokens = input_ids[start_index : i + 2]

                # Decoding.
                token: str = tokenizer.decode([token_id])
                context: str = tokenizer.decode(context_tokens)

                # Gathering all raw data.
                feature_token_activation_log[index] = feature_token_activation_log.get(
                    index, []
                ) + [token]
                feature_context_activation_log[
                    index
                ] = feature_context_activation_log.get(index, []) + [context]

                # DataFrame information.
                features.append(index)
                tokens.append(token)
                contexts.append(context)
                positions.append(i)
                feature_activations.append(encoding[i][feature_index].item())

    feature_df: pd.DataFrame = pd.DataFrame(
        {
            "feature": features,
            "activation": feature_activations,
            "token": tokens,
            "context": contexts,
            "position": positions,
        }
    )

    return feature_df, feature_token_activation_log, feature_context_activation_log


def feature_context_activation_log(
    hooked_transformer: HookedTransformer,
    autoencoder: Autoencoder,
    validation_loader: DataLoader,
    sample_size: int,
    context_window_size: int = 5,
) -> dict[int, list[str]]:
    """Get log of contexts for features that make it activate.

    Args:
        hooked_transformer (HookedTransformer): Hooked transformer to get activations.
        autoencoder (Autoencoder): Autoencoder to extract feature.
        validation_loader (DataLoader): Validation data loader.
        sample_size (int): Sample size.
        context_window_size (int, optional): Size of context window to consider. Defaults to 5.

    Returns:
        dict[int, list[str]]: Mapping of feature index to list of decoded contexts.
    """
    # Mapping of feature to decoded contexts.
    feature_context_activation_log: dict[int, list[str]] = {}
    tokenizer = hooked_transformer.tokenizer

    for i, sample in tqdm(
        enumerate(validation_loader),
        total=len(validation_loader),
        desc="Gathering information",
    ):
        if i >= sample_size:
            break

        activations = get_model_activations(
            model=hooked_transformer,
            model_input=torch.tensor(sample["input_ids"]),
            layer=0,
            activation_name="blocks.0.mlp.hook_post",
        )

        _, _, encoding, *_ = autoencoder(activations, use_ghost_gradients=False)

        input_ids: list[int] = sample["input_ids"]
        for i, sample in enumerate(input_ids):
            for feature_index in encoding[i].nonzero():
                index: int = feature_index.item()
                start_index = max(0, i - context_window_size - 1)
                context_tokens = input_ids[start_index : i + 2]
                feature_context_activation_log[
                    index
                ] = feature_context_activation_log.get(index, []) + [
                    tokenizer.decode(context_tokens)
                ]

    return feature_context_activation_log


def load_autoencoder_from_config(
    config: AutoencoderEvaluationConfig,
) -> tuple[Autoencoder, HookedTransformer]:
    """Lead autoencoder and hooked target transformer from evaluation configuration.

    Args:
        config (AutoencoderEvaluationConfig): Configuration of autoencoder evaluation.

    Returns:
        tuple[Autoencoder, HookedTransformer]: Models loaded from config.
    """
    path_segments = urlparse(config.wandb_url).path.split("/")

    entity: str = path_segments[1]
    project_name: str = path_segments[2]
    run_id: str = path_segments[4]
    artifact_name: str = (
        f"{entity}/{project_name}/model-checkpoint-0:{config.wandb_model_version}"
    )

    autoencoder, target_model = load_autoencoder_from_wandb(
        artifact_name=artifact_name,
        entity=entity,
        project_name=project_name,
        run_id=run_id,
    )

    return autoencoder, target_model


def target_model_using_reconstructed_latents(
    config: AutoencoderEvaluationConfig,
    model: Autoencoder | None = None,
    target_model: HookedTransformer | None = None,
) -> HookedTransformer:
    """Evaluate generation of target model with substituted reconstructed latents.

    Args:
        config (AutoencoderEvaluationConfig): Configuration of autoencoder evaluation.
        model (Autoencoder | None, optional): Autoencoder model to use. Defaults to None.
        target_model (HookedTransformer | None, optional): Target model of autoencoder to hook here. Defaults to None.

    Returns:
        HookedTransformer: Hooked transformer with an added hook to insert reconstructed latent.
    """
    if not model and not target_model:
        model, target_model = load_autoencoder_from_config(config=config)

    def replace_mlp_output_hook(
        module: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor
    ):
        """Replace MLP activations with SAE reconstructions on the fly.

        Args:
            module (nn.Module): MLP module to which hook is attached.
            inputs (torch.Tensor): Inputs to module.
            outputs (torch.Tensor): Original outputs of module.
        """
        _, reconstruction, *_ = model(outputs, use_ghost_gradients=False)
        return reconstruction

    # Adding forward hook to replace MLP activations with reconstructions.
    target_model.blocks[0].mlp.hook_post.register_forward_hook(replace_mlp_output_hook)
    return target_model


def evaluate(config_path: str) -> None:
    accelerator = Accelerator()
    evaluation_config = AutoencoderEvaluationConfig.from_toml_path(
        file_path=config_path
    )

    autoencoder, clean_target_model = load_autoencoder_from_config(
        config=evaluation_config
    )
    hooked_target_model = target_model_using_reconstructed_latents(
        config=evaluation_config
    )

    _, validation_loader = datasplit_from_dataset_config(
        dataset_config=evaluation_config.dataset_config, training_config=accelerator
    )

    validation_loader = validation_loader.shuffle().select(
        range(evaluation_config.sample_size)
    )

    average_logit_cross_entropy = evaluate_average_logit_cross_entropy(
        hooked_target_model=hooked_target_model,
        clean_target_model=clean_target_model,
        validation_loader=validation_loader,
        sample_size=evaluation_config.sample_size,
    )

    print("Average logit cross-entropy:", average_logit_cross_entropy)

    feature_df, feature_log, context_log = feature_activation_log(
        hooked_transformer=clean_target_model,
        autoencoder=autoencoder,
        validation_loader=validation_loader,
        sample_size=evaluation_config.sample_size,
        context_window_size=8,
    )

    # UMAP.
    umap_embeddings, clustering_labels = autoencoder_feature_clustering(
        autoencoder=autoencoder,
    )

    # Write outputs.
    output_path: Path = Path(evaluation_config.output_data_path)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_df.to_csv(output_path / "feature_information.csv")
    (output_path / "feature_token_activation_log.json").open("w+").write(
        json.dumps(feature_log)
    )
    (output_path / "feature_context_activation_log.json").open("w+").write(
        json.dumps(context_log)
    )

    np.save(output_path / "umap_embedding_cluster_labels.npy", clustering_labels)
    np.save(output_path / "umap_embeddings.npy", umap_embeddings)


if __name__ == "__main__":
    typer.run(evaluate)
