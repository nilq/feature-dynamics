"""Like correlate, but on the fly."""

from tqdm import tqdm
import typer
import torch
import datasets
import pandas as pd
import matplotlib.pyplot as plt
import h5py

from torch.nn.functional import normalize
from evaluation.autoencoder.evaluate import *
from pathlib import Path
from torch.utils.data import DataLoader
from training.transformer.data import (
    datasplit_from_dataset_config,
    tokenizer_from_dataset_config,
)
from sklearn.preprocessing import KBinsDiscretizer
from training.autoencoder.train import get_uniform_sample_loader
from evaluation.autoencoder.config import ComparisonConfig

import os


PATH_WITH_ENOUGH_SPACE: str = os.getenv("ACTIVATION_CACHE", "")


# To load corresponding autoencoder.
grandparent_dir: Path = Path(__file__).parent / ".." / ".."
model_name_to_config: dict[str, Path] = {
    "baby-python": grandparent_dir
    / "experiments/evaluation/autoencoder/baby_python_mistral_1L_tiny_base.toml",
    "lua": grandparent_dir
    / "experiments/evaluation/autoencoder/baby-python-mistral-1L-tiny-lua-ft/baby_python.toml",
    "tiny-stories": grandparent_dir
    / "experiments/evaluation/autoencoder/baby-python-mistral-1L-tiny-TinyStories-ft/baby_python.toml",
    "slerp": grandparent_dir
    / "experiments/evaluation/autoencoder/baby-python-mistral-1L-tiny-lua-stories-slerp/baby_python.toml",
}


def load_models(name: str) -> tuple[Autoencoder, HookedTransformer]:
    """Load autoencoder and hooked target model.

    Args:
        name (str): Name of model to load.

    Returns:
        tuple[Autoencoder, HookedTransformer]: Autoencoder and hooked target model.
    """
    accelerator = Accelerator()
    config_path = model_name_to_config[name].as_posix()

    evaluation_config = AutoencoderEvaluationConfig.from_toml_path(
        file_path=config_path
    )
    autoencoder, clean_target_model = load_autoencoder_from_config(
        config=evaluation_config
    )
    hooked_target_model = target_model_using_reconstructed_latents(
        config=evaluation_config
    )

    return autoencoder, hooked_target_model


def get_encodings(
    input_ids: list[int],
    autoencoder: Autoencoder,
    hooked_target_model: HookedTransformer,
) -> torch.Tensor:
    """Get autoencoder encodings for input token IDs.

    Args:
        input_ids (list[int]): Input token IDs.
        autoencoder (Autoencoder): Autoencoder.
        hooked_target_model (HookedTransformer): Hooked model to get activations for.

    Returns:
        torch.Tensor: _description_
    """
    activations = get_model_activations(
        model=hooked_target_model,
        model_input=torch.tensor(input_ids, device=autoencoder.device),
        layer=0,
        activation_name="blocks.0.mlp.hook_post",
    )
    _, _, encoding, *_ = autoencoder(activations, use_ghost_gradients=False)
    return encoding


def plot_feature_correlations(features_by_model: dict[str, list[int]], save_path: str):
    """
    Load features for specific models and plot their correlations for features at the same index.

    Args:
        features_by_model (dict[str, list[int]]): Dictionary where keys are model names and values are lists of feature indices.
        save_path (str): Directory path where HDF5 files are saved.

    Returns:
        None: The function saves plots as files and does not return any value.
    """
    feature_activations = load_specific_features(features_by_model, save_path)

    # Assuming all models have the same number of specified features, compare corresponding features
    models = list(features_by_model.keys())
    if len(models) < 2:
        raise ValueError("Need at least two models to compare features.")

    num_features = min(len(features) for features in features_by_model.values())
    for feature_idx in range(num_features):
        model_feature_data = {}

        # Collect data for the current feature index from all models
        for model in models:
            feature_number = features_by_model[model][feature_idx]
            model_feature_data[model] = feature_activations[model][feature_number]

        # Prepare data sets for plotting
        plotted_models = list(model_feature_data.keys())
        for i in range(len(plotted_models) - 1):
            for j in range(i + 1, len(plotted_models)):
                model_i, model_j = plotted_models[i], plotted_models[j]
                data_i, data_j = (
                    model_feature_data[model_i],
                    model_feature_data[model_j],
                )

                # Compute correlation
                correlation = np.corrcoef(data_i, data_j)[0, 1] * 100

                # Plotting
                plt.figure(figsize=(10, 6))
                plt.scatter(data_i, data_j, alpha=0.5)
                title = f"{model_i}/{features_by_model[model_i][feature_idx]} vs {model_j}/{features_by_model[model_j][feature_idx]} - correlation {correlation:.1f}%"
                plt.title(title)
                plt.xlabel(f"{model_i}/{features_by_model[model_i][feature_idx]}")
                plt.ylabel(f"{model_j}/{features_by_model[model_j][feature_idx]}")
                plt.grid(True)

                # Save the plot
                filename = f"{model_i}_{features_by_model[model_i][feature_idx]}__{model_j}_{features_by_model[model_j][feature_idx]}.png"
                plt.savefig(Path(save_path) / filename)
                plt.close()

    print("All plots generated and saved.")


def load_specific_features(
    features_by_model: dict[str, list[int]], save_path: str
) -> dict[str, dict[int, list[float]]]:
    """Load specific features for specific models.

    Args:
        features_by_model (dict[str, list[int]]): Dictionary where keys are model names and values are lists of feature indices.
        save_path (str): Directory path with activation data.

    Returns:
        dict[str, dict[int, list[float]]]: Dictionary of models and specific feature activation data.
    """
    results = {}
    path = Path(save_path)

    for model_name, features in features_by_model.items():
        model_file_path = path / f"activations_{model_name}.h5"
        feature_data = {}

        with h5py.File(model_file_path, "r") as file:
            activations_group = file["activations"]
            for feature_index in features:
                dataset_name = f"feature_{feature_index}"
                if dataset_name in activations_group:
                    feature_data[feature_index] = activations_group[dataset_name][:]

        results[model_name] = feature_data

    return results


def load_feature_activations(
    file_path: str, sample_size: int | None = None, seed: int = 42
) -> dict[str, dict[int, list[float]]]:
    """Load feature activations from an HDF5 file into the desired dictionary format.

    Args:
        file_path (str): Path to the HDF5 file containing activations.

    Returns:
        dict[str, dict[int, list[float]]]: Nested dictionary of models to features to activations.
    """
    model_feature_activations: dict[str, dict[int, list[float]]] = {}

    with h5py.File(file_path, "r") as h5f:
        for model_name in h5f.keys():
            model_data = {}
            feature_list = list(h5f[model_name].keys())

            if sample_size is not None and sample_size < len(feature_list):
                random.seed(seed)
                feature_list = random.sample(feature_list, sample_size)

            for feature_dataset in feature_list:
                feature_index = int(feature_dataset.split("_")[-1])
                activations = h5f[model_name][feature_dataset][:]
                model_data[feature_index] = activations.tolist()
            model_feature_activations[model_name] = model_data

    return model_feature_activations


def save_feature_activations(
    model_names: list[str], sample_loader: DataLoader, save_path: str
) -> dict[str, str]:
    """Correlate features

    Notes:
        Saves a separate HDF5 file for each model containing a mapping of features to their activations on samples.

    Args:
        model_names (list[str]): List of models.
        sample_loader (DataLoader): Samples to get activations over.
        save_path (str): Directory path to incrementally save HDF5 files.

    Returns:
        dict[str, str]: Mapping of file paths to saved activations for each model.
    """
    models: dict[str, tuple[Autoencoder, HookedTransformer]] = {
        model_name: load_models(name=model_name) for model_name in model_names
    }

    saved_files: dict[str, str] = {}
    path = Path(save_path)

    for model_name, (autoencoder, hooked_model) in models.items():
        model_save_path = path / f"activations_{model_name}.h5"
        with h5py.File(model_save_path, "w") as h5f:
            group = h5f.create_group("activations")
            for feature in range(16_384):
                group.create_dataset(
                    f"feature_{feature}",
                    (0,),
                    maxshape=(None,),
                    dtype="float32",
                    compression="gzip",
                )

            for sample in tqdm(
                sample_loader, desc=f"Computing activations for {model_name}"
            ):
                input_ids: list[int] = sample["input_ids"]
                encoding = get_encodings(
                    input_ids=input_ids,
                    autoencoder=autoencoder,
                    hooked_target_model=hooked_model,
                )

                for feature in range(16_384):
                    activations = encoding[:, feature].tolist()
                    h5_dataset = group[f"feature_{feature}"]
                    current_length = h5_dataset.shape[0]
                    new_length = current_length + len(activations)
                    h5_dataset.resize((new_length,))
                    h5_dataset[current_length:new_length] = activations

        saved_files[model_name] = model_save_path.as_posix()

    return saved_files


def correlated_features(
    feature_activation_paths: dict[str, str],
    correlation_threshold: float,
    output_test_scatter: bool = True,
    sample_size: int | None = None,
    seed: int = 42,
) -> dict[tuple[str, str], float]:
    """Get list of pairs of highly correlated features.

    Args:
        feature_activation_paths (dict[str, str]): Feature activations file paths by model.
        correlation_threshold (float): How correlated we need features to be.
        output_test_scatter (bool, optional): Whether to output a test scatter.
            Defaults to True.

    Returns:
        dict[tuple[str, str], float]: Correlated feature names and how correlated they are.
    """
    if len(feature_activation_paths) > 2:
        raise ValueError("Can't currently handle more than two models here.")

    data: dict[str, list[float]] = {}

    for model, path in feature_activation_paths.items():
        feature_activations = load_feature_activations(
            file_path=path, sample_size=sample_size, seed=seed
        )
        for feature, feature_data in feature_activations["activations"].items():
            data[f"{model}/{feature}"] = feature_data

    df = pd.DataFrame(data)

    data_tensor = torch.tensor(df.values).float()
    data_tensor = data_tensor.to("cuda")

    mean = data_tensor.mean(0, keepdim=True)
    std_dev = data_tensor.std(0, unbiased=False, keepdim=True)
    normalized_data = (data_tensor - mean) / torch.where(
        std_dev == 0, torch.tensor(1.0).cuda(), std_dev
    )

    correlation_matrix = torch.mm(normalized_data.t(), normalized_data) / (
        data_tensor.shape[0] - 1
    )
    correlation_matrix = torch.clamp(correlation_matrix, -1, 1)

    torch.diagonal(correlation_matrix).fill_(0)
    high_corr_mask = correlation_matrix > correlation_threshold

    high_corr_indices = high_corr_mask.nonzero()
    high_corr_indices_cpu = high_corr_indices.cpu().numpy()

    pairs_with_correlation: dict[tuple[str, str], float] = {}

    for idx in high_corr_indices_cpu:
        row, col = idx

        name_a = df.columns[row]
        name_b = df.columns[col]
        corr_value = correlation_matrix[row, col].item()

        if (
            name_a == list(feature_activation_paths.keys())[0]
            and name_a.split("/")[0] != name_b.split("/")[0]
        ):
            pairs_with_correlation[(name_a, name_b)] = corr_value

    # Plot a wee sample.
    if output_test_scatter:
        test_a, test_b = list(pairs_with_correlation.keys())[0]
        plt.figure(figsize=(10, 6))
        plt.scatter(df[test_a], df[test_b], alpha=0.5)
        plt.title(
            f"{test_a} vs {test_b} - correlation {pairs_with_correlation[(test_a, test_b)]}"
        )
        plt.xlabel(test_a)
        plt.ylabel(test_b)
        plt.grid(True)
        plt.savefig("test.png")

    return pairs_with_correlation


def correlate_feature_activations(
    model_names: list[str],
    sample_loader: DataLoader,
    similarity_threshold: float,
    consistency_threshold: float,
):
    """Correlate features across models. See https://arxiv.org/pdf/1511.07543, section 3.1, 3.2.

    Notes:
        This is done us

    Args:
        model_names (list[str]): Names of models to correlate.
        sample_loader (DataLoader): DataLoader of samples with input IDs.

    Returns:
        list[dict[str, int]]:
            List of maps of feature traces, e.g. `[{'baby-python': 14688, 'tiny-stories': 15811, 'slerp': 15807}, ...]`
    """
    models: dict[str, tuple[Autoencoder, HookedTransformer]] = {
        model_name: load_models(name=model_name) for model_name in model_names
    }

    similarity_log = []
    min_diffs = []

    for sample in tqdm(sample_loader, desc="Computing similarities"):
        input_ids: list[int] = sample["input_ids"]

        # For each model, get the sparse encoding of for the sample sequence.
        model_encodings: dict[str, torch.Tensor] = {
            model_name: get_encodings(
                input_ids=input_ids,
                autoencoder=models[model_name][0],
                hooked_target_model=models[model_name][1],
            )
            for model_name in model_names
        }

        sample_similar_neurons = {}
        base_encodings = model_encodings[model_names[0]]

        # For each token activation.
        for feature in range(base_encodings.shape[1]):
            base_values = base_encodings[[i for i in range(len(input_ids))], feature]
            most_similar = {model_names[0]: feature}

            # Find the most similar activation in each other model.
            for model_name in model_names[1:]:
                other_encodings = model_encodings[model_name]

                differences = torch.abs(
                    normalize(other_encodings) - base_values.view(1, len(input_ids), 1)
                )
                aggregated_differences = differences.sum(dim=1)

                # Debugging ...
                min_diffs.append(aggregated_differences.min().item())

                if (
                    aggregated_differences.mean() < similarity_threshold
                    and base_values.sum() != 0
                ):
                    closest_row_index = torch.argmin(aggregated_differences)

                    most_similar[model_name] = closest_row_index.item()

            if len(most_similar) > 1:
                sample_similar_neurons[feature] = most_similar

        similarity_log.append(sample_similar_neurons)

    similar_features: list[dict[str, int]] = []

    inconsistency_list = []
    inconsistency_map = {}

    for feature, original_family in similarity_log[0].items():
        # Mapping family member to inconsistency count.
        inconsistent_members: dict[str, int] = {}
        consistent_group = original_family.copy()

        for entry in similarity_log[1:]:
            for model, feature in dict(
                set(entry.get(feature, {}).items()) - set(original_family.items())
            ).items():
                inconsistent_members[model] = inconsistent_members.get(model, 0) + 1

        for member, inconsistency in inconsistent_members.items():
            inconsistency_list.append(inconsistency)
            inconsistency_map[member] = inconsistency_map.get(member, 0) + inconsistency
            if member in consistent_group and inconsistency / len(
                similarity_log[1:]
            ) > (1 - consistency_threshold):
                del consistent_group[member]

        if len(consistent_group) > 1:
            similar_features.append(consistent_group)

    return similar_features


def correlate(file_path: str) -> None:
    comparison_config = ComparisonConfig.from_toml_path(file_path=file_path)

    _, dataset = datasplit_from_dataset_config(
        dataset_config=comparison_config.dataset_config,
        training_config=Accelerator(),
    )

    sample_loader: DataLoader = get_uniform_sample_loader(
        dataset,
        comparison_config.sample_size,
        batch_size=1,
    )

    # Output path for small files.
    output_path: Path = Path(comparison_config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Recreate output path in place with more space.
    output_path_with_space: Path = (
        Path(PATH_WITH_ENOUGH_SPACE) / comparison_config.output_path
    )
    output_path_with_space.parent.mkdir(parents=True, exist_ok=True)
    output_path_with_space = output_path_with_space.parent

    if not os.getenv("ONLY_CORRELATION"):
        activation_paths: dict[str, str] = save_feature_activations(
            model_names=comparison_config.model_names[:2],
            sample_loader=sample_loader,
            save_path=(output_path_with_space),
        )

    pairs_and_how_correlated_they_are: dict[tuple[str, str], float] = (
        correlated_features(
            feature_activation_paths={
                model: output_path_with_space / f"activations_{model}.h5"
                for model in comparison_config.model_names[:2]
            },
            correlation_threshold=comparison_config.correlation_threshold,
            sample_size=50000,
        )
    )

    pairs_data: dict[str, float] = {
        f"{a}->{b}": correlation
        for (a, b), correlation in pairs_and_how_correlated_they_are.items()
    }
    (output_path.parent / "pairs_and_how_correlated_they_are.json").open("w+").write(
        json.dumps(pairs_data)
    )

    similar_features: list[dict[str, int]] = correlate_feature_activations(
        model_names=comparison_config.model_names,
        sample_loader=sample_loader,
        similarity_threshold=comparison_config.similarity_threshold,
        consistency_threshold=comparison_config.consistency_threshold,
    )

    output_path.open("w+").write(json.dumps(similar_features))


if __name__ == "__main__":
    typer.run(correlate)
