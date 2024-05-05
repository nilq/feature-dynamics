"""Like correlate, but on the fly."""

from tqdm import tqdm
import typer
import torch
import datasets

from torch.nn.functional import normalize
from evaluation.autoencoder.evaluate import *
from pathlib import Path
from torch.utils.data import DataLoader
from training.transformer.data import (
    datasplit_from_dataset_config,
    tokenizer_from_dataset_config,
)
from training.autoencoder.train import get_uniform_sample_loader
from evaluation.autoencoder.config import ComparisonConfig

# To load corresponding autoencoder.
grandparent_dir: Path = Path(__file__).parent / ".." / ".."
model_name_to_config: dict[str, Path] = {
    "baby-python": grandparent_dir / "experiments/evaluation/autoencoder/baby_python_mistral_1L_tiny_base.toml",
    "lua": grandparent_dir / "experiments/evaluation/autoencoder/baby-python-mistral-1L-tiny-lua-ft/baby_python.toml",
    "tiny-stories": grandparent_dir / "experiments/evaluation/autoencoder/baby-python-mistral-1L-tiny-TinyStories-ft/baby_python.toml",
    "slerp": grandparent_dir / "experiments/evaluation/autoencoder/baby-python-mistral-1L-tiny-lua-stories-slerp/baby_python.toml"
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


def get_encodings(input_ids: list[int], autoencoder: Autoencoder, hooked_target_model: HookedTransformer) -> torch.Tensor:
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


def correlate_feature_activations(model_names: list[str], sample_loader: DataLoader, similarity_threshold: float, consistency_threshold: float):
    """Correlate features across models.

    Args:
        model_names (list[str]): Names of models to correlate.
        sample_loader (DataLoader): DataLoader of samples with input IDs.

    Returns:
        list[dict[str, int]]:
            List of maps of feature traces, e.g. `[{'baby-python': 14688, 'tiny-stories': 15811, 'slerp': 15807}, ...]`
    """
    models: dict[str, tuple[Autoencoder, HookedTransformer]] = {
        model_name: load_models(name=model_name)
        for model_name in model_names
    }

    similarity_log = []
    min_diffs = []

    for sample in tqdm(sample_loader, desc="Computing similarities"):
        input_ids: list[int] = sample["input_ids"]
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

        for feature in range(base_encodings.shape[1]):
            base_values = base_encodings[[i for i in range(len(input_ids))], feature]
            most_similar = { model_names[0]: feature }

            for model_name in model_names[1:]:
                other_encodings = model_encodings[model_name]

                base_values = base_encodings[[i for i in range(len(input_ids))], feature]
                differences = torch.abs(normalize(other_encodings) - base_values.view(1, len(input_ids), 1))
                aggregated_differences = differences.sum(dim=1)

                min_diffs.append(aggregated_differences.min().item())
                if aggregated_differences.mean() < similarity_threshold and base_values.sum() != 0:
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
            for model, feature in dict(set(entry.get(feature, {}).items()) - set(original_family.items())).items():
                inconsistent_members[model] = inconsistent_members.get(model, 0) + 1

        for member, inconsistency in inconsistent_members.items():
            inconsistency_list.append(inconsistency)
            inconsistency_map[member] = inconsistency_map.get(member, 0) + inconsistency
            if member in consistent_group and inconsistency / len(similarity_log[1:]) > (1 - consistency_threshold):
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

    similar_features: list[dict[str, int]] = correlate_feature_activations(
        model_names=comparison_config.model_names,
        sample_loader=sample_loader,
        similarity_threshold=comparison_config.similarity_threshold,
        consistency_threshold=comparison_config.consistency_threshold
    )


if __name__ == "__main__":
    typer.run(correlate)
