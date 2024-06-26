"""Activation data."""

from __future__ import annotations
from copy import copy

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataset import random_split
from transformer_lens import HookedTransformer

from datasets import load_dataset
from accelerate import Accelerator

from models.sparse_autoencoder.utils import get_model_activations
from training.transformer.config import DatasetConfig
from training.transformer.data import datasplit_from_dataset_config


def split_dataset(
    dataset: ActivationDataset, validation_percentage: float
) -> tuple[Dataset, Dataset]:
    """Split a dataset into training and validation sets.

    Args:
        dataset (Dataset): The dataset to split.
        validation_percentage (float): The fraction of the dataset to allocate to the validation set.

    Returns:
        tuple[Dataset, Dataset]: Training and validation datasets.
    """
    # Calculate the number of samples to include in each set
    total_size = len(dataset.text_dataset)
    val_size = int(total_size * validation_percentage)
    train_size = total_size - val_size

    train_indices, val_indices = random_split(range(total_size), [train_size, val_size])

    train_dataset = copy(dataset)
    validation_dataset = copy(dataset)

    train_dataset.text_dataset = Subset(dataset.text_dataset, train_indices.indices)
    validation_dataset.text_dataset = Subset(dataset.text_dataset, val_indices.indices)

    return train_dataset, validation_dataset


class ActivationDataset(Dataset):
    """Model activation dataset."""

    def __init__(
        self,
        dataset_id: str,
        dataset_text_key: str,
        dataset_split: str,
        model: HookedTransformer,
        target_layer: int,
        target_activation_name: str,
        tokenizer_id: str,
        dtype: str = "bfloat32",
        block_size: int = 1024,
    ) -> None:
        """Initialise with model/layer/activation target and source text dataset.

        Args:
            dataset_id (str): Name of HuggingFace text dataset to use.
            dataset_text_key (str): Name of text column in dataset.
            dataset_split (str): Which dataset split to use.
            model (HookedTransformer): Hooked model to get activations from.
            target_layer (int): Target model layer index.
            target_activation_name (str): Target layer activation name.
        """
        super().__init__()

        # self.text_dataset = load_dataset(dataset_id, split=dataset_split)[
        #     dataset_text_key
        # ]
        self.text_dataset = datasplit_from_dataset_config(
            DatasetConfig(
                dataset_id=dataset_id,
                dataset_text_key=dataset_text_key,
                tokenizer_id=tokenizer_id,
                block_size=block_size,
            ),
            training_config=Accelerator(),
        )[0 if dataset_split == "train" else 1]

        self.model = model
        self.target_layer = target_layer
        self.target_activation_name = target_activation_name

    def __len__(self) -> int:
        """Get length of dataset.

        Returns:
            int: Length of text dataset.
        """
        return len(self.text_dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get item.

        Args:
            index (int): Item index.

        Returns:
            torch.Tensor:
                Activations of target layer activation for text data at index.
        """
        text: str = torch.tensor(
            self.text_dataset[index]["input_ids"], device=self.model.cfg.device
        ).unsqueeze(0)

        activations = get_model_activations(
            model=self.model,
            model_input=text,
            layer=self.target_layer,
            activation_name=self.target_activation_name,
        )

        return activations
