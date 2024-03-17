"""Activation data."""

import torch
from torch.utils.data import Dataset, DataLoader
from transformer_lens import HookedTransformer

from datasets import load_dataset

from models.sparse_autoencoder.utils import get_model_activations


class ActivationDataset(Dataset):
    """Model activation dataset."""

    def __init__(
        self,
        dataset_name: str,
        dataset_text_column: str,
        dataset_split: str,
        model: HookedTransformer,
        target_layer: int,
        target_activation_name: str,
    ) -> None:
        """Initialise with model/layer/activation target and source text dataset.

        Args:
            dataset_name (str): Name of HuggingFace text dataset to use.
            dataset_text_column (str): Name of text column in dataset.
            dataset_split (str): Which dataset split to use.
            model (HookedTransformer): Hooked model to get activations from.
            target_layer (int): Target model layer index.
            target_activation_name (str): Target layer activation name.
        """
        super().__init__()

        self.text_dataset = load_dataset(dataset_name, split=dataset_split)[
            dataset_text_column
        ]

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
        text: str = self.text_dataset[index]
        activations = get_model_activations(
            model=self.model,
            model_input=text,
            layer=self.target_layer,
            activation_name=self.target_activation_name,
        )

        return activations
