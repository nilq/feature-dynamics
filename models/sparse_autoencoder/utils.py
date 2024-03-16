"""Sparse autoencoder utilities."""

import torch

from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def hooked_model_fixed(model_name: str) -> HookedTransformer:
    """Get hooked model in a way that doesn't explode.

    Args:
        model_name (str): Name/ID of HuggingFace model.

    Returns:
        HookedTransformer: Hooked transformer.
    """
    model = HookedTransformer.from_pretrained(model_name, center_writing_weights=False)
    vanilla_tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.tokenizer = vanilla_tokenizer
    
    # Padding is not needed, but TransformerLens thinks it is. We play along.
    model.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return model


@torch.no_grad()
def get_model_activations(
    model: HookedTransformer,
    tokens: list[int],
    layer: int,
    activation_name: str,
    batch_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get model activations at given layer.

    Args:
        model (HookedTransformer): Hooked transformer model.
        tokens (list[int]): Token IDs to get activations for.
        layer (int): Index of layer.
        activation_name (str): Name of activations to get.
        batch_size (int, optional): Batch size of subsampled activations.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Subsampled activations, all activations.
    """
    _, cache = model.run_with_cache(tokens, stop_at_layer=layer + 1, names_filter=activation_name)

    activations = cache[activation_name]
    activations = activations.reshape(-1, activations.shape[-1])

    return activations
