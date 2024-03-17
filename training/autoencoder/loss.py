"""Reconstruction loss etc. (https://github.com/neelnanda-io/1L-Sparse-Autoencoder/blob/main/utils.py)"""

import torch
from models.sparse_autoencoder.model import Autoencoder


def replacement_hook(mlp_post, hook, encoder: Autoencoder):
    mlp_post_reconstr = encoder(mlp_post)[1]
    return mlp_post_reconstr

def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post

def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.
    return mlp_post


@torch.no_grad()
def reconstruction_loss(encoder: Autoencoder) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """_summary_

    Args:
        encoder (Autoencoder): _description_

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: _description_
    """
    losses = []
