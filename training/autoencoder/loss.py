"""Reconstruction loss etc. (https://github.com/neelnanda-io/1L-Sparse-Autoencoder/blob/main/utils.py)"""

import torch


from torch.utils.data import DataLoader
from models.sparse_autoencoder.utils import get_model_activations
from training.autoencoder.data import ActivationDataset
from models.sparse_autoencoder.model import Autoencoder
from functools import partial
from transformer_lens import HookedTransformer


def replacement_hook(mlp_post, hook, encoder: Autoencoder):
    mlp_post_reconstr = encoder(mlp_post)[1]
    return mlp_post_reconstr


def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post


def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.0
    return mlp_post


@torch.no_grad()
def get_reconstruction_loss(
    model, encoder, samples: list[str], target_activation_name: str
) -> tuple[float, float, float, float]:
    loss_list = []

    for text in samples:
        loss = model(text, return_type="loss")
        recons_loss = model.run_with_hooks(
            text,
            return_type="loss",
            fwd_hooks=[
                (
                    target_activation_name,
                    partial(replacement_hook, encoder=encoder),
                )
            ],
        )
        zero_abl_loss = model.run_with_hooks(
            text,
            return_type="loss",
            fwd_hooks=[(target_activation_name, zero_ablate_hook)],
        )
        loss_list.append((loss, recons_loss, zero_abl_loss))

    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()
    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)

    return score, loss, recons_loss, zero_abl_loss
