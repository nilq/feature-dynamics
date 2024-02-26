"""Book keeping of artifacts."""

import wandb
import tempfile
import numpy as np

from models.transformer import Transformer, TransformerBlock


def transformer_block_mlp_weights(block: TransformerBlock) -> dict[str, np.ndarray]:
    return {
        "linear_in": block.feed_forward.linear_in.weight.cpu().detach().numpy(),
        "linear_hidden": block.feed_forward.linear_hidden.weight.cpu().detach().numpy(),
        "linear_gate": block.feed_forward.linear_gate.weight.cpu().detach().numpy(),
    }


def transformer_block_qkv_weights(block: TransformerBlock) -> dict[str, np.ndarray]:
    qkv_weights = block.attention.qkv_transform.weight.cpu().detach().numpy()
    head_dim = block.attention.embedding_dim // block.attention.num_heads
    qkv_weights = qkv_weights.reshape(-1, 3, block.attention.num_heads, head_dim)

    return {
        "query_weights": qkv_weights[:, 0, :, :],
        "key_weights": qkv_weights[:, 1, :, :],
        "value_weights": qkv_weights[:, 2, :, :],
    }


def wandb_save_transformer_states(model: Transformer, epoch: int) -> None:
    """Save model states.

    Args:
        model (Transformer): Model to save states of.
        epoch (int): Epoch of states.
    """
    for i, block in enumerate(model.blocks):
        # Save MLP weights of block.
        mlp_weights = transformer_block_mlp_weights(block)
        mlp_artifact = wandb.Artifact(
            f"mlp_weights_epoch_{epoch}_block_{i}", type="model_weights"
        )
        for layer, weights in mlp_weights.items():
            with tempfile.NamedTemporaryFile(delete=True, suffix=".npy") as tmp_file:
                np.save(tmp_file, weights)
                mlp_artifact.add_file(
                    tmp_file.name, name=f"mlp_{layer}_epoch_{epoch}_block_{i}.npy"
                )

        wandb.log_artifact(mlp_artifact)

        # Save QKV weights of block.
        qkv_weights = transformer_block_qkv_weights(block)
        qkv_artifact = wandb.Artifact(
            f"qkv_weights_epoch_{epoch}_block_{i}", type="model_weights"
        )
        for layer, weights in qkv_weights.items():
            with tempfile.NamedTemporaryFile(delete=True, suffix=".npy") as tmp_file:
                np.save(tmp_file, weights)
                qkv_artifact.add_file(
                    tmp_file.name, name=f"qkv_{layer}_epoch_{epoch}_block_{i}.npy"
                )

        wandb.log_artifact(qkv_artifact)
