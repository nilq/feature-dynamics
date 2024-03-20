"""Test reconstruction loss."""

from models.sparse_autoencoder.model import Autoencoder, AutoencoderConfig
from models.sparse_autoencoder.utils import hooked_model_fixed
from training.autoencoder.loss import get_reconstruction_loss


def test_loss_doesnt_explode() -> None:
    """Test that loss runs without crashing, shapes work etc."""
    model = hooked_model_fixed("nilq/mistral-1L-tiny")
    encoder = Autoencoder(AutoencoderConfig(input_size=1024, hidden_size=1024 * 30))

    assert get_reconstruction_loss(
        model, encoder, ["hello, this is a test"], "blocks.0.mlp.hook_post"
    ), "Unfortunately the reconstruction loss broke."
