"""Sparse autoencoder model."""

from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoencoderConfig(PretrainedConfig):
    """Sparse autoencoder HuggingFace config."""
    model_type = "autoencoder"
    def __init__(self, hidden_size=128, input_size=1024, activation_type="relu", tied=False, l1_coefficient: int = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.activation_type = activation_type
        self.l1_coefficient = l1_coefficient
        self.tied = tied


class Autoencoder(PreTrainedModel):
    """Autoencoder model, HuggingFace-ready."""
    config_class = AutoencoderConfig

    def __init__(self, config: AutoencoderConfig):
        super().__init__(config)
        self.config = config
        self.pre_bias = nn.Parameter(torch.zeros(config.input_size))
        self.encoder = nn.Linear(config.input_size, config.hidden_size, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(config.hidden_size))
        self.l1_coefficient = config.l1_coefficient

        # TODO: Add more maybe.
        if config.activation_type == "relu":
            self.activation = nn.ReLU()

        if config.tied:
            self.decoder = TiedTranspose(self.encoder)
        else:
            self.decoder = nn.Linear(config.hidden_size, config.input_size, bias=False)

        self.register_buffer("stats_last_nonzero", torch.zeros(config.hidden_size, dtype=torch.long))

    def make_decoder_weights_and_gradient_unit_norm(self) -> None:
        """Make weights and gradients unit norm."""
        norm_decoder_weights = self.decoder.weight / self.decoder.woight.norm(dim=-1, keepdim=True)
        norm_decoder_gradient_proj = (self.decoder.weight.grad * norm_decoder_weights).sum(dim=-1, keepdim=True) * norm_decoder_weights

        self.decoder.weight.grad -= norm_decoder_gradient_proj
        self.decoder.weight.data = norm_decoder_weights

    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pre-activation.

        Args:
            x (torch.Tensor): Input data (shape: [batch, inputs]).

        Returns:
            torch.Tensor: Autonecoder latents before activations.
        """
        x = x - self.pre_bias
        return self.encoder(x) + self.latent_bias

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations.

        Args:
            x (torch.Tensor): Input data (shape: [batch, inputs]).

        Returns:
            torch.Tensor: Autoencoder latents.
        """
        return self.activation(self.encode_pre_act(x))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents.

        Args:
            latents (torch.Tensor): Autoencoder latents (shape: [batch, hidden_size]).

        Returns:
            torch.Tensor: Reconstructed data (shape: [batch, input_size]).
        """
        return self.decoder(latents) + self.pre_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input data (shape: [batch, input_size]).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Loss, i.e. L2 + L1.
                - Autoencoder latents pre-activation (shape: [batch, hidden_size]).
                - Autoencoder latents (shape: [batch, hidden_size]).
                - Reconstructed data (shape: [batch, input_size]).
        """
        latents_pre_act = self.encode_pre_act(x)
        latents = self.encode(x)
        reconstructed = self.decode(latents)

        self.stats_last_nonzero *= (latents == 0).all(dim=0).long()
        self.stats_last_nonzero += 1

        l2_loss = (reconstructed.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coefficient * (latents.float().abs().sum())
        loss = l2_loss + l1_loss

        return loss, latents_pre_act, latents, reconstructed


class TiedTranspose(nn.Module):
    """Tied transpose module."""
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None, "Don't want any bias here."
        return F.linear(x, self.linear.weight.t())

    @property
    def weight(self):
        return self.linear.weight.t()

    @property
    def bias(self):
        # TiedTranspose layers do not have a separate bias.
        return None
