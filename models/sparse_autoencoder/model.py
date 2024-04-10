"""Sparse autoencoder model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig, AutoConfig
from torch.nn.init import kaiming_uniform_

from models.sparse_autoencoder.geometric_median import geometric_median


def autoencoder_loss(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
    latent_activations: torch.Tensor,
    l1_weight: float,
) -> torch.Tensor:
    return (
        normalized_mean_squared_error(reconstruction, original_input)
        + normalized_L1_loss(latent_activations, original_input) * l1_weight
    )


def normalized_mean_squared_error(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    return (
        ((reconstruction - original_input) ** 2).mean(dim=1)
        / (original_input**2).mean(dim=1)
    ).mean()


def normalized_L1_loss(
    latent_activations: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    return (latent_activations.abs().sum(dim=1) / original_input.norm(dim=1)).mean()


class AutoencoderConfig(PretrainedConfig):
    """Sparse autoencoder HuggingFace config."""

    model_type = "autoencoder"

    def __init__(
        self,
        hidden_size=128,
        input_size=1024,
        activation_type="relu",
        tied=False,
        l1_coefficient: int = 0.1,
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.activation_type = activation_type
        self.l1_coefficient = l1_coefficient
        self.tied = tied
        self.torch_dtype = (
            torch_dtype
            if torch_dtype in ["auto", None]
            else getattr(torch, torch_dtype)
        )


class Autoencoder(PreTrainedModel):
    """Autoencoder model, HuggingFace-ready."""

    config_class = AutoencoderConfig

    def __init__(self, config: AutoencoderConfig):
        super().__init__(config)
        self.config = config

        # self.pre_bias = nn.Parameter(
        #     torch.zeros(config.input_size, dtype=config.torch_dtype)
        # )
        self.decoder_bias = nn.Parameter(
            torch.zeros(config.input_size, dtype=config.torch_dtype)
        )

        self.encoder = nn.Linear(
            config.input_size, config.hidden_size, bias=False, dtype=config.torch_dtype
        )

        self.latent_bias = nn.Parameter(
            torch.zeros(config.hidden_size, dtype=config.torch_dtype)
        )

        self.l1_coefficient = config.l1_coefficient

        kaiming_uniform_(self.encoder.weight, nonlinearity="relu")

        # TODO: Add more maybe.
        if config.activation_type == "relu":
            self.activation = nn.ReLU()

        if config.tied:
            self.decoder = TiedTranspose(self.encoder)
        else:
            self.decoder = nn.Linear(
                config.hidden_size,
                config.input_size,
                bias=False,
                dtype=config.torch_dtype,
            )
            kaiming_uniform_(self.decoder.weight, nonlinearity="relu")

    @torch.no_grad()
    def initialise_decoder_bias_with_geometric_median(
        self, all_activations: torch.Tensor
    ):
        previous_decoder_bias = self.decoder_bias.clone().to(all_activations.device)
        median: torch.Tensor = geometric_median(all_activations)

        previous_distances = torch.norm(
            all_activations - previous_decoder_bias.unsqueeze(0), dim=1
        )
        distances = torch.norm(all_activations - median.unsqueeze(0), dim=1)

        print("Reinitializing decoder bias with geometric median of activations")
        print(f"Previous distances: {previous_distances.median().item()}")
        print(f"New distances: {distances.median().item()}")

        self.decoder_bias.data = median.to(
            self.decoder_bias.device, dtype=self.decoder_bias.dtype
        )

    @torch.no_grad()
    def remove_gradients_parallel_to_decoder_directions(self):
        parallel_component = (self.decoder.weight.grad * self.decoder.weight.data).sum(
            dim=1
        )
        expanded_parallel_component = parallel_component.unsqueeze(1).expand_as(
            self.decoder.weight.grad
        )
        self.decoder.weight.grad -= (
            expanded_parallel_component * self.decoder.weight.data
        )

    @torch.no_grad()
    def make_decoder_weights_and_gradient_unit_norm(self) -> None:
        """Make weights and gradients unit norm."""
        norm_decoder_weights = self.decoder.weight / self.decoder.weight.norm(
            dim=-1, keepdim=True
        )
        norm_decoder_gradient_proj = (
            self.decoder.weight.grad * norm_decoder_weights
        ).sum(dim=-1, keepdim=True) * norm_decoder_weights

        self.decoder.weight.grad -= norm_decoder_gradient_proj
        self.decoder.weight.data = norm_decoder_weights

    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pre-activation.

        Args:
            x (torch.Tensor): Input data (shape: [batch, inputs]).

        Returns:
            torch.Tensor: Autonecoder latents before activations.
        """
        # x = x - self.decoder_bias
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
        return self.decoder(latents) + self.decoder_bias

    def forward(
        self,
        x: torch.Tensor,
        use_ghost_gradients: bool,
        ghost_gradient_neuron_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input data (shape: [batch, input_size...]).
            use_ghost_gradients (bool): Whether to use ghost gradients.
            ghost_gradient_neuron_mask (torch.Tensor | None, optional):
                Dead neuron mask, required for ghost gradients.
                Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Loss, i.e. L2 + L1 + ghost residual if set.
                - Reconstructed data (shape: [batch, input_size]).
                - Autoencoder latents (shape: [batch, hidden_size]).
                - L2 loss.
                - L1 loss.
        """
        latents_pre_act = self.encode_pre_act(x)
        latents = self.encode(x)
        reconstructed = self.decode(latents)

        l2_loss_ghost_residual: torch.Tensor = torch.tensor(
            0.0, dtype=self.dtype, device=self.device
        )

        # Ghost gradient protocol, helps keep neurons alive.
        # https://transformer-circuits.pub/2024/jan-update/index.html#dict-learning-resampling
        if (
            self.training
            and use_ghost_gradients
            and ghost_gradient_neuron_mask
            and ghost_gradient_neuron_mask.sum() > 0
        ):
            residual: torch.Tensor = x - reconstructed
            residual_centered: torch.Tensor = residual - residual.mean(
                dim=0, keepdim=True
            )
            l2_norm_residual: torch.Tensor = torch.norm(residual, dim=-1)

            dead_neuron_feature_activations = torch.exp(
                latents_pre_act[:, ghost_gradient_neuron_mask]
            )
            ghosts = (
                dead_neuron_feature_activations
                @ self.decoder.weight.data[ghost_gradient_neuron_mask, :]
            )
            l2_norm_ghosts = torch.norm(ghosts, dim=-1)
            norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghosts * 2)
            ghosts *= norm_scaling_factor[:, None].detach()

            l2_loss_ghost_residual = (ghosts.float() - residual.float()).pow(2) / (
                residual_centered.detach() ** 2
            ).sum(-1, keepdim=True).sqrt()
            l2_rescaling_factor = (l2_loss / (l2_loss_ghost_residual + 1e-6)).detach()
            l2_loss_ghost_residual = (
                l2_rescaling_factor * l2_loss_ghost_residual
            ).mean()

        # NOTE: Non-normalised old L2.
        # l2_loss = (reconstructed.float() - x.float()).pow(2).sum(-1).mean()

        x_centered = x - x.mean(dim=0, keepdim=True)
        l2_loss = (
            torch.pow((reconstructed - x.float()), 2)
            / x_centered.norm(dim=-1, keepdim=True)
        ).mean()
        # (x_centered ** 2).sum(dim=-1, keepdim=True).sqrt()).mean()

        sparsity = latents.norm(p=1.0, dim=1).mean()
        l1_loss = self.l1_coefficient * sparsity  # (latents.float().abs().sum())
        loss = l2_loss + l1_loss + l2_loss_ghost_residual

        return loss, reconstructed, latents, l2_loss, l1_loss


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
