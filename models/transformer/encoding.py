"""Positional encodings."""

import torch
import torch.nn as nn


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_seq_len: int) -> None:
        super().__init__()

        self.positional_factor = 1.0 / (
            10000 ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim)
        )

        positions = torch.arange(max_seq_len).unsqueeze(1).float()
        scaled_positions = positions * self.positional_factor

        embeddings = torch.cat(
            [torch.sin(scaled_positions), torch.cos(scaled_positions)], dim=1
        )

        self.register_buffer("embeddings", embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.embeddings[: x.size(1)]
