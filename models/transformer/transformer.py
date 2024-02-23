"""GPT-like architecture."""

import einops
import torch
import torch.nn as nn
import xformers.ops as xops

from torch.nn import functional as F
from models.transformer.encoding import RotaryPositionalEncoding


class Attention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        self.qkv_transform = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.out_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_dim, sequence_dim, _ = x.size()

        chunks = self.qkv_transform(x).chunk(3, dim=-1)
        query, key, value = (
            einops.rearrange(chunk, "b l (head k) -> b head l k", head=self.num_heads)
            for chunk in chunks
        )

        attention_output = xops.memory_efficient_attention(
            query, key, value,
            attn_bias=xops.LowerTriangularMask()
        )
        attention_output = einops.rearrange(attention_output, "1 s h d -> s (h d)", h=self.n_heads)
        return self.out_transform(attention_output)

        # Scaled Dot-Product Attention
        # scaling_factor = query.size(-1) ** 0.5
        # attention_scores = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor
        # attention_scores = F.softmax(attention_scores, dim=-1)
        # attention_output = attention_scores @ value

        # # Rearrange and transform the output
        # attention_output = einops.rearrange(
        #     attention_output, "b head l k -> b l (head k)"
        # )
        # return self.out_transform(attention_output)


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.linear_in = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear_hidden = nn.Linear(hidden_dim, input_dim, bias=False)
        self.linear_gate = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.linear_in(x))
        gate = self.linear_gate(hidden)
        return self.linear_hidden(hidden * gate)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon)
        return x * norm * self.weight


class TransformerBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        hidden_dim: int,
        norm_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attention = Attention(embedding_dim=embedding_dim, num_heads=num_heads)
        self.attention_norm = RMSNorm(dim=embedding_dim, epsilon=norm_epsilon)
        self.feed_forward_norm = RMSNorm(dim=embedding_dim, epsilon=norm_epsilon)
        self.feed_forward = FeedForward(input_dim=embedding_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.attention(self.attention_norm(x))
        after_attention = x + attention
        return after_attention + self.feed_forward(
            self.feed_forward_norm(after_attention)
        )


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_dim: int,
        embedding_dim: int,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        block_hidden_dim: int = 0,
    ) -> None:
        super().__init__()

        block_hidden_dim = block_hidden_dim or embedding_dim

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.positional_encoding = RotaryPositionalEncoding(
            embedding_dim=embedding_dim, max_seq_len=max_seq_len
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    hidden_dim=block_hidden_dim,
                    num_heads=num_heads,
                )
                for i in range(num_layers)
            ]
        )
        self.feed_forward = nn.Linear(embedding_dim, vocab_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.positional_encoding(self.embedding(x))
        for block in self.blocks:
            x = block(x)
        x = self.feed_forward(x)
        return F.log_softmax(x, dim=-1)
