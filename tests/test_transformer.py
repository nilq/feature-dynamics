
from models.transformer import Transformer
import torch


def test_transformer() -> None:
    # Parameters for the transformer.
    vocab_dim = 8
    embedding_dim = 16
    max_seq_len = 256
    num_layers = 1
    num_heads = 2

    # Initialize tiny tiny transformer model.
    tinyformer = Transformer(
        vocab_dim=vocab_dim,
        embedding_dim=embedding_dim,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    sample_input = torch.randint(low=0, high=vocab_dim, size=(1, max_seq_len))
    output = tinyformer(sample_input)

    assert output.shape == (1, max_seq_len, vocab_dim), "Output shape is incorrect."
