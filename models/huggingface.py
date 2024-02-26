"""HuggingFace wrappers."""

import torch

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

from models.transformer import Transformer


class TransformerConfig(PretrainedConfig):
    """Pretty generic transformer config."""

    model_type = "transformer"

    def __init__(
        self,
        vocab_dim: int = 20001,
        embedding_dim: int = 256,
        max_seq_len: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        block_hidden_dim: int = 256,
        **kwargs,
    ) -> None:
        self.vocab_dim = vocab_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.block_hidden_dim = block_hidden_dim

        super().__init__(**kwargs)


class TransformerModel(PreTrainedModel):
    config_class = TransformerConfig

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)

        self.transformer = Transformer(
            vocab_dim=config.vocab_dim,
            embedding_dim=config.embedding_dim,
            max_seq_len=config.max_seq_len,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
        )

    def forward(
        self,
        input_ids,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
    ) -> CausalLMOutput:
        logits, hidden_states, attentions = self.transformer(
            input_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        loss = None
        if labels:
            loss = torch.nn.CrossEntropyLoss()(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )
