import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from .embedding import Embedding
from .linear import Linear
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float = 10000.0,
        attn_pdrop: float = 0.0,
        residual_pdrop: float = 0.0,
    ) -> None:
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)

        max_seq_len = max_seq_len if max_seq_len is not None else context_length
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    theta=theta,
                    attn_pdrop=attn_pdrop,
                    residual_pdrop=residual_pdrop,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model)

        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: Int[Tensor, "batch_size seq_len"],
    ) -> Float[Tensor, "batch_size seq_len vocab_size"]:
        x = self.token_embeddings(input_ids)

        for block in self.layers:
            x = block(x)

        x = self.ln_final(x)

        logits = self.lm_head(x)

        return logits
