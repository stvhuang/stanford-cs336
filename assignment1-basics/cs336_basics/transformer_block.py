import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .multihead_self_attention import MultiHeadSelfAttention
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
        attn_pdrop: float = 0.0,
        residual_pdrop: float = 0.0,
    ) -> None:
        super().__init__()

        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
        )

        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

        self.residual_dropout = nn.Dropout(residual_pdrop) if residual_pdrop > 0.0 else None

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        use_causal_mask: bool = True,
    ) -> Float[Tensor, "... seq_len d_model"]:
        normed = self.ln1(x)
        attn_out = self.attn(normed, use_causal_mask=use_causal_mask)

        if self.residual_dropout is not None:
            attn_out = self.residual_dropout(attn_out)

        z = x + attn_out

        normed = self.ln2(z)
        ff_out = self.ffn(normed)

        if self.residual_dropout is not None:
            ff_out = self.residual_dropout(ff_out)

        y = z + ff_out

        return y
