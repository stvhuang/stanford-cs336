import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
        use_rope: bool = True,
    ) -> None:
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if use_rope:
            assert theta is not None
            assert max_seq_len is not None

            self.rope = RotaryPositionalEmbedding(
                theta,
                self.d_k,
                max_seq_len,
                device=device,
            )
        else:
            self.rope = None

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        use_causal_mask: bool = True,
    ) -> Float[Tensor, "... seq_len d_model"]:
        *batch_dims, seq_len, d_model = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(*batch_dims, seq_len, self.num_heads, self.d_k)
        Q = Q.transpose(-3, -2)

        K = K.view(*batch_dims, seq_len, self.num_heads, self.d_k)
        K = K.transpose(-3, -2)

        V = V.view(*batch_dims, seq_len, self.num_heads, self.d_v)
        V = V.transpose(-3, -2)

        token_positions = torch.arange(seq_len, device=x.device)

        for _ in range(len(batch_dims) + 1):  # +1 for num_heads
            token_positions = token_positions.unsqueeze(0)

        token_positions = token_positions.expand(*batch_dims, self.num_heads, seq_len)

        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = None

        if use_causal_mask:
            mask = torch.tril(
                torch.ones(
                    seq_len,
                    seq_len,
                    dtype=torch.bool,
                    device=x.device,
                )
            )

            for _ in range(len(batch_dims) + 1):
                mask = mask.unsqueeze(0)

            mask = mask.expand(*batch_dims, self.num_heads, seq_len, seq_len)

        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(-3, -2)
        attn_output = attn_output.reshape(*batch_dims, seq_len, self.d_model)

        output = self.output_proj(attn_output)

        return output
