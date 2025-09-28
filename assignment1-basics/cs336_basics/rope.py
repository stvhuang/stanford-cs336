import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ) -> None:
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        dim_indices = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        freqs = theta ** (-dim_indices / d_k)

        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)

        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)

        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)

        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        cos_cached: Tensor = self.cos_cached
        sin_cached: Tensor = self.sin_cached

        original_shape = token_positions.shape
        flat_positions = token_positions.flatten()

        cos = cos_cached[flat_positions].reshape(*original_shape, -1)
        sin = sin_cached[flat_positions].reshape(*original_shape, -1)

        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)

        x_even = x_reshaped[..., 0]
        x_odd = x_reshaped[..., 1]

        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated = x_even * sin + x_odd * cos

        x_rotated = torch.stack([x_even_rotated, x_odd_rotated], dim=-1)

        return x_rotated.reshape(*x.shape)
