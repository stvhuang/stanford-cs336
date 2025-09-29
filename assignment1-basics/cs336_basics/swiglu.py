import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .linear import Linear
from .silu import silu


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... d_model"],
    ) -> Float[Tensor, "... d_model"]:
        silu_w1x = silu(self.w1(x))
        gated_w3x = silu_w1x * self.w3(x)
        output = self.w2(gated_w3x)

        return output
