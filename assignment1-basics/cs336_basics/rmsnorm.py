import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(
        self,
        x: Float[Tensor, "... d_model"],
    ) -> Float[Tensor, "... d_model"]:
        x_f32 = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x_f32**2, dim=-1, keepdim=True) + self.eps)

        x_normalized = (x_f32 / rms) * self.weight

        output = x_normalized.to(x.dtype)

        return output
