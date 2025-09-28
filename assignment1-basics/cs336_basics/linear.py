import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )
        nn.init.trunc_normal_(self.weight)

    def forward(
        self,
        x: Float[Tensor, "... in_features"],
    ) -> Float[Tensor, "... out_features"]:
        output = torch.einsum("...i,oi->...o", x, self.weight)

        return output
