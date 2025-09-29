import torch
from jaxtyping import Float
from torch import Tensor


def sigmoid(
    in_features: Float[Tensor, "..."],
) -> Float[Tensor, "..."]:
    output = 1.0 / (1.0 + torch.exp(-in_features))

    return output


def silu(
    in_features: Float[Tensor, "..."],
) -> Float[Tensor, "..."]:
    output = in_features * sigmoid(in_features)

    return output
