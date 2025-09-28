import torch
from jaxtyping import Float
from torch import Tensor


def softmax(
    in_features: Float[Tensor, "..."],
    dim: int,
) -> Float[Tensor, "..."]:
    x = in_features
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max

    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    output = exp_x / sum_exp

    return output
