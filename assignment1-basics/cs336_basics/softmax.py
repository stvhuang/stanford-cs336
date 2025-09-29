import torch
from jaxtyping import Float
from torch import Tensor


def softmax(
    in_features: Float[Tensor, "..."],
    dim: int,
) -> Float[Tensor, "..."]:
    x = in_features

    # shift
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x = x - x_max

    # exp and normalize
    x_exp = torch.exp(x)
    output = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

    return output
