from collections.abc import Iterable

import torch


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    epsilon: float = 1e-6,
) -> None:
    gradients = []

    for param in parameters:
        if param.grad is not None:
            gradients.append(param.grad)

    if not gradients:
        return

    total_norm_squared = torch.tensor(0.0)

    for grad in gradients:
        total_norm_squared = total_norm_squared + torch.sum(grad**2)

    total_norm = torch.sqrt(total_norm_squared)

    clip_factor = max_l2_norm / (total_norm + epsilon)

    if clip_factor < 1.0:
        for grad in gradients:
            grad.mul_(clip_factor)
