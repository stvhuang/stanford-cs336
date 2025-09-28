from collections.abc import Iterable

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid {lr=}")

        if eps < 0.0:
            raise ValueError(f"Invalid {eps=}")

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid {betas[0]=}")

        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid {betas[1]=}")

        if weight_decay < 0.0:
            raise ValueError(f"Invalid {weight_decay=}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data

                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param.data)
                    state["exp_avg_sq"] = torch.zeros_like(param.data)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                step_size = lr * (bias_correction2**0.5) / bias_correction1

                param.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-step_size)

                if weight_decay != 0:
                    param.data.add_(param.data, alpha=-lr * weight_decay)

        return loss
