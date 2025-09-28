import math


def cosine_schedule_with_warmup(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_learning_rate + cosine_decay * (max_learning_rate - min_learning_rate)

    return min_learning_rate
