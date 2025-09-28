import torch


def cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    vocab_size = inputs.shape[-1]

    logits_flat = inputs.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    max_logits = logits_flat.max(dim=-1, keepdim=True)[0]

    logits_shifted = logits_flat - max_logits

    log_sum_exp = torch.log(torch.sum(torch.exp(logits_shifted), dim=-1))

    target_logits = logits_shifted[torch.arange(logits_flat.shape[0]), targets_flat]

    loss_per_example = -target_logits + log_sum_exp

    return loss_per_example.mean()
