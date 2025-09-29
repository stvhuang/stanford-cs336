import torch
from jaxtyping import Float, Int


def cross_entropy(
    inputs: Float[torch.Tensor, "batch_size ... vocab_size"],
    targets: Int[torch.Tensor, "batch_size ..."],
) -> torch.Tensor:
    batch_size = inputs.shape[0]
    vocab_size = inputs.shape[-1]

    logits_flat = inputs.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    logits_max = logits_flat.max(dim=-1, keepdim=True)[0]
    logits_shifted = logits_flat - logits_max

    log_sum_exp = torch.log(torch.sum(torch.exp(logits_shifted), dim=-1))

    target_logits = logits_shifted[torch.arange(batch_size), targets_flat]

    loss_per_example = -target_logits + log_sum_exp

    loss = loss_per_example.mean()

    return loss
