import torch
from jaxtyping import Bool, Float
from torch import Tensor


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... values d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    d_k = Q.shape[-1]

    attn_scores = torch.einsum("...qd,...kd->...qk", Q, K) / (d_k**0.5)

    if mask is not None:
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

    attn_scores_max = torch.max(attn_scores, dim=-1, keepdim=True).values
    attn_scores_shifted = attn_scores - attn_scores_max
    attn_scores_shifted_exp = torch.exp(attn_scores_shifted)

    attn_weights = attn_scores_shifted_exp / torch.sum(attn_scores_shifted_exp, dim=-1, keepdim=True)

    output = torch.einsum("...qk,...kv->...qv", attn_weights, V)

    return output
