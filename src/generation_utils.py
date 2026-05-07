"""
Utilities for generation-time logit processing.
"""
import torch


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    repetition_penalty: float,
    newline_token_id: int = 198,
    special_token_start: int = 100257,
) -> torch.Tensor:
    """
    Apply repetition penalty to previously generated tokens.

    Excludes newline and special ChatML/tiktoken token IDs from penalization.

    Args:
        logits: Next-token logits of shape (B, vocab_size)
        generated_ids: Generated token IDs so far of shape (B, T)
        repetition_penalty: Penalty factor (1.0 disables)
        newline_token_id: Token ID for newline to exclude
        special_token_start: IDs >= this value are treated as special tokens

    Returns:
        Logits tensor with repetition penalty applied (in-place and returned).
    """
    if repetition_penalty == 1.0:
        return logits

    for i in range(generated_ids.shape[0]):
        generated_tokens = torch.unique(generated_ids[i])
        mask = (generated_tokens != newline_token_id) & (generated_tokens < special_token_start)
        penalize_tokens = generated_tokens[mask]

        if penalize_tokens.numel() > 0:
            logits[i, penalize_tokens] = torch.where(
                logits[i, penalize_tokens] < 0,
                logits[i, penalize_tokens] * repetition_penalty,
                logits[i, penalize_tokens] / repetition_penalty
            )

    return logits
