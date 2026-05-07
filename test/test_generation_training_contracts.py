"""
Unit tests for training/generation contracts:
- no label shift in collation
- padding labels ignored
- model applies causal shift exactly once
- repetition penalty excludes newline and special token IDs
"""
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import collate_fn
from src.generation_utils import apply_repetition_penalty
from src.model import create_model


def _tiny_config():
    return {
        "model": {
            "vocab_size": 128,
            "d_model": 32,
            "num_heads": 4,
            "num_layers": 2,
            "context_length": 16,
            "dropout": 0.0,
            "use_gradient_checkpointing": False,
        }
    }


def test_collate_fn_uses_pad_id_and_masks_padded_targets():
    batch = [
        torch.tensor([11, 12, 13], dtype=torch.long),
        torch.tensor([21], dtype=torch.long),
    ]

    input_ids, targets = collate_fn(batch, max_length=8, pad_token_id=999, ignore_index=-100)

    assert input_ids.shape == (2, 3)
    assert targets.shape == (2, 3)

    # Pad token applied to inputs
    assert input_ids[1].tolist() == [21, 999, 999]
    # Padded labels ignored in loss
    assert targets[1].tolist() == [21, -100, -100]
    # Non-padded labels preserved
    assert targets[0].tolist() == [11, 12, 13]


def test_model_forward_applies_single_causal_shift_and_checks_shape():
    torch.manual_seed(0)
    model = create_model(_tiny_config())
    model.eval()

    input_ids = torch.randint(0, 128, (2, 6))
    targets = input_ids.clone()
    targets[1, 5] = -100  # emulate padded label masking

    with torch.no_grad():
        logits, loss = model(input_ids, targets=targets)

    expected = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
        targets[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    assert torch.allclose(loss, expected, atol=1e-6)

    with pytest.raises(ValueError):
        _ = model(input_ids, targets=targets[:, :-1])


def test_repetition_penalty_excludes_newline_and_special_ids():
    vocab_size = 100300
    logits = torch.zeros((1, vocab_size), dtype=torch.float32)
    logits[0, 10] = 2.0
    logits[0, 20] = -2.0
    logits[0, 198] = 3.0
    logits[0, 100257] = -3.0

    generated_ids = torch.tensor([[10, 20, 198, 100257]], dtype=torch.long)

    out = apply_repetition_penalty(logits.clone(), generated_ids, repetition_penalty=2.0)
    # Penalized
    assert out[0, 10].item() == pytest.approx(1.0)
    assert out[0, 20].item() == pytest.approx(-4.0)
    # Excluded from penalization
    assert out[0, 198].item() == pytest.approx(3.0)
    assert out[0, 100257].item() == pytest.approx(-3.0)


def test_repetition_penalty_noop_at_one():
    logits = torch.tensor([[1.5, -0.5, 0.0]], dtype=torch.float32)
    generated_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    out = apply_repetition_penalty(logits.clone(), generated_ids, repetition_penalty=1.0)
    assert torch.equal(out, logits)
