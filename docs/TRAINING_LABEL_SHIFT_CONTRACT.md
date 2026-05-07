# Training Label-Shift Contract (Canonical)

This is the source-of-truth contract for autoregressive next-token training in this repository.

## Invariants
1. **DataLoader returns unshifted labels**  
   `targets` must align with `input_ids` in shape: `(B, T)`.

2. **Model applies causal shift exactly once**  
   Loss is computed on:
   - `shift_logits = logits[:, :-1, :]`
   - `shift_targets = targets[:, 1:]`

3. **Padding labels are ignored in loss**  
   Any padded target positions must be set to `-100` by collation.

## Why
This prevents:
- double-shift bugs,
- identity-task bugs,
- accidental training on padded tokens.

## Implementation Locations
- Collation and padding mask: `src/data.py`
- Causal shift and CE loss: `src/model.py`

## Historical Notes
Older fix logs and incident docs may describe intermediate states.  
If any historical document conflicts with this file, this file is authoritative.
