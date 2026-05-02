"""
Fine-Tuning Suite for 117M Transformer with LoRA and Chain-of-Thought
Hardware-adaptive fine-tuning supporting GPU (CUDA/MPS), TPU, and CPU.
"""

from .base_trainer import LoRAFineTuner, SYSTEM_PROMPT
from .data_utils import (
    CoTDataset,
    create_cot_dataset,
    prepare_tokenizer,
    create_sample_cot_data,
    validate_cot_format,
    SPECIAL_TOKENS,
)

__all__ = [
    'LoRAFineTuner',
    'SYSTEM_PROMPT',
    'CoTDataset',
    'create_cot_dataset',
    'prepare_tokenizer',
    'create_sample_cot_data',
    'validate_cot_format',
    'SPECIAL_TOKENS',
]
