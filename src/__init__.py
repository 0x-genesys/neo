"""
Production-ready transformer training pipeline.

This package provides a complete implementation for training
decoder-only transformer language models (GPT-style) from scratch.

Main components:
- model: Transformer architecture
- data: Data loading and preprocessing
- trainer: Training infrastructure
- inference: Production text generation

Example usage:
    from src.model import create_model
    from src.data import load_data
    from src.trainer import Trainer
    
    config = yaml.safe_load(open('config/model_config.yaml'))
    train_loader, val_loader, _, tokenizer = load_data(config)
    model = create_model(config)
    trainer = Trainer(model, train_loader, val_loader, tokenizer, config)
    trainer.train()
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from .model import DecoderOnlyTransformer, create_model
from .data import load_data, TextDataset
from .trainer import Trainer
from .inference import TextGenerator

__all__ = [
    'DecoderOnlyTransformer',
    'create_model',
    'load_data',
    'TextDataset',
    'Trainer',
    'TextGenerator',
]
