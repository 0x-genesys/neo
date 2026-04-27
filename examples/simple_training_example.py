"""
Simple example: Train a tiny transformer in 5 minutes.

This script demonstrates the complete pipeline with minimal configuration.
Perfect for understanding the workflow before scaling up.
"""
import sys
sys.path.append('..')

import torch
import yaml
from src.model import create_model
from src.data import load_data
from src.trainer import Trainer


def main():
    print("="*80)
    print("Simple Transformer Training Example")
    print("="*80)
    print("\nThis will train a TINY model on tiny_shakespeare dataset.")
    print("Expected time: ~5 minutes on CPU, ~1 minute on GPU")
    print("="*80 + "\n")
    
    # Simple configuration
    config = {
        'model': {
            'vocab_size': 50257,  # Will be set by tokenizer
            'd_model': 128,       # Small model
            'num_heads': 4,
            'num_layers': 2,      # Just 2 layers
            'context_length': 128,
            'dropout': 0.1
        },
        'training': {
            'batch_size': 8,      # Small batch
            'gradient_accumulation_steps': 1,
            'max_epochs': 2,      # Just 2 epochs
            'max_steps': 500,     # Stop after 500 steps
            'learning_rate': 3e-4,
            'weight_decay': 0.01,
            'warmup_steps': 50,
            'max_grad_norm': 1.0,
            'eval_interval': 100,
            'save_interval': 500,
            'log_interval': 10
        },
        'data': {
            'dataset_name': 'tiny_shakespeare',
            'dataset_config': None,
            'train_split': 'train',
            'val_split': 'validation',
            'test_split': 'test',
            'max_length': 128,
            'num_workers': 0  # 0 for simplicity
        },
        'tokenizer': {
            'type': 'gpt2',
            'vocab_size': 50257
        },
        'optimizer': {
            'type': 'adamw',
            'betas': [0.9, 0.95],
            'eps': 1e-8
        },
        'scheduler': {
            'type': 'cosine',
            'min_lr': 3e-5
        },
        'system': {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'mixed_precision': torch.cuda.is_available(),
            'compile_model': False,
            'seed': 42
        },
        'checkpoint': {
            'save_dir': 'example_checkpoints',
            'resume_from': None,
            'save_best_only': False
        },
        'logging': {
            'use_wandb': False,
            'wandb_project': 'transformer-example',
            'wandb_entity': None,
            'log_dir': 'example_logs'
        },
        'generation': {
            'max_new_tokens': 50,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.95,
            'num_samples': 2
        }
    }
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, tokenizer = load_data(config)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(model, train_loader, val_loader, tokenizer, config)
    
    # Train
    print("\nStarting training...")
    print("(Press Ctrl+C to stop early)\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")
        trainer.save_checkpoint('example_interrupted.pt')
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print("\nYour model is saved in: example_checkpoints/")
    print("\nTo generate text, run:")
    print("  python -m src.inference --model example_checkpoints/best_model.pt --interactive")
    print("\nOr try:")
    print("  python -m src.inference --model example_checkpoints/best_model.pt --prompt 'Once upon a time'")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
