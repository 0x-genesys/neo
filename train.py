"""
Main training script.
"""
import torch
import yaml
import argparse
import random
import numpy as np
from pathlib import Path

from src.model import create_model
from src.data import load_data
from src.trainer import Trainer


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train a transformer language model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/model_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Override dataset name from config'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size from config'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override max epochs from config'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Override learning rate from config'
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.resume:
        config['checkpoint']['resume_from'] = args.resume
    if args.dataset:
        config['data']['dataset_name'] = args.dataset
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['max_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Set random seed
    set_seed(config['system']['seed'])
    
    print("\n" + "="*80)
    print("Configuration:")
    print("="*80)
    print(f"Dataset: {config['data']['dataset_name']}")
    print(f"Model: {config['model']['num_layers']} layers, {config['model']['d_model']} dim, {config['model']['num_heads']} heads")
    print(f"Context length: {config['model']['context_length']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Max epochs: {config['training']['max_epochs']}")
    print(f"Max steps: {config['training']['max_steps']}")
    print("="*80 + "\n")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, tokenizer = load_data(config)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    
    # Optimize for device
    from src.device_utils import optimize_for_device, select_device
    
    # Select device first
    if config['system']['device'] == 'auto':
        device = select_device('auto', verbose=True)
    else:
        device = torch.device(config['system']['device'])
    
    model = optimize_for_device(
        model, 
        device=device,
        compile_model=config['system']['compile_model']
    )
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, tokenizer, config)
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint('interrupted_checkpoint.pt')
        print("Checkpoint saved. You can resume training with --resume interrupted_checkpoint.pt")
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\nSaving checkpoint...")
        trainer.save_checkpoint('error_checkpoint.pt')
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
