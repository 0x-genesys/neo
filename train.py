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
        help='Path to local checkpoint to resume from'
    )
    parser.add_argument(
        '--resume-remote',
        type=str,
        default=None,
        help='Checkpoint filename from HuggingFace Hub (e.g., "checkpoint.pt")'
    )
    parser.add_argument(
        '--model-repo',
        type=str,
        default='0x-genesys/neo_weights_checkpoints',
        help='HuggingFace model repository ID'
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
    parser.add_argument(
        '--multi-gpu',
        action='store_true',
        help='Use all available GPUs with DataParallel'
    )
    parser.add_argument(
        '--gpu-ids',
        type=str,
        default=None,
        help='Comma-separated GPU IDs to use (e.g., "0,1")'
    )
    parser.add_argument(
        '--tpu',
        action='store_true',
        help='Use TPU for training (requires torch_xla)'
    )
    parser.add_argument(
        '--tpu-cores',
        type=int,
        default=8,
        help='Number of TPU cores to use (default: 8)'
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.resume:
        config['checkpoint']['resume_from'] = args.resume
    elif args.resume_remote:
        # Download checkpoint from HuggingFace Hub
        from src.remote_model_loader import get_remote_checkpoint_path
        print(f"\n📥 Resuming from remote checkpoint: {args.resume_remote}")
        local_path = get_remote_checkpoint_path(args.resume_remote, args.model_repo)
        config['checkpoint']['resume_from'] = local_path
    if args.dataset:
        # Support both dataset_name (HuggingFace) and train_file (binary)
        if args.dataset.endswith('.bin'):
            config['data']['dataset_type'] = 'binary'
            config['data']['train_file'] = args.dataset
        else:
            config['data']['dataset_name'] = args.dataset
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['max_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Set random seed
    set_seed(config['system']['seed'])
    
    # Handle TPU setup
    use_tpu = args.tpu
    if use_tpu:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.parallel_loader as pl
            import torch_xla.distributed.xla_multiprocessing as xmp
            
            print(f"\n✅ TPU training enabled!")
            print(f"   torch_xla version: {torch_xla.__version__}")
            print(f"   TPU cores: {args.tpu_cores}")
            print(f"   Note: TPU training uses XLA compiler for optimization")
            print()
            
            # Override device to TPU
            config['system']['device'] = 'tpu'
            
        except ImportError:
            print("⚠️  TPU requested but torch_xla not installed.")
            print("   Install with: pip install torch_xla")
            print("   Falling back to auto device selection.")
            use_tpu = False
    
    # Handle multi-GPU setup
    use_multi_gpu = args.multi_gpu
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
        use_multi_gpu = True
    
    # Check GPU availability
    if use_multi_gpu:
        if not torch.cuda.is_available():
            print("⚠️  Multi-GPU requested but CUDA not available. Using single device.")
            use_multi_gpu = False
        elif torch.cuda.device_count() < 2:
            print(f"⚠️  Multi-GPU requested but only {torch.cuda.device_count()} GPU available. Using single GPU.")
            use_multi_gpu = False
        else:
            if gpu_ids is None:
                gpu_ids = list(range(torch.cuda.device_count()))
            print(f"\n✅ Multi-GPU training enabled!")
            print(f"   Using GPUs: {gpu_ids}")
            print(f"   Total GPUs: {len(gpu_ids)}")
            for gpu_id in gpu_ids:
                gpu_name = torch.cuda.get_device_name(gpu_id)
                gpu_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
                print(f"   GPU {gpu_id}: {gpu_name} ({gpu_mem:.2f}GB)")
            print()
    
    print("\n" + "="*80)
    print("Configuration:")
    print("="*80)
    
    # Handle both dataset types
    dataset_type = config['data'].get('dataset_type', 'huggingface')
    if dataset_type == 'binary':
        train_file = config['data'].get('train_file', 'N/A')
        print(f"Dataset: Binary format")
        print(f"  Train file: {train_file}")
        print(f"  Val file: {config['data'].get('val_file', 'N/A')}")
    else:
        dataset_name = config['data'].get('dataset_name', 'N/A')
        print(f"Dataset: {dataset_name}")
    
    print(f"Model: {config['model']['num_layers']} layers, {config['model']['d_model']} dim, {config['model']['num_heads']} heads")
    print(f"Context length: {config['model']['context_length']}")
    print(f"Batch size: {config['training']['batch_size']}")
    if use_multi_gpu:
        print(f"Effective batch size: {config['training']['batch_size'] * len(gpu_ids)} (per-GPU: {config['training']['batch_size']})")
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
    
    # Wrap model with DataParallel if using multiple GPUs
    if use_multi_gpu:
        print(f"🔧 Wrapping model with DataParallel for {len(gpu_ids)} GPUs...")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        print(f"✅ Model parallelized across GPUs: {gpu_ids}")
        print(f"   Primary GPU: {gpu_ids[0]}")
        print(f"   Batch will be split across {len(gpu_ids)} GPUs")
        print()
    
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
