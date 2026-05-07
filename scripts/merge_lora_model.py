#!/usr/bin/env python3
"""
Merge LoRA adapter weights with base model to create a standalone model.

This script merges a fine-tuned LoRA adapter with the base model to create
a single .pt file that can be used for inference without PEFT dependencies.
"""
import argparse
import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import DecoderOnlyTransformer
from src.tokenizer_utils import load_tokenizer


def merge_lora_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    model_config: dict = None,
):
    """
    Merge LoRA adapter with base model.
    
    Args:
        base_model_path: Path to base model checkpoint (.pt)
        adapter_path: Path to LoRA adapter directory
        output_path: Path to save merged model
        model_config: Model configuration (if not in checkpoint)
    """
    print("\n" + "="*80)
    print("🔧 Merging LoRA Adapter with Base Model")
    print("="*80 + "\n")
    
    # Load base model
    print(f"📂 Loading base model from: {base_model_path}")
    checkpoint = torch.load(base_model_path, map_location='cpu')
    
    # Extract config from checkpoint if available
    if model_config is None:
        if 'config' in checkpoint:
            model_config = checkpoint['config']['model']
        else:
            # Default 117M config
            model_config = {
                'vocab_size': 100277,
                'd_model': 768,
                'num_heads': 12,
                'num_layers': 12,
                'context_length': 512,
                'dropout': 0.1,
            }
            print("⚠️  No config in checkpoint, using default 117M config")
    
    print(f"   Model: {model_config['num_layers']} layers, {model_config['d_model']} dim")
    
    # Create base model
    model = DecoderOnlyTransformer(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        context_length=model_config['context_length'],
        dropout=model_config['dropout'],
    )
    
    # Load base weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"✅ Base model loaded")
    
    # Load LoRA adapter
    print(f"\n📂 Loading LoRA adapter from: {adapter_path}")
    
    try:
        from peft import PeftModel
        
        # Load adapter
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"✅ LoRA adapter loaded")
        
        # Merge adapter with base model
        print(f"\n🔄 Merging adapter with base model...")
        model = model.merge_and_unload()
        print(f"✅ Adapter merged successfully")
        
    except ImportError:
        print("❌ PEFT library not installed. Install with: pip install peft")
        return False
    except Exception as e:
        print(f"❌ Error loading/merging adapter: {e}")
        return False
    
    # Save merged model
    print(f"\n💾 Saving merged model to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint with same format as training
    merged_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'model': model_config,
        },
    }
    
    # Copy training metadata if available
    if 'epoch' in checkpoint:
        merged_checkpoint['epoch'] = checkpoint['epoch']
    if 'global_step' in checkpoint:
        merged_checkpoint['global_step'] = checkpoint['global_step']
    if 'best_val_loss' in checkpoint:
        merged_checkpoint['best_val_loss'] = checkpoint['best_val_loss']
    
    torch.save(merged_checkpoint, output_path)
    print(f"✅ Merged model saved")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Information:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: {output_path.stat().st_size / 1e6:.2f} MB")
    
    print("\n" + "="*80)
    print("✅ Merge Complete!")
    print("="*80)
    print(f"\n🚀 Use the merged model for inference:")
    print(f"   python src/inference.py --model {output_path} --prompt \"Your prompt\"")
    print("="*80 + "\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Merge LoRA adapter with base model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge with local base model
  python scripts/merge_lora_model.py \\
      --base-model checkpoints/best_model.pt \\
      --adapter finetuned_model_gpu/best_model \\
      --output merged_model.pt
  
  # Merge with remote base model
  python scripts/merge_lora_model.py \\
      --base-model-remote best_model.pt \\
      --adapter finetuned_model_gpu/best_model \\
      --output merged_model.pt
  
  # Upload to HuggingFace Hub
  python scripts/merge_lora_model.py \\
      --base-model checkpoints/best_model.pt \\
      --adapter finetuned_model_gpu/best_model \\
      --output merged_model.pt \\
      --upload \\
      --repo-id 0x-genesys/neo_weights_checkpoints
        """
    )
    
    # Model arguments
    parser.add_argument(
        '--base-model',
        type=str,
        help='Path to base model checkpoint (.pt)'
    )
    parser.add_argument(
        '--base-model-remote',
        type=str,
        help='Base model filename from HuggingFace Hub (e.g., "best_model.pt")'
    )
    parser.add_argument(
        '--model-repo',
        type=str,
        default='0x-genesys/neo_weights_checkpoints',
        help='HuggingFace model repository ID'
    )
    
    # Adapter arguments
    parser.add_argument(
        '--adapter',
        type=str,
        required=True,
        help='Path to LoRA adapter directory'
    )
    
    # Output arguments
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save merged model (.pt)'
    )
    
    # Upload arguments
    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload merged model to HuggingFace Hub'
    )
    parser.add_argument(
        '--upload-repo',
        type=str,
        default='0x-genesys/neo_weights_checkpoints',
        help='HuggingFace repository for upload'
    )
    parser.add_argument(
        '--upload-path',
        type=str,
        default='finetune/',
        help='Path within repository (e.g., "finetune/")'
    )
    parser.add_argument(
        '--upload-name',
        type=str,
        help='Custom name for uploaded file (default: same as output filename)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.base_model and not args.base_model_remote:
        parser.error("Either --base-model or --base-model-remote is required")
    
    # Get base model path
    if args.base_model_remote:
        from src.remote_model_loader import get_remote_checkpoint_path
        print(f"\n📥 Downloading base model from HuggingFace Hub...")
        print(f"   Repository: {args.model_repo}")
        print(f"   File: {args.base_model_remote}")
        base_model_path = get_remote_checkpoint_path(args.base_model_remote, args.model_repo)
    else:
        base_model_path = args.base_model
    
    # Merge adapter
    success = merge_lora_adapter(
        base_model_path=base_model_path,
        adapter_path=args.adapter,
        output_path=args.output,
    )
    
    if not success:
        sys.exit(1)
    
    # Upload if requested
    if args.upload:
        print("\n" + "="*80)
        print("📤 Uploading to HuggingFace Hub")
        print("="*80 + "\n")
        
        try:
            from huggingface_hub import HfApi, create_repo
            
            api = HfApi()
            
            # Create repo if it doesn't exist
            try:
                create_repo(args.upload_repo, repo_type="model", exist_ok=True)
                print(f"✅ Repository ready: {args.upload_repo}")
            except Exception as e:
                print(f"⚠️  Repository may already exist: {e}")
            
            # Upload file
            output_filename = Path(args.output).name
            upload_filename = args.upload_name if args.upload_name else output_filename
            repo_path = f"{args.upload_path}{upload_filename}"
            
            print(f"📤 Uploading {output_filename} as {upload_filename} to {repo_path}...")
            api.upload_file(
                path_or_fileobj=args.output,
                path_in_repo=repo_path,
                repo_id=args.upload_repo,
                repo_type="model",
            )
            
            print(f"✅ Upload complete!")
            print(f"\n🔗 Model URL:")
            print(f"   https://huggingface.co/{args.upload_repo}/blob/main/{repo_path}")
            print("\n" + "="*80 + "\n")
            
        except ImportError:
            print("❌ huggingface_hub not installed. Install with: pip install huggingface-hub")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            print("\n💡 Make sure you're logged in:")
            print("   huggingface-cli login")


if __name__ == '__main__':
    main()
