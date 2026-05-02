#!/usr/bin/env python3
"""
Convert PyTorch 2.4+ checkpoint to PyTorch 2.2 compatible format.

This script loads a checkpoint and re-saves it in a format compatible with PyTorch 2.2.
"""
import torch
import sys
import argparse
from pathlib import Path

def convert_checkpoint(input_path: str, output_path: str):
    """
    Convert checkpoint to PyTorch 2.2 compatible format.
    
    Args:
        input_path: Path to input checkpoint
        output_path: Path to save converted checkpoint
    """
    print(f"\n{'='*80}")
    print("PyTorch Checkpoint Converter")
    print(f"{'='*80}\n")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"\n❌ Input checkpoint not found: {input_path}")
        sys.exit(1)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Loading checkpoint...")
    try:
        # Try loading with weights_only=False (PyTorch 2.4+)
        try:
            checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Fall back to older method
            checkpoint = torch.load(input_path, map_location='cpu')
        
        print(f"✅ Checkpoint loaded successfully")
        
        # Print checkpoint info
        if isinstance(checkpoint, dict):
            print(f"\n📊 Checkpoint contents:")
            for key in checkpoint.keys():
                if key == 'model_state_dict':
                    num_params = sum(p.numel() for p in checkpoint[key].values())
                    print(f"   - {key}: {num_params:,} parameters")
                elif isinstance(checkpoint[key], dict):
                    print(f"   - {key}: {len(checkpoint[key])} items")
                else:
                    print(f"   - {key}: {type(checkpoint[key]).__name__}")
        
        # Re-save in compatible format
        print(f"\n💾 Saving converted checkpoint...")
        torch.save(checkpoint, output_path)
        print(f"✅ Checkpoint saved: {output_path}")
        
        # Verify by loading
        print(f"\n🔍 Verifying converted checkpoint...")
        try:
            verify = torch.load(output_path, map_location='cpu')
        except TypeError:
            verify = torch.load(output_path, map_location='cpu')
        print(f"✅ Verification successful")
        
        print(f"\n{'='*80}")
        print("✅ Conversion Complete!")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch checkpoint to compatible format'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Input checkpoint path'
    )
    parser.add_argument(
        'output',
        type=str,
        help='Output checkpoint path'
    )
    
    args = parser.parse_args()
    
    convert_checkpoint(args.input, args.output)


if __name__ == '__main__':
    main()
