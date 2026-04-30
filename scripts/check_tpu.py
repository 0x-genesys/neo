#!/usr/bin/env python3
"""
Check TPU availability and configuration.

Usage:
    python scripts/check_tpu.py
"""

import sys
import os

def check_environment():
    """Check if running on Kaggle."""
    print("="*80)
    print("Environment Check")
    print("="*80)
    
    if os.path.exists('/kaggle'):
        print("✅ Running on Kaggle")
    else:
        print("⚠️  Not running on Kaggle")
    
    print()

def check_torch_xla():
    """Check if torch_xla is installed."""
    print("="*80)
    print("torch_xla Check")
    print("="*80)
    
    try:
        import torch_xla
        print(f"✅ torch_xla installed")
        print(f"   Version: {torch_xla.__version__}")
        return True
    except ImportError as e:
        print(f"❌ torch_xla not installed")
        print(f"   Error: {e}")
        print()
        print("To install torch_xla:")
        print("  bash scripts/setup_kaggle_tpu.sh")
        print()
        print("Or manually:")
        print("  curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py")
        print("  python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev")
        return False
    
    print()

def check_tpu():
    """Check TPU availability."""
    print("="*80)
    print("TPU Availability Check")
    print("="*80)
    
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        
        # Try to get TPU device
        try:
            # Use new API (torch_xla 2.0+)
            device = xm.xla_device()
            print(f"✅ TPU available")
            print(f"   Device: {device}")
            
            # Get TPU info using new API
            try:
                # Try new API first
                if hasattr(xr, 'world_size'):
                    world_size = xr.world_size()
                    ordinal = xr.global_ordinal()
                    print(f"   Cores: {world_size}")
                    print(f"   Current ordinal: {ordinal}")
                # Fall back to old API
                elif hasattr(xm, 'xrt_world_size'):
                    world_size = xm.xrt_world_size()
                    ordinal = xm.get_ordinal()
                    print(f"   Cores: {world_size}")
                    print(f"   Current ordinal: {ordinal}")
                else:
                    print(f"   ⚠️  Could not determine core count (API changed)")
                    print(f"   This is OK - TPU is still available")
            except Exception as e:
                print(f"   ⚠️  Could not get TPU info: {e}")
                print(f"   This is OK - TPU is still available")
            
            return True
            
        except Exception as e:
            print(f"❌ TPU not available")
            print(f"   Error: {e}")
            print()
            print("To enable TPU in Kaggle:")
            print("1. Click the Settings icon (gear) in the notebook")
            print("2. Under 'Accelerator', select 'TPU v3-8'")
            print("3. Click 'Save'")
            print("4. The notebook will restart")
            print()
            print("After restart, run this script again to verify.")
            return False
            
    except ImportError:
        print("❌ torch_xla not installed (run check_torch_xla first)")
        return False
    
    print()

def check_pytorch():
    """Check PyTorch installation."""
    print("="*80)
    print("PyTorch Check")
    print("="*80)
    
    try:
        import torch
        print(f"✅ PyTorch installed")
        print(f"   Version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("❌ PyTorch not installed")
    
    print()

def main():
    """Main check function."""
    print()
    print("="*80)
    print("TPU Availability Checker")
    print("="*80)
    print()
    
    # Check environment
    check_environment()
    
    # Check PyTorch
    check_pytorch()
    
    # Check torch_xla
    xla_installed = check_torch_xla()
    
    if not xla_installed:
        print("="*80)
        print("❌ Setup incomplete - torch_xla not installed")
        print("="*80)
        sys.exit(1)
    
    # Check TPU
    tpu_available = check_tpu()
    
    if tpu_available:
        print("="*80)
        print("✅ All checks passed - TPU is ready!")
        print("="*80)
        print()
        print("You can now train with:")
        print("  python train.py --config config/auto_training_117m_balanced.yaml --tpu")
        print()
        sys.exit(0)
    else:
        print("="*80)
        print("❌ TPU not available")
        print("="*80)
        print()
        print("Please enable TPU in Kaggle settings and restart the notebook.")
        print()
        sys.exit(1)

if __name__ == '__main__':
    main()
