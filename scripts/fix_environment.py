#!/usr/bin/env python3
"""
Environment diagnostic and fix script.

This script checks your Python environment and helps fix common issues.
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ Python 3.8+ required")
        return False
    else:
        print("   ✅ Python version OK")
        return True


def check_pytorch():
    """Check PyTorch installation."""
    print("\n🔥 Checking PyTorch...")
    
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"   ✅ MPS (Apple Silicon) available")
        else:
            print(f"   ℹ️  CPU only (no GPU acceleration)")
        
        # Check version compatibility
        version_parts = torch.__version__.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1])
        
        if major < 2:
            print(f"   ⚠️  PyTorch 2.0+ recommended (found {torch.__version__})")
            print(f"   Consider upgrading: pip install --upgrade torch torchvision")
            return False
        else:
            print(f"   ✅ PyTorch version OK")
            return True
            
    except ImportError:
        print("   ❌ PyTorch not installed")
        print("   Install with: pip install torch torchvision")
        return False


def check_transformers():
    """Check transformers library."""
    print("\n🤗 Checking transformers...")
    
    try:
        import transformers
        print(f"   Transformers version: {transformers.__version__}")
        
        # Check if PyTorch backend is available
        try:
            from transformers import AutoTokenizer
            print("   ✅ Transformers OK")
            return True
        except Exception as e:
            print(f"   ⚠️  Transformers warning: {e}")
            return False
            
    except ImportError:
        print("   ❌ Transformers not installed")
        print("   Install with: pip install transformers")
        return False


def check_other_dependencies():
    """Check other required dependencies."""
    print("\n📦 Checking other dependencies...")
    
    required = {
        'numpy': 'numpy',
        'tqdm': 'tqdm',
        'yaml': 'pyyaml',
        'tensorboard': 'tensorboard',
        'tiktoken': 'tiktoken',
        'datasets': 'datasets',
        'huggingface_hub': 'huggingface-hub'
    }
    
    all_ok = True
    for module_name, package_name in required.items():
        try:
            __import__(module_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} not installed")
            all_ok = False
    
    return all_ok


def check_amp_compatibility():
    """Check mixed precision (AMP) compatibility."""
    print("\n⚡ Checking mixed precision support...")
    
    try:
        import torch
        
        # Try different import paths
        try:
            from torch.amp import autocast, GradScaler
            print("   ✅ torch.amp available (PyTorch 2.4+)")
            return True
        except ImportError:
            try:
                from torch.cuda.amp import autocast, GradScaler
                print("   ✅ torch.cuda.amp available (PyTorch 2.0-2.3)")
                return True
            except ImportError:
                print("   ⚠️  GradScaler not available")
                print("   Mixed precision training will be disabled")
                return False
                
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False


def suggest_fixes():
    """Suggest fixes for common issues."""
    print("\n🔧 Suggested fixes:")
    print("="*50)
    
    print("\n1. Upgrade PyTorch (recommended):")
    print("   pip install --upgrade torch torchvision")
    
    print("\n2. Reinstall all dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n3. Create fresh virtual environment:")
    print("   python3 -m venv venv_new")
    print("   source venv_new/bin/activate")
    print("   pip install -r requirements.txt")
    
    print("\n4. For Mac with Apple Silicon:")
    print("   pip install --upgrade torch torchvision torchaudio")
    
    print("\n5. For CUDA systems:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")


def test_imports():
    """Test critical imports."""
    print("\n🧪 Testing critical imports...")
    
    try:
        print("   Testing: src.model")
        from src.model import create_model
        print("   ✅ src.model OK")
        
        print("   Testing: src.trainer")
        from src.trainer import Trainer
        print("   ✅ src.trainer OK")
        
        print("   Testing: src.data")
        from src.data import load_data
        print("   ✅ src.data OK")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False


def main():
    """Run all checks."""
    print("🔍 Environment Diagnostic Tool")
    print("="*50)
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch", check_pytorch),
        ("Transformers", check_transformers),
        ("Other Dependencies", check_other_dependencies),
        ("Mixed Precision", check_amp_compatibility),
        ("Critical Imports", test_imports),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ Check failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 SUMMARY")
    print("="*50)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20s}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} checks passed")
    
    if passed < len(results):
        suggest_fixes()
        return 1
    else:
        print("\n🎉 All checks passed! Your environment is ready.")
        print("\nYou can now run:")
        print("  python train.py --config config/quick_start.yaml")
        return 0


if __name__ == '__main__':
    exit(main())