#!/usr/bin/env python3
"""
Installation verification script.
Tests all components needed for transformer training.
"""

import sys
import platform

def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def print_result(test_name, passed, message=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {test_name}")
    if message:
        print(f"       {message}")

def test_python_version():
    """Test Python version."""
    print_header("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    
    passed = version.major == 3 and version.minor >= 8
    if not passed:
        print_result("Python Version", False, "Python 3.8+ required")
        return False
    
    if version.minor >= 14:
        print_result("Python Version", True, "⚠️  Warning: Python 3.14 - PyTorch may not be available")
        print("       Recommended: Python 3.11 or 3.12")
        return "warning"
    
    print_result("Python Version", True)
    return True

def test_pytorch():
    """Test PyTorch installation."""
    print_header("PyTorch")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print_result("PyTorch Import", True)
        
        # Test device availability
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        
        print(f"\nDevice Availability:")
        print(f"  CUDA (NVIDIA GPU): {cuda_available}")
        if cuda_available:
            print(f"    Device: {torch.cuda.get_device_name(0)}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        print(f"  MPS (Apple GPU): {mps_available}")
        if mps_available:
            if platform.machine() == 'arm64':
                print(f"    Apple Silicon detected")
            else:
                print(f"    Intel Mac with MPS")
        
        print(f"  CPU: Available")
        
        if cuda_available:
            device_type = "CUDA"
        elif mps_available:
            device_type = "MPS"
        else:
            device_type = "CPU"
        
        print(f"\n✅ Will use: {device_type}")
        
        return True
    except ImportError as e:
        print_result("PyTorch Import", False, str(e))
        print("\n⚠️  PyTorch not installed!")
        print("   Install with: pip install torch torchvision torchaudio")
        print("   See INSTALLATION.md for details")
        return False

def test_transformers():
    """Test transformers library."""
    print_header("HuggingFace Transformers")
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        print_result("Transformers Import", True)
        
        # Test tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        print(f"✅ GPT-2 tokenizer loaded: {len(tokenizer)} tokens")
        
        return True
    except ImportError as e:
        print_result("Transformers Import", False, str(e))
        print("\n⚠️  Install with: pip install transformers")
        return False
    except Exception as e:
        print_result("Tokenizer Test", False, str(e))
        return False

def test_datasets():
    """Test datasets library."""
    print_header("HuggingFace Datasets")
    try:
        import datasets
        print(f"Datasets version: {datasets.__version__}")
        print_result("Datasets Import", True)
        
        # Test loading a small dataset with new method
        print("\nTesting dataset loading (wikitext-2)...")
        from datasets import load_dataset
        try:
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:10]')
            print(f"✅ Loaded {len(dataset)} examples")
            return True
        except Exception as e:
            print(f"⚠️  Dataset loading failed: {e}")
            print("   This is OK - datasets will download during training")
            return "warning"
    except ImportError as e:
        print_result("Datasets Import", False, str(e))
        print("\n⚠️  Install with: pip install datasets")
        return False
    except Exception as e:
        print_result("Dataset Loading", False, str(e))
        print("   This might be a network issue. Try again later.")
        return "warning"

def test_other_dependencies():
    """Test other required libraries."""
    print_header("Other Dependencies")
    
    dependencies = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'tqdm': 'tqdm',
        'yaml': 'pyyaml',
        'tensorboard': 'tensorboard',
    }
    
    all_passed = True
    for module, package in dependencies.items():
        try:
            __import__(module)
            print_result(package, True)
        except ImportError:
            print_result(package, False, f"Install with: pip install {package}")
            all_passed = False
    
    return all_passed

def test_project_structure():
    """Test project structure."""
    print_header("Project Structure")
    
    import os
    required_files = [
        'src/model.py',
        'src/trainer.py',
        'src/data.py',
        'src/inference.py',
        'src/device_utils.py',
        'config/model_config.yaml',
        'train.py',
    ]
    
    all_exist = True
    for file in required_files:
        exists = os.path.exists(file)
        print_result(file, exists)
        if not exists:
            all_exist = False
    
    return all_exist

def test_device_utils():
    """Test device detection utilities."""
    print_header("Device Detection")
    try:
        from src.device_utils import select_device, get_device_info
        
        device = select_device('auto', verbose=False)
        print(f"Selected device: {device}")
        
        info = get_device_info()
        print(f"\nDevice Info:")
        print(f"  Platform: {info['platform']}")
        print(f"  Architecture: {info['architecture']}")
        print(f"  CUDA Available: {info['cuda_available']}")
        print(f"  MPS Available: {info['mps_available']}")
        
        print_result("Device Detection", True)
        return True
    except Exception as e:
        print_result("Device Detection", False, str(e))
        return False

def test_huggingface_auth():
    """Test HuggingFace authentication."""
    print_header("HuggingFace Authentication")
    try:
        from huggingface_hub import HfFolder
        
        token = HfFolder.get_token()
        if token:
            print(f"✅ Token found: {token[:10]}...")
            print_result("HuggingFace Auth", True)
        else:
            print("⚠️  No token found (optional for public datasets)")
            print("   To login: huggingface-cli login")
            print("   See HUGGINGFACE_SETUP.md for details")
            print_result("HuggingFace Auth", True, "Not required for public datasets")
        
        return True
    except ImportError as e:
        print(f"⚠️  HuggingFace Hub not found: {e}")
        print("   Installing...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'huggingface-hub', '-q'])
            print("✅ Installed huggingface-hub")
            print_result("HuggingFace Hub", True, "Installed successfully")
            return True
        except:
            print_result("HuggingFace Hub", False, "Install with: pip install huggingface-hub")
            return False

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("  🧪 TRANSFORMER TRAINING - INSTALLATION VERIFICATION")
    print("="*80)
    
    results = {}
    
    # Run tests
    results['python'] = test_python_version()
    results['pytorch'] = test_pytorch()
    results['transformers'] = test_transformers()
    results['datasets'] = test_datasets()
    results['dependencies'] = test_other_dependencies()
    results['structure'] = test_project_structure()
    results['device'] = test_device_utils()
    results['hf_auth'] = test_huggingface_auth()
    
    # Summary
    print_header("Summary")
    
    passed = sum(1 for v in results.values() if v is True)
    warnings = sum(1 for v in results.values() if v == "warning")
    failed = sum(1 for v in results.values() if v is False)
    total = len(results)
    
    print(f"\nTests: {passed} passed, {warnings} warnings, {failed} failed, {total} total")
    
    if failed == 0 and warnings == 0:
        print("\n" + "="*80)
        print("  ✅ ALL TESTS PASSED!")
        print("="*80)
        print("\n🚀 You're ready to start training!")
        print("\nNext steps:")
        print("  1. (Optional) Login to HuggingFace: huggingface-cli login")
        print("  2. Quick start: bash quick_start.sh")
        print("  3. Or train manually: python train.py --config config/model_config.yaml")
        return True
    elif failed == 0:
        print("\n" + "="*80)
        print("  ⚠️  TESTS PASSED WITH WARNINGS")
        print("="*80)
        print("\n✅ You can proceed, but check warnings above")
        return True
    else:
        print("\n" + "="*80)
        print("  ❌ SOME TESTS FAILED")
        print("="*80)
        print("\n⚠️  Please fix the issues above before training")
        print("\nCommon fixes:")
        print("  • Python 3.14: Use Python 3.11 or 3.12 (see INSTALLATION.md)")
        print("  • Missing packages: pip install -r requirements.txt")
        print("  • PyTorch: See INSTALLATION.md for platform-specific instructions")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
