#!/usr/bin/env python3
"""
Test script to verify fine-tuning setup.
Checks dependencies, creates sample data, and validates configuration.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def test_imports():
    """Test that all required packages are installed."""
    print("\n" + "="*80)
    print("🔍 Testing Package Imports")
    print("="*80)
    
    packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('peft', 'PEFT (LoRA)'),
        ('tqdm', 'Progress bars'),
        ('yaml', 'YAML config'),
    ]
    
    all_ok = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name:20s} - OK")
        except ImportError:
            print(f"❌ {name:20s} - MISSING")
            all_ok = False
    
    if not all_ok:
        print("\n⚠️  Some packages are missing. Install with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("\n✅ All packages installed!")
    return True


def test_device():
    """Test device detection."""
    print("\n" + "="*80)
    print("🖥️  Testing Device Detection")
    print("="*80)
    
    import torch
    
    devices = []
    
    if torch.cuda.is_available():
        devices.append(('CUDA', torch.cuda.get_device_name(0)))
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append(('MPS', 'Apple Silicon GPU'))
    
    devices.append(('CPU', f'{torch.get_num_threads()} threads'))
    
    print(f"\nAvailable devices:")
    for device_type, device_name in devices:
        print(f"  ✅ {device_type}: {device_name}")
    
    return True


def test_model_import():
    """Test that model can be imported."""
    print("\n" + "="*80)
    print("🔧 Testing Model Import")
    print("="*80)
    
    try:
        from src.model import DecoderOnlyTransformer
        print("✅ Model import successful")
        
        # Try creating a small model
        model = DecoderOnlyTransformer(
            vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_layers=2,
            context_length=128,
            dropout=0.1,
        )
        print(f"✅ Model creation successful ({model.get_num_params()/1e6:.2f}M params)")
        return True
    
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False


def test_finetuning_imports():
    """Test fine-tuning module imports."""
    print("\n" + "="*80)
    print("🎓 Testing Fine-Tuning Imports")
    print("="*80)
    
    try:
        from src.finetuning import (
            LoRAFineTuner,
            SYSTEM_PROMPT,
            CoTDataset,
            create_cot_dataset,
            prepare_tokenizer,
            SPECIAL_TOKENS,
        )
        print("✅ Fine-tuning module imports successful")
        print(f"✅ System prompt loaded ({len(SYSTEM_PROMPT)} chars)")
        print(f"✅ Special tokens: {list(SPECIAL_TOKENS.keys())}")
        return True
    
    except Exception as e:
        print(f"❌ Fine-tuning imports failed: {e}")
        return False


def test_data_creation():
    """Test sample data creation."""
    print("\n" + "="*80)
    print("📊 Testing Data Creation")
    print("="*80)
    
    try:
        from src.finetuning.data_utils import create_sample_cot_data, validate_cot_format
        
        # Create test directory
        test_dir = Path('test_data')
        test_dir.mkdir(exist_ok=True)
        
        # Create sample data
        test_file = test_dir / 'test_cot.jsonl'
        create_sample_cot_data(str(test_file), num_samples=10)
        
        # Validate format
        is_valid = validate_cot_format(str(test_file))
        
        if is_valid:
            print("✅ Data creation and validation successful")
            
            # Clean up
            test_file.unlink()
            test_dir.rmdir()
            
            return True
        else:
            print("❌ Data validation failed")
            return False
    
    except Exception as e:
        print(f"❌ Data creation failed: {e}")
        return False


def test_tokenizer():
    """Test tokenizer loading."""
    print("\n" + "="*80)
    print("📝 Testing Tokenizer")
    print("="*80)
    
    try:
        from src.tokenizer_utils import load_tokenizer
        from src.finetuning.data_utils import prepare_tokenizer
        
        tokenizer = load_tokenizer()
        print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
        
        tokenizer = prepare_tokenizer(tokenizer)
        print(f"✅ Tokenizer prepared with special tokens")
        
        # Test encoding
        text = "<|im_start|>user\nHello<|im_end|>"
        tokens = tokenizer.encode(text)
        print(f"✅ Encoding test successful ({len(tokens)} tokens)")
        
        return True
    
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return False


def test_lora():
    """Test LoRA application."""
    print("\n" + "="*80)
    print("🔧 Testing LoRA Application")
    print("="*80)
    
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from src.model import DecoderOnlyTransformer
        
        # Create small model
        model = DecoderOnlyTransformer(
            vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_layers=2,
            context_length=128,
            dropout=0.1,
        )
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=['c_attn', 'c_proj'],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"✅ LoRA applied successfully")
        print(f"   Total params: {total_params:,}")
        print(f"   Trainable params: {trainable_params:,}")
        print(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        return True
    
    except Exception as e:
        print(f"❌ LoRA test failed: {e}")
        return False


def test_config():
    """Test configuration file."""
    print("\n" + "="*80)
    print("⚙️  Testing Configuration")
    print("="*80)
    
    try:
        import yaml
        
        config_path = Path('config/finetuning_config.yaml')
        if not config_path.exists():
            print(f"⚠️  Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Configuration loaded")
        print(f"   LoRA rank: {config['lora']['r']}")
        print(f"   LoRA alpha: {config['lora']['alpha']}")
        print(f"   Learning rate: {config['training']['learning_rate']}")
        print(f"   Batch size: {config['training']['batch_size']}")
        
        return True
    
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("🧪 Fine-Tuning Setup Test Suite")
    print("="*80)
    
    tests = [
        ("Package Imports", test_imports),
        ("Device Detection", test_device),
        ("Model Import", test_model_import),
        ("Fine-Tuning Imports", test_finetuning_imports),
        ("Data Creation", test_data_creation),
        ("Tokenizer", test_tokenizer),
        ("LoRA Application", test_lora),
        ("Configuration", test_config),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("📊 Test Summary")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} - {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*80)
        print("🎉 All Tests Passed!")
        print("="*80)
        print("\n✅ Fine-tuning setup is ready!")
        print("\nNext steps:")
        print("  1. Prepare data: python scripts/prepare_finetuning_data.py all")
        print("  2. Run training: bash scripts/quick_finetune.sh")
        print("="*80 + "\n")
        return 0
    else:
        print("\n" + "="*80)
        print("⚠️  Some Tests Failed")
        print("="*80)
        print("\nPlease fix the issues above before proceeding.")
        print("="*80 + "\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
