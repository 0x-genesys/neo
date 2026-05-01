#!/usr/bin/env python3
"""
Comprehensive validation script for all configurations.
Validates configs, code flows, and resilience.
"""
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("❌ Error: pyyaml not installed")
    print("   Please install: pip install pyyaml")
    print("   Or activate venv: source venv/bin/activate")
    sys.exit(1)

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def validate_config_file(config_path):
    """Validate a single config file."""
    print(f"\n{'='*80}")
    print(f"Validating: {config_path.name}")
    print(f"{'='*80}")
    
    errors = []
    warnings = []
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print("✅ YAML syntax valid")
    except Exception as e:
        errors.append(f"YAML parsing error: {e}")
        return errors, warnings
    
    # Validate required sections
    required_sections = ['model', 'training', 'data', 'tokenizer', 'optimizer', 
                        'scheduler', 'system', 'checkpoint', 'logging', 'generation']
    
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
        else:
            print(f"✅ Section '{section}' present")
    
    if errors:
        return errors, warnings
    
    # Validate model config
    model = config['model']
    required_model_keys = ['vocab_size', 'd_model', 'num_heads', 'num_layers', 
                          'context_length', 'dropout']
    
    for key in required_model_keys:
        if key not in model:
            errors.append(f"Missing model.{key}")
    
    # Validate head_dim
    if 'num_heads' in model and 'd_model' in model:
        head_dim = model['d_model'] / model['num_heads']
        if head_dim != int(head_dim):
            errors.append(f"d_model ({model['d_model']}) not divisible by num_heads ({model['num_heads']})")
        else:
            print(f"✅ head_dim = {int(head_dim)} (d_model={model['d_model']}, num_heads={model['num_heads']})")
    
    # Validate gradient checkpointing
    if 'use_gradient_checkpointing' not in model:
        warnings.append("use_gradient_checkpointing not specified (will default to False)")
    
    # Validate training config
    training = config['training']
    required_training_keys = ['batch_size', 'gradient_accumulation_steps', 'max_steps',
                             'learning_rate', 'warmup_steps']
    
    for key in required_training_keys:
        if key not in training:
            errors.append(f"Missing training.{key}")
    
    # Calculate effective batch size
    if 'batch_size' in training and 'gradient_accumulation_steps' in training:
        effective_batch = training['batch_size'] * training['gradient_accumulation_steps']
        print(f"✅ Effective batch size: {effective_batch} (batch={training['batch_size']}, accum={training['gradient_accumulation_steps']})")
        
        if effective_batch < 32:
            warnings.append(f"Effective batch size ({effective_batch}) is small, may affect convergence")
    
    # Validate warmup ratio
    if 'warmup_steps' in training and 'max_steps' in training:
        warmup_ratio = training['warmup_steps'] / training['max_steps']
        print(f"✅ Warmup ratio: {warmup_ratio:.1%} ({training['warmup_steps']}/{training['max_steps']} steps)")
        
        if warmup_ratio < 0.01:
            warnings.append(f"Warmup ratio ({warmup_ratio:.1%}) is very small")
        elif warmup_ratio > 0.1:
            warnings.append(f"Warmup ratio ({warmup_ratio:.1%}) is large")
    
    # Validate tokenizer
    tokenizer = config['tokenizer']
    if 'type' not in tokenizer:
        errors.append("Missing tokenizer.type")
    else:
        tok_type = tokenizer['type']
        print(f"✅ Tokenizer: {tok_type}")
        
        # Check if GPT-4 tokenizer
        if 'gpt-4' in tok_type.lower() or 'xenova' in tok_type.lower():
            expected_vocab = 100277
            if tokenizer.get('vocab_size', expected_vocab) != expected_vocab:
                warnings.append(f"GPT-4 tokenizer should have vocab_size={expected_vocab}")
        elif 'gpt2' in tok_type.lower():
            expected_vocab = 50257
            if tokenizer.get('vocab_size', expected_vocab) != expected_vocab:
                warnings.append(f"GPT-2 tokenizer should have vocab_size={expected_vocab}")
    
    # Validate dataset
    data = config['data']
    if 'dataset_name' not in data:
        errors.append("Missing data.dataset_name")
    else:
        dataset_name = data['dataset_name']
        print(f"✅ Dataset: {dataset_name}")
        
        # Check dataset size appropriateness
        if 'wikitext-2' in str(data.get('dataset_config', '')):
            if model.get('num_layers', 0) > 12:
                warnings.append("wikitext-2 is small for large models (>12 layers)")
    
    # Validate system config
    system = config['system']
    if 'mixed_precision' not in system:
        warnings.append("mixed_precision not specified")
    
    if system.get('device') == 'cuda' and not system.get('mixed_precision'):
        warnings.append("GPU training without mixed_precision (slower, more memory)")
    
    # Estimate parameter count
    if all(k in model for k in ['vocab_size', 'd_model', 'num_layers']):
        vocab_size = model['vocab_size']
        d_model = model['d_model']
        num_layers = model['num_layers']
        
        # Approximate: 12 * num_layers * d_model^2 + vocab_size * d_model
        params = 12 * num_layers * d_model * d_model + vocab_size * d_model
        params_m = params / 1e6
        params_b = params / 1e9
        
        if params_b >= 1:
            print(f"✅ Estimated parameters: {params_b:.2f}B")
        else:
            print(f"✅ Estimated parameters: {params_m:.0f}M")
        
        # Check if gradient checkpointing is appropriate
        if params_m > 100 and not model.get('use_gradient_checkpointing'):
            warnings.append(f"Model is large ({params_m:.0f}M params) but gradient_checkpointing is disabled")
    
    # Estimate memory requirements (FP16)
    if all(k in model for k in ['vocab_size', 'd_model', 'num_layers']):
        model_memory_gb = params * 2 / 1e9  # FP16
        optimizer_memory_gb = params * 8 / 1e9  # Adam states
        
        # Rough activation estimate
        batch_size = training.get('batch_size', 1)
        context_length = model.get('context_length', 512)
        activation_memory_gb = batch_size * context_length * d_model * num_layers * 4 * 2 / 1e9
        
        total_memory_gb = model_memory_gb + optimizer_memory_gb + activation_memory_gb
        
        print(f"✅ Estimated memory (FP16):")
        print(f"   Model: {model_memory_gb:.1f}GB")
        print(f"   Optimizer: {optimizer_memory_gb:.1f}GB")
        print(f"   Activations: {activation_memory_gb:.1f}GB")
        print(f"   Total: {total_memory_gb:.1f}GB")
        
        # Check if fits in common GPUs
        if total_memory_gb > 80:
            warnings.append(f"Model requires {total_memory_gb:.0f}GB (needs multiple A100 80GB GPUs)")
        elif total_memory_gb > 40:
            warnings.append(f"Model requires {total_memory_gb:.0f}GB (needs A100 80GB)")
        elif total_memory_gb > 24:
            warnings.append(f"Model requires {total_memory_gb:.0f}GB (needs A100 40GB or better)")
        elif total_memory_gb > 16:
            warnings.append(f"Model requires {total_memory_gb:.0f}GB (needs V100 32GB or A100)")
    
    return errors, warnings


def validate_code_imports():
    """Validate that all required imports work."""
    print(f"\n{'='*80}")
    print("Validating Code Imports")
    print(f"{'='*80}")
    
    errors = []
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        errors.append(f"PyTorch import failed: {e}")
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError as e:
        errors.append(f"Transformers import failed: {e}")
    
    try:
        import datasets
        print(f"✅ Datasets {datasets.__version__}")
    except ImportError as e:
        errors.append(f"Datasets import failed: {e}")
    
    try:
        import tiktoken
        print(f"✅ Tiktoken {tiktoken.__version__}")
    except ImportError as e:
        errors.append(f"Tiktoken import failed (GPT-4 tokenizer won't work): {e}")
    
    try:
        from src.model import create_model
        print(f"✅ src.model imports successfully")
    except ImportError as e:
        errors.append(f"src.model import failed: {e}")
    
    try:
        from src.data import load_data
        print(f"✅ src.data imports successfully")
    except ImportError as e:
        errors.append(f"src.data import failed: {e}")
    
    try:
        from src.trainer import Trainer
        print(f"✅ src.trainer imports successfully")
    except ImportError as e:
        errors.append(f"src.trainer import failed: {e}")
    
    return errors


def validate_tokenizer_loading():
    """Test tokenizer loading."""
    print(f"\n{'='*80}")
    print("Validating Tokenizer Loading")
    print(f"{'='*80}")
    
    errors = []
    warnings = []
    
    # Test GPT-2
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('gpt2')
        print(f"✅ GPT-2 tokenizer: {len(tok):,} tokens")
    except Exception as e:
        errors.append(f"GPT-2 tokenizer failed: {e}")
    
    # Test GPT-4
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('Xenova/gpt-4')
        print(f"✅ GPT-4 tokenizer: {len(tok):,} tokens")
    except Exception as e:
        warnings.append(f"GPT-4 tokenizer failed (tiktoken may not be installed): {e}")
    
    return errors, warnings


def main():
    """Run all validations."""
    print("="*80)
    print("CONFIGURATION AND CODE VALIDATION")
    print("="*80)
    
    all_errors = []
    all_warnings = []
    
    # Validate code imports
    import_errors = validate_code_imports()
    all_errors.extend(import_errors)
    
    # Validate tokenizer loading
    tok_errors, tok_warnings = validate_tokenizer_loading()
    all_errors.extend(tok_errors)
    all_warnings.extend(tok_warnings)
    
    # Find all config files
    config_dir = Path('config')
    config_files = sorted(config_dir.glob('*.yaml'))
    
    print(f"\n{'='*80}")
    print(f"Found {len(config_files)} configuration files")
    print(f"{'='*80}")
    
    # Validate each config
    for config_file in config_files:
        errors, warnings = validate_config_file(config_file)
        
        if errors:
            print(f"\n❌ ERRORS in {config_file.name}:")
            for error in errors:
                print(f"   - {error}")
            all_errors.extend(errors)
        
        if warnings:
            print(f"\n⚠️  WARNINGS in {config_file.name}:")
            for warning in warnings:
                print(f"   - {warning}")
            all_warnings.extend(warnings)
        
        if not errors and not warnings:
            print(f"\n✅ {config_file.name} is valid!")
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Configs validated: {len(config_files)}")
    print(f"Errors: {len(all_errors)}")
    print(f"Warnings: {len(all_warnings)}")
    
    if all_errors:
        print(f"\n❌ VALIDATION FAILED with {len(all_errors)} error(s)")
        return 1
    elif all_warnings:
        print(f"\n⚠️  VALIDATION PASSED with {len(all_warnings)} warning(s)")
        return 0
    else:
        print(f"\n✅ ALL VALIDATIONS PASSED!")
        return 0


if __name__ == '__main__':
    exit(main())
