#!/usr/bin/env python3
"""
Test script to verify PEFT compliance of the model.
"""
import torch
import yaml
from pathlib import Path

print("\n" + "="*80)
print("🧪 Testing PEFT Compliance")
print("="*80 + "\n")

# Test 1: Model Creation
print("Test 1: Model Creation with PEFT Attributes")
print("-" * 80)

from src.model import create_model

# Load config
config_path = Path("config/auto_training_117m_balanced.yaml")
if not config_path.exists():
    print(f"❌ Config file not found: {config_path}")
    exit(1)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print(f"✅ Loaded config from: {config_path}")

# Create model
model = create_model(config)
print(f"✅ Model created successfully")

# Test 2: Verify PEFT Attributes
print("\nTest 2: Verify PEFT Attributes")
print("-" * 80)

required_attrs = ['config', 'generation_config', 'main_input_name']
for attr in required_attrs:
    if hasattr(model, attr):
        print(f"✅ Model has '{attr}' attribute")
    else:
        print(f"❌ Model missing '{attr}' attribute")
        exit(1)

# Test 3: Verify Config Contents
print("\nTest 3: Verify Config Contents")
print("-" * 80)

required_config_attrs = [
    'vocab_size', 'hidden_size', 'num_hidden_layers',
    'num_attention_heads', 'max_position_embeddings',
    'model_type', 'is_encoder_decoder'
]

for attr in required_config_attrs:
    if hasattr(model.config, attr):
        value = getattr(model.config, attr)
        print(f"✅ config.{attr} = {value}")
    else:
        print(f"❌ config missing '{attr}' attribute")
        exit(1)

# Test 4: Verify Generation Config
print("\nTest 4: Verify Generation Config")
print("-" * 80)

gen_config_attrs = ['max_length', 'bos_token_id', 'eos_token_id', 'pad_token_id']
for attr in gen_config_attrs:
    if hasattr(model.generation_config, attr):
        value = getattr(model.generation_config, attr)
        print(f"✅ generation_config.{attr} = {value}")
    else:
        print(f"❌ generation_config missing '{attr}' attribute")
        exit(1)

# Test 5: Verify Main Input Name
print("\nTest 5: Verify Main Input Name")
print("-" * 80)

if model.main_input_name == "input_ids":
    print(f"✅ main_input_name = '{model.main_input_name}'")
else:
    print(f"❌ main_input_name = '{model.main_input_name}' (expected 'input_ids')")
    exit(1)

# Test 6: LoRA Application
print("\nTest 6: LoRA Application")
print("-" * 80)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj", "net.0", "net.2", "lm_head"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=True,
    )
    
    peft_model = get_peft_model(model, lora_config)
    print(f"✅ LoRA applied successfully")
    
    # Verify PEFT model structure
    if hasattr(peft_model, 'base_model'):
        print(f"✅ PEFT model has 'base_model' attribute")
    else:
        print(f"❌ PEFT model missing 'base_model' attribute")
        exit(1)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_pct = 100 * trainable_params / total_params
    
    print(f"✅ Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")
    print(f"✅ Total parameters: {total_params:,}")
    
except ImportError:
    print(f"⚠️  PEFT library not installed, skipping LoRA test")
except Exception as e:
    print(f"❌ LoRA application failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 7: Forward Pass
print("\nTest 7: Forward Pass")
print("-" * 80)

try:
    # Create dummy input
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, config['model']['vocab_size'], (batch_size, seq_length))
    
    # Forward pass (no targets)
    with torch.no_grad():
        logits, loss = model(input_ids=input_ids)
    
    print(f"✅ Forward pass successful")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Loss: {loss}")
    
    # Verify logits shape
    expected_shape = (batch_size, seq_length, config['model']['vocab_size'])
    if logits.shape == expected_shape:
        print(f"✅ Logits shape correct: {logits.shape}")
    else:
        print(f"❌ Logits shape incorrect: {logits.shape} (expected {expected_shape})")
        exit(1)
    
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 8: Forward Pass with PEFT
print("\nTest 8: Forward Pass with PEFT")
print("-" * 80)

try:
    # Forward pass with PEFT model
    with torch.no_grad():
        logits_peft, loss_peft = peft_model(input_ids=input_ids)
    
    print(f"✅ PEFT forward pass successful")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Logits shape: {logits_peft.shape}")
    print(f"   Loss: {loss_peft}")
    
    # Verify logits shape
    if logits_peft.shape == expected_shape:
        print(f"✅ PEFT logits shape correct: {logits_peft.shape}")
    else:
        print(f"❌ PEFT logits shape incorrect: {logits_peft.shape} (expected {expected_shape})")
        exit(1)
    
except Exception as e:
    print(f"❌ PEFT forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 9: Save/Load Adapter
print("\nTest 9: Save/Load Adapter")
print("-" * 80)

try:
    import tempfile
    import shutil
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    adapter_path = temp_dir / "test_adapter"
    
    # Save adapter
    peft_model.save_pretrained(adapter_path)
    print(f"✅ Adapter saved to: {adapter_path}")
    
    # Check files
    expected_files = ['adapter_config.json', 'adapter_model.safetensors']
    for file in expected_files:
        file_path = adapter_path / file
        if file_path.exists():
            print(f"✅ Found: {file}")
        else:
            # Try .bin extension
            file_path_bin = adapter_path / file.replace('.safetensors', '.bin')
            if file_path_bin.exists():
                print(f"✅ Found: {file.replace('.safetensors', '.bin')}")
            else:
                print(f"⚠️  Missing: {file}")
    
    # Load adapter
    from peft import PeftModel
    loaded_model = PeftModel.from_pretrained(model, adapter_path)
    print(f"✅ Adapter loaded successfully")
    
    # Verify loaded model
    if hasattr(loaded_model, 'config'):
        print(f"✅ Loaded model has config")
        if loaded_model.config.vocab_size == config['model']['vocab_size']:
            print(f"✅ Config vocab_size matches: {loaded_model.config.vocab_size}")
        else:
            print(f"❌ Config vocab_size mismatch: {loaded_model.config.vocab_size} != {config['model']['vocab_size']}")
    
    # Clean up
    shutil.rmtree(temp_dir)
    print(f"✅ Cleaned up temporary directory")
    
except Exception as e:
    print(f"❌ Save/Load adapter failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Summary
print("\n" + "="*80)
print("✅ All Tests Passed!")
print("="*80)
print("\n📊 Summary:")
print(f"   Model: {config['model']['num_layers']} layers, {config['model']['d_model']} dim")
print(f"   Vocabulary: {config['model']['vocab_size']:,} tokens")
print(f"   Context: {config['model']['context_length']} tokens")
print(f"   PEFT Compliance: ✅ VERIFIED")
print(f"   LoRA Support: ✅ VERIFIED")
print(f"   Save/Load: ✅ VERIFIED")
print("\n🚀 Model is ready for fine-tuning!")
print("="*80 + "\n")

