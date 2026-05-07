import torch
import yaml
from pathlib import Path


import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


from src.model import create_model
from peft import LoraConfig, get_peft_model

print("🧪 Testing PEFT Save Mechanics...")

# 1. Load your dummy config (adjust path if needed)
with open("config/auto_training_117m_balanced.yaml", 'r') as f:
    config = yaml.safe_load(f)

# 2. Initialize the base model
print("Building base model...")
model = create_model(config)

# 3. Apply a dummy LoRA configuration
print("Applying LoRA wrapper...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['c_attn', 'c_proj', 'net.0', 'net.2', 'lm_head'],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
peft_model = get_peft_model(model, lora_config)

# 4. Attempt the save
test_save_dir = Path("test_dummy_save")
print(f"Attempting to save to {test_save_dir}...")

try:
    peft_model.save_pretrained(test_save_dir)
    print("✅ SUCCESS: The model saved perfectly! The SimpleNamespace bug is fixed.")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_save_dir)
    print("🧹 Cleanup complete.")
except Exception as e:
    print(f"❌ FAILED: The save mechanism crashed again.\nError: {e}")