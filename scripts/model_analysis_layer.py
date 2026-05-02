"""
Model Analysis Script - Analyze layer structure of the transformer model.
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.model import create_model
import yaml

# Load your config
config_path = Path(__file__).parent.parent / 'config' / 'auto_training_117m_balanced.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize the model
model = create_model(config)

# Print the structure
print("\n" + "="*80)
print("Model Structure Analysis")
print("="*80)
print(model)

# Print parameter counts
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n" + "="*80)
print("Parameter Statistics")
print("="*80)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
print("="*80)
