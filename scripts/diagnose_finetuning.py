"""
Diagnostic script to check fine-tuning setup.
Verifies data loading, tokenization, and initial model state.
"""
import sys
from pathlib import Path
import torch
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import create_model
from src.tokenizer_utils import load_tokenizer
from src.finetuning.data_utils import create_cot_dataset, prepare_tokenizer, SYSTEM_PROMPT

def main():
    print("\n" + "="*80)
    print("🔍 Fine-Tuning Diagnostic Tool")
    print("="*80 + "\n")
    
    # Load config
    config_path = Path(__file__).parent.parent / 'config' / 'auto_training_117m_balanced.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load tokenizer
    print("1️⃣  Loading tokenizer...")
    tokenizer = load_tokenizer()
    tokenizer = prepare_tokenizer(tokenizer)
    vocab_size = len(tokenizer)
    print(f"   Vocab size: {vocab_size}")
    
    # Load model
    print("\n2️⃣  Loading model...")
    model = create_model(config)
    print(f"   Model vocab size: {config['model']['vocab_size']}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load a small sample of data
    print("\n3️⃣  Loading data sample...")
    train_path = Path("data/hf_cot/train.jsonl")
    if not train_path.exists():
        print(f"   ❌ Data not found: {train_path}")
        return
    
    train_dataset, _ = create_cot_dataset(
        train_path=str(train_path),
        val_path=None,
        tokenizer=tokenizer,
        max_length=512,
        system_prompt=SYSTEM_PROMPT,
    )
    
    # Check first example
    print("\n4️⃣  Checking first training example...")
    example = train_dataset[0]
    input_ids = example['input_ids']
    labels = example['labels']
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Max input ID: {input_ids.max().item()}")
    print(f"   Max label ID (excluding -100): {labels[labels != -100].max().item() if (labels != -100).any() else 'N/A'}")
    print(f"   Num padding tokens: {(input_ids == tokenizer.pad_token_id).sum().item()}")
    print(f"   Num masked labels: {(labels == -100).sum().item()}")
    
    # Check if IDs are in range
    if input_ids.max() >= vocab_size:
        print(f"\n   ❌ ERROR: Input IDs exceed vocab size!")
        print(f"      Max ID: {input_ids.max().item()}, Vocab size: {vocab_size}")
    else:
        print(f"\n   ✅ All input IDs within range")
    
    if (labels != -100).any() and labels[labels != -100].max() >= vocab_size:
        print(f"   ❌ ERROR: Label IDs exceed vocab size!")
    else:
        print(f"   ✅ All label IDs within range")
    
    # Test forward pass
    print("\n5️⃣  Testing forward pass...")
    model.eval()
    with torch.no_grad():
        # Create a small batch
        batch_input = input_ids.unsqueeze(0)  # (1, seq_len)
        batch_labels = labels.unsqueeze(0)
        
        try:
            logits, loss = model(batch_input, targets=batch_labels)
            print(f"   ✅ Forward pass successful!")
            print(f"   Logits shape: {logits.shape}")
            print(f"   Loss: {loss.item():.4f}")
            
            if loss.item() > 8.0:
                print(f"\n   ⚠️  WARNING: Loss is very high ({loss.item():.4f})!")
                print(f"      Expected: ~2-3 for pre-trained model")
                print(f"      This suggests a problem with data or model setup")
            elif loss.item() < 5.0:
                print(f"   ✅ Loss looks reasonable for pre-trained model")
        except Exception as e:
            print(f"   ❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Decode a sample
    print("\n6️⃣  Sample decoded text:")
    sample_tokens = input_ids[:100].tolist()  # First 100 tokens
    sample_text = tokenizer.decode(sample_tokens)
    print(f"   {sample_text[:200]}...")
    
    print("\n" + "="*80)
    print("✅ Diagnostic Complete!")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
