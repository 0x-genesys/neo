"""
Model evaluation script.
"""
import torch
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from pathlib import Path

from src.model import create_model
from src.data import load_data


def calculate_perplexity(model, data_loader, device):
    """Calculate perplexity on a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for input_ids, targets in tqdm(data_loader, desc="Calculating perplexity"):
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            logits, loss = model(input_ids, targets)
            
            # Count non-padding tokens
            num_tokens = (targets != 0).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained transformer model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Dataset split to evaluate')
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint['config']
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, tokenizer = load_data(config)
    
    # Select data loader
    if args.split == 'train':
        data_loader = train_loader
    elif args.split == 'val':
        data_loader = val_loader
    else:
        data_loader = test_loader
    
    if data_loader is None:
        print(f"Error: {args.split} split not available")
        return
    
    # Create and load model
    print("Creating model...")
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"\nEvaluating on {args.split} split...")
    
    # Calculate perplexity
    perplexity, avg_loss = calculate_perplexity(model, data_loader, device)
    
    print("\n" + "="*80)
    print("Evaluation Results:")
    print("="*80)
    print(f"Split: {args.split}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print("="*80)
    
    # Generate sample texts
    print("\nGenerating sample texts...")
    from src.data import get_sample_prompts
    
    prompts = get_sample_prompts()
    
    print("\n" + "="*80)
    print("Sample Generations:")
    print("="*80)
    
    for prompt in prompts[:3]:
        # Tokenize (don't use return_tensors='pt' due to transformers version issue)
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.8,
                top_k=50,
                top_p=0.95
            )
        
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("-"*80)


if __name__ == '__main__':
    main()
