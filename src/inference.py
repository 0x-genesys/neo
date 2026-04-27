"""
Production inference module for text generation.
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from pathlib import Path
import yaml


class TextGenerator:
    """Production-ready text generator."""
    
    def __init__(self, model_path, config_path=None, device=None):
        """
        Initialize the text generator.
        
        Args:
            model_path: Path to saved model checkpoint
            config_path: Path to config file (optional)
            device: Device to run on (cuda/cpu/mps)
        """
        self.model_path = Path(model_path)
        
        # Load checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = checkpoint.get('config', None)
            if self.config is None:
                raise ValueError("Config not found in checkpoint and no config_path provided")
        
        # Setup device
        if device is None:
            # Handle both module and script execution
            try:
                from .device_utils import select_device
            except ImportError:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from src.device_utils import select_device
            self.device = select_device('auto', verbose=True)
        else:
            self.device = torch.device(device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['tokenizer']['type']
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create and load model
        try:
            from .model import create_model
        except ImportError:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from src.model import create_model
        
        self.model = create_model(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
        print(f"Vocabulary size: {len(self.tokenizer)}")
        print(f"Context length: {self.config['model']['context_length']}")
    
    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        stop_tokens=None
    ):
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Nucleus sampling threshold
            num_return_sequences: Number of sequences to generate
            stop_tokens: List of tokens to stop generation
            
        Returns:
            List of generated texts
        """
        # Tokenize prompt (don't use return_tensors='pt' due to transformers version issue)
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        input_ids = input_ids.repeat(num_return_sequences, 1)
        input_ids = input_ids.to(self.device)
        
        # Generate
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # Decode
        generated_texts = []
        for i in range(num_return_sequences):
            # Convert tensor to list for tokenizer
            token_ids = output_ids[i].tolist()
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            
            # Apply stop tokens if provided
            if stop_tokens:
                for stop_token in stop_tokens:
                    if stop_token in text:
                        text = text[:text.index(stop_token)]
            
            generated_texts.append(text)
        
        return generated_texts
    
    def interactive_mode(self):
        """Run interactive text generation."""
        print("\n" + "="*80)
        print("Interactive Text Generation Mode")
        print("="*80)
        print("Type your prompt and press Enter to generate text.")
        print("Type 'quit' or 'exit' to stop.")
        print("Type 'config' to see current generation settings.")
        print("="*80 + "\n")
        
        # Default settings
        settings = {
            'max_new_tokens': 100,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.95
        }
        
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                if prompt.lower() == 'config':
                    print("\nCurrent settings:")
                    for key, value in settings.items():
                        print(f"  {key}: {value}")
                    continue
                
                if not prompt:
                    continue
                
                # Check for setting changes
                if prompt.startswith('set '):
                    parts = prompt.split()
                    if len(parts) == 3:
                        key, value = parts[1], parts[2]
                        if key in settings:
                            try:
                                if key == 'max_new_tokens' or key == 'top_k':
                                    settings[key] = int(value)
                                else:
                                    settings[key] = float(value)
                                print(f"Updated {key} to {settings[key]}")
                            except ValueError:
                                print(f"Invalid value for {key}")
                        else:
                            print(f"Unknown setting: {key}")
                    continue
                
                # Generate
                print("\nGenerating...")
                generated = self.generate(
                    prompt,
                    max_new_tokens=settings['max_new_tokens'],
                    temperature=settings['temperature'],
                    top_k=settings['top_k'],
                    top_p=settings['top_p']
                )
                
                print("\n" + "-"*80)
                print(generated[0])
                print("-"*80)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    def batch_generate(self, prompts, **kwargs):
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Generation parameters
            
        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            generated = self.generate(prompt, **kwargs)
            results.append(generated[0])
        return results
    
    def get_perplexity(self, text):
        """
        Calculate perplexity of the model on given text.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity score
        """
        # Tokenize (don't use return_tensors='pt' due to transformers version issue)
        input_ids = self.tokenizer.encode(text)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits, loss = self.model(input_ids, input_ids)
        
        # Calculate perplexity
        perplexity = torch.exp(loss).item()
        
        return perplexity


def main():
    """CLI for inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Text generation with trained transformer')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--prompt', type=str, help='Text prompt for generation')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--max-tokens', type=int, default=100, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.95, help='Nucleus sampling threshold')
    parser.add_argument('--num-samples', type=int, default=1, help='Number of samples to generate')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = TextGenerator(args.model, args.config)
    
    if args.interactive:
        generator.interactive_mode()
    elif args.prompt:
        generated = generator.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_samples
        )
        
        print("\n" + "="*80)
        print("Generated Text:")
        print("="*80)
        for i, text in enumerate(generated, 1):
            print(f"\nSample {i}:")
            print(text)
            print("-"*80)
    else:
        print("Please provide --prompt or use --interactive mode")


if __name__ == '__main__':
    main()
