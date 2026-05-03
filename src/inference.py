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
    
    def _deep_merge_configs(self, base_config, override_config):
        """
        Deep merge two configs. Override config takes precedence.
        
        Args:
            base_config: Base configuration (from checkpoint)
            override_config: Override configuration (from user file)
            
        Returns:
            Merged configuration
        """
        import copy
        merged = copy.deepcopy(base_config)
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                merged[key] = self._deep_merge_configs(merged[key], value)
            else:
                # Override value
                merged[key] = value
        
        return merged
    
    def __init__(self, model_path, config_path=None, device=None, model_repo=None):
        """
        Initialize the text generator.
        
        Args:
            model_path: Path to saved model checkpoint (local or remote filename)
            config_path: Path to config file (optional)
            device: Device to run on (cuda/cpu/mps)
            model_repo: HuggingFace repository ID for remote loading (optional)
        """
        # Handle remote model loading
        if model_repo:
            print(f"📥 Loading model from HuggingFace Hub...")
            print(f"   Repository: {model_repo}")
            print(f"   File: {model_path}")
            try:
                from .remote_model_loader import get_remote_checkpoint_path
            except ImportError:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from src.remote_model_loader import get_remote_checkpoint_path
            
            # Download checkpoint from HuggingFace Hub
            model_path = get_remote_checkpoint_path(model_path, model_repo)
        
        self.model_path = Path(model_path)
        
        # Load checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get config - always start with checkpoint config
        checkpoint_config = checkpoint.get('config', None)
        if checkpoint_config is None:
            raise ValueError("Config not found in checkpoint. This checkpoint may be corrupted or from an old version.")
        
        # If user provides a config file, merge it (user config overrides only specified fields)
        if config_path:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Deep merge: checkpoint config as base, user config overrides
            self.config = self._deep_merge_configs(checkpoint_config, user_config)
            print(f"✅ Merged config from {config_path} with checkpoint config")
        else:
            self.config = checkpoint_config
        
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
        tokenizer_type = self.config['tokenizer']['type']
        print(f"Loading tokenizer: {tokenizer_type}")
        
        # Check if using tiktoken
        if tokenizer_type == 'tiktoken' or 'cl100k' in tokenizer_type.lower():
            try:
                import tiktoken
            except ImportError:
                raise ImportError("tiktoken is required for this model. Install with: pip install tiktoken")
            
            print("Using tiktoken cl100k_base (GPT-4) tokenizer")
            encoding = tiktoken.get_encoding("cl100k_base")
            
            # Create a wrapper to match HuggingFace interface (same as training code)
            class TiktokenWrapper:
                def __init__(self, encoding):
                    self.encoding = encoding
                    self.vocab_size = encoding.n_vocab
                    self.eos_token = "<|endoftext|>"
                    self.pad_token = "<|endoftext|>"
                    # Get special token IDs properly
                    self.eos_token_id = encoding.encode_single_token(self.eos_token)
                    self.pad_token_id = self.eos_token_id
                
                def encode(self, text, **kwargs):
                    return self.encoding.encode(text, allowed_special='all')
                
                def decode(self, tokens, **kwargs):
                    return self.encoding.decode(tokens)
                
                def __len__(self):
                    return self.vocab_size
            
            self.tokenizer = TiktokenWrapper(encoding)
            print(f"✅ Tiktoken loaded: vocab_size={self.tokenizer.vocab_size:,}")
        else:
            # Use HuggingFace tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
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
        repetition_penalty=1.2,
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
            repetition_penalty: Penalty for previously generated tokens (1.0 = disabled)
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
            top_p=top_p,
            repetition_penalty=repetition_penalty
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
        
        generation_config = self.config.get('generation', {})

        # Default settings
        settings = {
            'max_new_tokens': generation_config.get('max_new_tokens', 100),
            'temperature': generation_config.get('temperature', 0.8),
            'top_k': generation_config.get('top_k', 50),
            'top_p': generation_config.get('top_p', 0.95),
            'repetition_penalty': generation_config.get('repetition_penalty', 1.2),
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
                    top_p=settings['top_p'],
                    repetition_penalty=settings['repetition_penalty'],
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
    parser.add_argument('--model', type=str, help='Path to local model checkpoint')
    parser.add_argument('--model-remote', type=str, help='Remote model filename from HuggingFace Hub (e.g., "best_model.pt")')
    parser.add_argument('--model-repo', type=str, default='0x-genesys/neo_weights_checkpoints', 
                        help='HuggingFace model repository ID')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--prompt', type=str, help='Text prompt for generation')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--max-tokens', type=int, default=100, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.95, help='Nucleus sampling threshold')
    parser.add_argument('--repetition-penalty', type=float, default=1.2, help='Penalty for repeated tokens (1.0 disables)')
    parser.add_argument('--num-samples', type=int, default=1, help='Number of samples to generate')
    
    args = parser.parse_args()
    
    # Validate model arguments
    if not args.model and not args.model_remote:
        parser.error("Either --model or --model-remote must be provided")
    if args.model and args.model_remote:
        parser.error("Cannot specify both --model and --model-remote")
    
    # Determine model path and repo
    model_path = args.model_remote if args.model_remote else args.model
    model_repo = args.model_repo if args.model_remote else None
    
    # Initialize generator
    generator = TextGenerator(model_path, args.config, model_repo=model_repo)
    
    if args.interactive:
        generator.interactive_mode()
    elif args.prompt:
        generated = generator.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
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
