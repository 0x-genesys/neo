#!/usr/bin/env python3
"""
Chat inference with fine-tuned LoRA model.

This script loads a base model with LoRA adapter for Chain-of-Thought inference.
Supports both local and remote model loading from HuggingFace Hub.
"""
import torch
import sys
import argparse
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model import DecoderOnlyTransformer
from src.tokenizer_utils import load_tokenizer
from src.finetuning.base_trainer import SYSTEM_PROMPT
from src.finetuning.data_utils import SPECIAL_TOKENS


class ChatGenerator:
    """Chat generator with LoRA fine-tuned model."""
    
    def __init__(
        self,
        base_model_path: str = None,
        adapter_path: str = "chat_adapter",
        base_model_remote: str = None,
        adapter_remote: str = None,
        model_repo: str = "0x-genesys/neo_weights_checkpoints",
        config_path: str = None,
        device: str = "auto",
        system_prompt: str = None,
    ):
        """
        Initialize chat generator.
        
        Args:
            base_model_path: Path to local base model (.pt)
            adapter_path: Path to local LoRA adapter directory
            base_model_remote: Remote base model filename (e.g., "final_model.pt")
            adapter_remote: Remote adapter filename (e.g., "chat_adapter")
            model_repo: HuggingFace repository ID
            config_path: Path to config file (optional)
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            system_prompt: Custom system prompt (uses default if None)
        """
        print("\n" + "="*80)
        print("🤖 Chat Inference with Fine-Tuned LoRA Model")
        print("="*80 + "\n")
        
        # System prompt
        self.system_prompt = system_prompt if system_prompt else SYSTEM_PROMPT
        
        # Handle remote model loading
        if base_model_remote:
            print(f"📥 Downloading base model from HuggingFace Hub...")
            print(f"   Repository: {model_repo}")
            print(f"   File: {base_model_remote}")
            from src.remote_model_loader import get_remote_checkpoint_path
            base_model_path = get_remote_checkpoint_path(base_model_remote, model_repo)
            print(f"✅ Downloaded to: {base_model_path}")
        
        if adapter_remote:
            print(f"\n📥 Downloading adapter from HuggingFace Hub...")
            print(f"   Repository: {model_repo}")
            print(f"   Path: finetune/{adapter_remote}")
            from huggingface_hub import snapshot_download
            adapter_path = snapshot_download(
                repo_id=model_repo,
                allow_patterns=f"finetune/{adapter_remote}/*",
                local_dir="./cache",
            )
            adapter_path = Path(adapter_path) / "finetune" / adapter_remote
            print(f"✅ Downloaded to: {adapter_path}")
        
        # Validate paths
        if not base_model_path:
            raise ValueError("Either base_model_path or base_model_remote must be provided")
        
        base_model_path = Path(base_model_path)
        adapter_path = Path(adapter_path)
        
        if not base_model_path.exists():
            raise FileNotFoundError(f"Base model not found: {base_model_path}")
        
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        
        # Load config
        print(f"\n📂 Loading base model from: {base_model_path}")
        checkpoint = torch.load(base_model_path, map_location='cpu')
        
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Loaded config from: {config_path}")
        elif 'config' in checkpoint:
            config = checkpoint['config']
            print(f"✅ Loaded config from checkpoint")
        else:
            # Default 117M config
            config = {
                'model': {
                    'vocab_size': 100277,
                    'd_model': 768,
                    'num_heads': 12,
                    'num_layers': 12,
                    'context_length': 512,
                    'dropout': 0.1,
                }
            }
            print(f"⚠️  Using default 117M config")
        
        self.config = config
        model_config = config['model']
        
        # Setup device
        if device == 'auto':
            from src.device_utils import select_device
            self.device = select_device('auto', verbose=True)
        else:
            self.device = torch.device(device)
        
        print(f"\n🔧 Creating model...")
        print(f"   Architecture: {model_config['num_layers']} layers, {model_config['d_model']} dim")
        
        # Create base model
        self.model = DecoderOnlyTransformer(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            context_length=model_config['context_length'],
            dropout=model_config['dropout'],
        )
        
        # Load base weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"✅ Base model loaded")
        
        # Load LoRA adapter
        print(f"\n📂 Loading LoRA adapter from: {adapter_path}")
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print(f"✅ LoRA adapter loaded")
        except ImportError:
            print("❌ PEFT library not installed. Install with: pip install peft")
            raise
        except Exception as e:
            print(f"❌ Error loading adapter: {e}")
            raise
        
        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        print(f"\n📚 Loading tokenizer...")
        self.tokenizer = load_tokenizer()
        
        # Add special tokens if not present
        special_tokens = [SPECIAL_TOKENS['im_start'], SPECIAL_TOKENS['im_end']]
        existing_tokens = set(self.tokenizer.encode(SPECIAL_TOKENS['im_start']))
        
        if len(existing_tokens) > 1:  # Token was split, need to add
            print(f"   Adding special tokens: {special_tokens}")
            # Note: For tiktoken, special tokens are already in vocabulary
            # This is just for compatibility check
        
        print(f"✅ Tokenizer loaded (vocab size: {len(self.tokenizer)})")
        
        print("\n" + "="*80)
        print("✅ Model Ready for Chat!")
        print("="*80 + "\n")
    
    def format_prompt(self, user_message: str, include_thought: bool = True) -> str:
        """
        Format user message into CoT prompt.
        
        Args:
            user_message: User's input message
            include_thought: Whether to include thought block starter
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # System message
        prompt_parts.append(
            f"{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['system']}\n"
            f"{self.system_prompt}{SPECIAL_TOKENS['im_end']}"
        )
        
        # User message
        prompt_parts.append(
            f"{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['user']}\n"
            f"{user_message}{SPECIAL_TOKENS['im_end']}"
        )
        
        # Thought block starter (for CoT)
        if include_thought:
            prompt_parts.append(
                f"{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['thought']}\n"
            )
        
        return '\n'.join(prompt_parts)
    
    def parse_response(self, generated_text: str) -> dict:
        """
        Parse generated text into thought and assistant response.
        
        Args:
            generated_text: Generated text from model
            
        Returns:
            Dictionary with 'thought' and 'response' keys
        """
        result = {
            'thought': '',
            'response': '',
            'full_text': generated_text
        }
        
        # Extract thought block
        thought_start = f"{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['thought']}\n"
        thought_end = f"{SPECIAL_TOKENS['im_end']}"
        
        if thought_start in generated_text:
            thought_section = generated_text.split(thought_start)[1]
            if thought_end in thought_section:
                result['thought'] = thought_section.split(thought_end)[0].strip()
        
        # Extract assistant response
        assistant_start = f"{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['assistant']}\n"
        assistant_end = f"{SPECIAL_TOKENS['im_end']}"
        
        if assistant_start in generated_text:
            assistant_section = generated_text.split(assistant_start)[1]
            if assistant_end in assistant_section:
                result['response'] = assistant_section.split(assistant_end)[0].strip()
            else:
                result['response'] = assistant_section.strip()
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        user_message: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        show_thought: bool = True,
    ) -> dict:
        """
        Generate response to user message.
        
        Args:
            user_message: User's input message
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            show_thought: Whether to show thought process
            
        Returns:
            Dictionary with 'thought', 'response', and 'full_text'
        """
        # Format prompt
        prompt = self.format_prompt(user_message, include_thought=True)
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Generate
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        # Decode
        generated_text = self.tokenizer.decode(output_ids[0].tolist())
        
        # Parse response
        result = self.parse_response(generated_text)
        
        return result
    
    def interactive_mode(
        self,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        show_thought: bool = True,
    ):
        """
        Run interactive chat mode.
        
        Args:
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            show_thought: Whether to show thought process
        """
        print("\n" + "="*80)
        print("💬 Interactive Chat Mode")
        print("="*80)
        print("Commands:")
        print("  - Type your message and press Enter")
        print("  - 'quit' or 'exit' to stop")
        print("  - 'config' to see current settings")
        print("  - 'thought on/off' to toggle thought display")
        print("  - 'set <param> <value>' to change settings")
        print("="*80 + "\n")
        
        settings = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'show_thought': show_thought,
        }
        
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'config':
                    print("\n⚙️  Current settings:")
                    for key, value in settings.items():
                        print(f"   {key}: {value}")
                    continue
                
                if user_input.lower().startswith('thought '):
                    toggle = user_input.split()[1].lower()
                    if toggle == 'on':
                        settings['show_thought'] = True
                        print("✅ Thought display enabled")
                    elif toggle == 'off':
                        settings['show_thought'] = False
                        print("✅ Thought display disabled")
                    continue
                
                if user_input.lower().startswith('set '):
                    parts = user_input.split()
                    if len(parts) == 3:
                        key, value = parts[1], parts[2]
                        if key in settings:
                            try:
                                if key == 'max_new_tokens' or key == 'top_k':
                                    settings[key] = int(value)
                                elif key == 'show_thought':
                                    settings[key] = value.lower() in ['true', '1', 'yes']
                                else:
                                    settings[key] = float(value)
                                print(f"✅ Updated {key} to {settings[key]}")
                            except ValueError:
                                print(f"❌ Invalid value for {key}")
                        else:
                            print(f"❌ Unknown setting: {key}")
                    continue
                
                # Generate response
                print("\n🤖 Assistant: ", end='', flush=True)
                
                result = self.generate(
                    user_input,
                    max_new_tokens=settings['max_new_tokens'],
                    temperature=settings['temperature'],
                    top_k=settings['top_k'],
                    top_p=settings['top_p'],
                    show_thought=settings['show_thought'],
                )
                
                # Display thought if enabled
                if settings['show_thought'] and result['thought']:
                    print(f"\n💭 [Thought: {result['thought']}]\n")
                
                # Display response
                print(result['response'])
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()


def main():
    """CLI for chat inference."""
    parser = argparse.ArgumentParser(
        description='Chat inference with fine-tuned LoRA model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with local models
  python src/finetuning/chat_inference.py \\
      --base-model checkpoints/best_model.pt \\
      --adapter finetuned_model_gpu/best_model \\
      --interactive
  
  # Interactive mode with remote models (default)
  python src/finetuning/chat_inference.py --interactive
  
  # Single prompt
  python src/finetuning/chat_inference.py \\
      --prompt "What is 17 * 23?" \\
      --show-thought
  
  # Custom config
  python src/finetuning/chat_inference.py \\
      --config config/inference.yaml \\
      --interactive
        """
    )
    
    # Model arguments
    parser.add_argument(
        '--base-model',
        type=str,
        help='Path to local base model (.pt)'
    )
    parser.add_argument(
        '--base-model-remote',
        type=str,
        default='final_model.pt',
        help='Remote base model filename (default: final_model.pt)'
    )
    parser.add_argument(
        '--adapter',
        type=str,
        help='Path to local LoRA adapter directory'
    )
    parser.add_argument(
        '--adapter-remote',
        type=str,
        default='chat_adapter',
        help='Remote adapter path (default: chat_adapter)'
    )
    parser.add_argument(
        '--model-repo',
        type=str,
        default='0x-genesys/neo_weights_checkpoints',
        help='HuggingFace repository ID'
    )
    
    # Config arguments
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file (optional)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cuda, mps, cpu)'
    )
    
    # Generation arguments
    parser.add_argument(
        '--prompt',
        type=str,
        help='Single prompt for generation'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=200,
        help='Maximum tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k sampling'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='Nucleus sampling'
    )
    parser.add_argument(
        '--show-thought',
        action='store_true',
        help='Show thought process'
    )
    
    args = parser.parse_args()
    
    # Determine model paths
    base_model_path = args.base_model
    base_model_remote = None if args.base_model else args.base_model_remote
    
    adapter_path = args.adapter if args.adapter else "chat_adapter"
    adapter_remote = None if args.adapter else args.adapter_remote
    
    # Initialize generator
    generator = ChatGenerator(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        base_model_remote=base_model_remote,
        adapter_remote=adapter_remote,
        model_repo=args.model_repo,
        config_path=args.config,
        device=args.device,
    )
    
    if args.interactive:
        generator.interactive_mode(
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            show_thought=args.show_thought,
        )
    elif args.prompt:
        result = generator.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            show_thought=args.show_thought,
        )
        
        print("\n" + "="*80)
        print("💬 Chat Response")
        print("="*80)
        
        if args.show_thought and result['thought']:
            print(f"\n💭 Thought Process:")
            print(result['thought'])
            print()
        
        print(f"🤖 Assistant:")
        print(result['response'])
        print("\n" + "="*80 + "\n")
    else:
        print("Please provide --prompt or use --interactive mode")
        print("Run with --help for usage examples")


if __name__ == '__main__':
    main()
