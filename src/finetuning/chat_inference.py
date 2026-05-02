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

# Check PyTorch version
def check_pytorch_version():
    """Check if PyTorch version is compatible."""
    torch_version = torch.__version__.split('+')[0]  # Remove +cu118 suffix if present
    major, minor = map(int, torch_version.split('.')[:2])
    
    if major < 2 or (major == 2 and minor < 2):
        print(f"\n❌ ERROR: PyTorch {torch_version} is too old")
        print(f"   Required: PyTorch >= 2.2.0")
        print(f"\n💡 To upgrade:")
        print(f"   pip install --upgrade torch torchvision torchaudio")
        print()
        sys.exit(1)
    elif major == 2 and minor < 4:
        print(f"\n⚠️  Note: PyTorch {torch_version} detected")
        print(f"   Some checkpoints saved with PyTorch 2.4+ may not load")
        print(f"   Current version works for most use cases")
        print()
    return True

check_pytorch_version()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model import create_model
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
            
            try:
                from huggingface_hub import snapshot_download
                import tempfile
                
                # Download the entire finetune/adapter_remote directory
                cache_dir = Path(tempfile.gettempdir()) / "hf_cache" / model_repo.replace("/", "_")
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                # Download with pattern matching
                downloaded_path = snapshot_download(
                    repo_id=model_repo,
                    allow_patterns=f"finetune/{adapter_remote}/*",
                    cache_dir=str(cache_dir),
                    repo_type="model",
                )
                
                adapter_path = Path(downloaded_path) / "finetune" / adapter_remote
                
                if not adapter_path.exists():
                    raise FileNotFoundError(f"Adapter not found at {adapter_path}")
                
                print(f"✅ Downloaded to: {adapter_path}")
                
            except Exception as e:
                print(f"❌ Error downloading adapter: {e}")
                print(f"\n💡 Make sure the adapter exists at:")
                print(f"   https://huggingface.co/{model_repo}/tree/main/finetune/{adapter_remote}")
                raise
        
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
        
        # Try loading checkpoint with backward compatibility
        try:
            # Try modern PyTorch 2.4+ method first
            checkpoint = torch.load(base_model_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Fall back to older PyTorch method (no weights_only parameter)
            try:
                checkpoint = torch.load(base_model_path, map_location='cpu')
            except AttributeError as e:
                if '_rebuild_device_tensor_from_cpu_tensor' in str(e):
                    print(f"\n❌ Checkpoint incompatibility detected!")
                    print(f"   This checkpoint was saved with a newer PyTorch version.")
                    print(f"   Current PyTorch: {torch.__version__}")
                    print(f"\n💡 Workaround: Re-save the checkpoint with your current PyTorch version")
                    print(f"   Or use the checkpoint from the same PyTorch version it was trained with")
                    raise RuntimeError(f"Checkpoint incompatibility. Please re-save checkpoint with PyTorch {torch.__version__}") from e
                else:
                    raise
        except Exception as e:
            print(f"\n❌ Error loading checkpoint: {e}")
            print(f"   Checkpoint path: {base_model_path}")
            raise
        
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
        
        # Create base model using factory function
        self.model = create_model(self.config)
        
        # Load base weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"✅ Base model loaded")
        
        # Add generation_config attribute for PEFT compatibility
        # PEFT expects this attribute but our custom model doesn't have it
        # We'll add a dummy one since we use our own generate() method
        if not hasattr(self.model, 'generation_config'):
            from types import SimpleNamespace
            vocab_size = self.model.token_embedding.num_embeddings
            self.model.generation_config = SimpleNamespace(
                max_length=model_config['context_length'],
                pad_token_id=vocab_size - 1,
                eos_token_id=vocab_size - 1,
                bos_token_id=0,
            )
        
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
    
    def parse_response(self, generated_text: str, debug: bool = False) -> dict:
        """
        Parse generated text into thought and assistant response.
        
        Args:
            generated_text: Generated text from model
            debug: Whether to print debug information
            
        Returns:
            Dictionary with 'thought', 'response', and 'full_text' keys
        """
        result = {
            'thought': '',
            'response': '',
            'full_text': generated_text
        }
        
        if debug:
            print(f"\n{'='*80}")
            print("🔍 DEBUG - Raw Generated Text:")
            print(f"{'='*80}")
            print(generated_text)
            print(f"{'='*80}\n")
        
        # Extract thought block
        thought_start = f"{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['thought']}\n"
        thought_end = f"{SPECIAL_TOKENS['im_end']}"
        
        if thought_start in generated_text:
            thought_section = generated_text.split(thought_start)[1]
            if thought_end in thought_section:
                result['thought'] = thought_section.split(thought_end)[0].strip()
            else:
                # Model generated thought but no end tag - take everything until next tag or end
                next_tag = f"{SPECIAL_TOKENS['im_start']}"
                if next_tag in thought_section:
                    result['thought'] = thought_section.split(next_tag)[0].strip()
                else:
                    result['thought'] = thought_section.strip()
        
        # Extract assistant response
        assistant_start = f"{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['assistant']}\n"
        assistant_end = f"{SPECIAL_TOKENS['im_end']}"
        
        if assistant_start in generated_text:
            assistant_section = generated_text.split(assistant_start)[1]
            if assistant_end in assistant_section:
                result['response'] = assistant_section.split(assistant_end)[0].strip()
            else:
                # Model generated assistant but no end tag - take everything
                result['response'] = assistant_section.strip()
        else:
            # FALLBACK: Model didn't generate assistant tag
            # Check if there's content after thought block
            if result['thought'] and thought_end in generated_text:
                # Take everything after thought end as response
                after_thought = generated_text.split(thought_end, 1)[1].strip()
                if after_thought:
                    result['response'] = after_thought
            elif not result['thought']:
                # No thought, no assistant tag - use everything after user message
                user_end = f"{SPECIAL_TOKENS['im_end']}"
                if user_end in generated_text:
                    parts = generated_text.split(user_end)
                    if len(parts) > 2:  # System, User, then response
                        result['response'] = parts[-1].strip()
        
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
        debug: bool = False,
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
            debug: Whether to print debug information
            
        Returns:
            Dictionary with 'thought', 'response', and 'full_text'
        """
        # Format prompt
        prompt = self.format_prompt(user_message, include_thought=True)
        
        if debug:
            print(f"\n{'='*80}")
            print("🔍 DEBUG - Formatted Prompt:")
            print(f"{'='*80}")
            print(prompt)
            print(f"{'='*80}\n")
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt)
        
        if debug:
            print(f"🔍 DEBUG - Tokenization:")
            print(f"   Input length: {len(input_ids)} tokens")
            print(f"   First 10 tokens: {input_ids[:10]}")
            print(f"   Last 10 tokens: {input_ids[-10:]}\n")
        
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Generate using base model's generate method (bypass PEFT wrapper)
        # PEFT's generate() expects generation_config which our custom model doesn't have
        if hasattr(self.model, 'base_model'):
            # PEFT wrapped model - access the base model
            # But we want the LoRA weights applied, so we need to use the wrapped model
            # Let's try using the model directly first
            if debug:
                print(f"🔍 DEBUG - Model Structure:")
                print(f"   Model type: {type(self.model).__name__}")
                print(f"   Has base_model: {hasattr(self.model, 'base_model')}")
                if hasattr(self.model, 'base_model'):
                    print(f"   Base model type: {type(self.model.base_model).__name__}")
                    if hasattr(self.model.base_model, 'model'):
                        print(f"   Inner model type: {type(self.model.base_model.model).__name__}")
                print()
            
            # Use the PEFT model's forward, not generate
            # We'll use the base model's generate but with PEFT's forward
            base_model = self.model.base_model.model
        else:
            # Not wrapped
            base_model = self.model
        
        # Generate using model with LoRA applied
        # We need to use the model's forward pass which includes LoRA
        # But call the base model's generate method with the PEFT model's forward
        if hasattr(self.model, 'base_model'):
            # PEFT wrapped model
            if debug:
                print(f"🔍 DEBUG - Using PEFT-wrapped model for generation")
                print(f"   This ensures LoRA weights are applied\n")
            
            # Use the base model's generate but it will call the PEFT model's forward
            # The generate method is on the base model, but forward goes through PEFT
            model_to_use = self.model.base_model.model
            
            # Temporarily replace the model's forward with PEFT's forward
            original_forward = model_to_use.forward
            model_to_use.forward = self.model.forward
            
            try:
                output_ids = model_to_use.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            finally:
                # Restore original forward
                model_to_use.forward = original_forward
        else:
            # Not wrapped - use directly
            if debug:
                print(f"🔍 DEBUG - Using unwrapped model\n")
            
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        
        if debug:
            print(f"🔍 DEBUG - Generation:")
            print(f"   Input length: {len(input_ids[0])} tokens")
            print(f"   Output length: {len(output_ids[0])} tokens")
            print(f"   Generated {len(output_ids[0]) - len(input_ids[0])} new tokens")
            print(f"   Output shape: {output_ids.shape}")
            print(f"   First 10 output tokens: {output_ids[0][:10].tolist()}")
            print(f"   Last 10 output tokens: {output_ids[0][-10:].tolist()}\n")
        
        # Decode
        generated_text = self.tokenizer.decode(output_ids[0].tolist())
        
        if debug:
            print(f"🔍 DEBUG - Decoded length: {len(generated_text)} characters\n")
        
        # Parse response
        result = self.parse_response(generated_text, debug=debug)
        
        return result
    
    def interactive_mode(
        self,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        show_thought: bool = True,
        debug: bool = False,
    ):
        """
        Run interactive chat mode.
        
        Args:
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            show_thought: Whether to show thought process
            debug: Whether to enable debug mode
        """
        print("\n" + "="*80)
        print("💬 Interactive Chat Mode")
        print("="*80)
        print("Commands:")
        print("  - Type your message and press Enter")
        print("  - 'quit' or 'exit' to stop")
        print("  - 'config' to see current settings")
        print("  - 'thought on/off' to toggle thought display")
        print("  - 'debug on/off' to toggle debug mode")
        print("  - 'set <param> <value>' to change settings")
        print("="*80 + "\n")
        
        settings = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'show_thought': show_thought,
            'debug': debug,
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
                
                if user_input.lower().startswith('debug '):
                    toggle = user_input.split()[1].lower()
                    if toggle == 'on':
                        settings['debug'] = True
                        print("✅ Debug mode enabled")
                    elif toggle == 'off':
                        settings['debug'] = False
                        print("✅ Debug mode disabled")
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
                    debug=settings.get('debug', False),
                )
                
                # Display thought if enabled
                if settings['show_thought'] and result['thought']:
                    print(f"\n💭 [Thought: {result['thought']}]\n")
                
                # Display response
                if result['response']:
                    print(result['response'])
                else:
                    # No response parsed - show raw output for debugging
                    print("\n⚠️  No response parsed. Raw output:")
                    print(f"{'='*80}")
                    print(result['full_text'])
                    print(f"{'='*80}")
                
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
  # Interactive mode with remote models (default - no arguments needed!)
  python src/finetuning/chat_inference.py --interactive
  
  # Interactive mode with local models
  python src/finetuning/chat_inference.py \\
      --base-model checkpoints/best_model.pt \\
      --adapter finetuned_model_gpu/best_model \\
      --interactive
  
  # Single prompt with remote models
  python src/finetuning/chat_inference.py \\
      --prompt "What is 2+2?"
  
  # Single prompt with local models
  python src/finetuning/chat_inference.py \\
      --base-model checkpoints/best_model.pt \\
      --adapter finetuned_model_gpu/best_model \\
      --prompt "Solve: 5x + 3 = 18"
  
  # Show thought process
  python src/finetuning/chat_inference.py \\
      --prompt "Explain quantum computing" \\
      --show-thought
  
  # Custom HuggingFace repository
  python src/finetuning/chat_inference.py \\
      --model-repo username/my-repo \\
      --base-model-remote my_model.pt \\
      --adapter-remote finetune/my_adapter \\
      --interactive
  
  # Custom generation parameters
  python src/finetuning/chat_inference.py \\
      --temperature 0.9 \\
      --top-k 100 \\
      --max-tokens 512 \\
      --interactive
  
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
        default=None,
        help='Path to local base model (.pt file). If not provided, uses --base-model-remote'
    )
    parser.add_argument(
        '--base-model-remote',
        type=str,
        default='best_model.pt',
        help='Remote base model filename from HuggingFace Hub (default: best_model.pt)'
    )
    parser.add_argument(
        '--adapter',
        type=str,
        default=None,
        help='Path to local LoRA adapter directory. If not provided, uses --adapter-remote'
    )
    parser.add_argument(
        '--adapter-remote',
        type=str,
        default='finetune/chat_adapter',
        help='Remote adapter path in HuggingFace Hub (default: finetune/chat_adapter)'
    )
    parser.add_argument(
        '--model-repo',
        type=str,
        default='0x-genesys/neo_weights_checkpoints',
        help='HuggingFace repository ID (default: 0x-genesys/neo_weights_checkpoints)'
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
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (show raw generation output)'
    )
    
    args = parser.parse_args()
    
    # Determine model paths - prioritize local, fall back to remote
    base_model_path = args.base_model
    base_model_remote = None if args.base_model else args.base_model_remote
    
    adapter_path = args.adapter
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
            debug=args.debug,
        )
    elif args.prompt:
        result = generator.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            show_thought=args.show_thought,
            debug=args.debug,
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
