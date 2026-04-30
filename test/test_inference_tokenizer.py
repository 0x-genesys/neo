"""
Test inference tokenizer loading.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_tiktoken_wrapper():
    """Test that TiktokenWrapper works correctly."""
    print("\n" + "="*80)
    print("TEST: Tiktoken Wrapper")
    print("="*80)
    
    try:
        import tiktoken
        
        # Create wrapper
        encoding = tiktoken.get_encoding("cl100k_base")
        
        class TiktokenWrapper:
            def __init__(self, encoding):
                self.encoding = encoding
                self.vocab_size = encoding.n_vocab
                self.eos_token = "<|endoftext|>"
                self.pad_token = "<|endoftext|>"
                # Use encode_single_token for special tokens
                self.eos_token_id = encoding.encode_single_token(self.eos_token)
                self.pad_token_id = self.eos_token_id
            
            def encode(self, text, **kwargs):
                return self.encoding.encode(text, allowed_special='all')
            
            def decode(self, tokens, **kwargs):
                return self.encoding.decode(tokens)
            
            def __len__(self):
                return self.vocab_size
        
        tokenizer = TiktokenWrapper(encoding)
        
        # Test encoding
        text = "Hello, world!"
        token_ids = tokenizer.encode(text)
        print(f"✅ Encoded '{text}' to {len(token_ids)} tokens")
        
        # Test decoding
        decoded = tokenizer.decode(token_ids)
        print(f"✅ Decoded back to: '{decoded}'")
        
        # Test vocab size
        print(f"✅ Vocab size: {tokenizer.vocab_size:,}")
        print(f"✅ Vocab size via len(): {len(tokenizer):,}")
        
        # Test special tokens
        print(f"✅ EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
        print(f"✅ PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_loading():
    """Test loading a checkpoint and extracting config."""
    print("\n" + "="*80)
    print("TEST: Checkpoint Config Loading")
    print("="*80)
    
    try:
        import torch
        from pathlib import Path
        
        # Find a checkpoint
        checkpoint_paths = [
            "checkpoints/production/best_model.pt",
            "checkpoints/production/checkpoint.pt",
        ]
        
        checkpoint_path = None
        for path in checkpoint_paths:
            if Path(path).exists():
                checkpoint_path = path
                break
        
        if not checkpoint_path:
            print("⚠️  No checkpoint found, skipping test")
            return True
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check if config exists
        if 'config' not in checkpoint:
            print("❌ Config not found in checkpoint")
            return False
        
        config = checkpoint['config']
        print(f"✅ Config loaded from checkpoint")
        
        # Check tokenizer config
        if 'tokenizer' in config:
            tokenizer_type = config['tokenizer'].get('type', 'unknown')
            print(f"✅ Tokenizer type: {tokenizer_type}")
        else:
            print("⚠️  No tokenizer config found")
        
        # Check model config
        if 'model' in config:
            print(f"✅ Model config found:")
            print(f"   - Layers: {config['model'].get('num_layers', 'N/A')}")
            print(f"   - Dimensions: {config['model'].get('d_model', 'N/A')}")
            print(f"   - Heads: {config['model'].get('num_heads', 'N/A')}")
            print(f"   - Context: {config['model'].get('context_length', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("INFERENCE TOKENIZER TESTS")
    print("="*80)
    
    results = []
    
    # Test 1: Tiktoken wrapper
    results.append(("Tiktoken Wrapper", test_tiktoken_wrapper()))
    
    # Test 2: Checkpoint loading
    results.append(("Checkpoint Config Loading", test_checkpoint_loading()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
