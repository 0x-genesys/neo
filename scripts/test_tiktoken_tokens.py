#!/usr/bin/env python3
"""
Test script to check which tiktoken encodings support ChatML tokens.
"""
import tiktoken

print("Testing tiktoken encodings for ChatML support...\n")

encodings_to_test = ["cl100k_base", "p50k_base", "r50k_base"]

for enc_name in encodings_to_test:
    print(f"{'='*80}")
    print(f"Encoding: {enc_name}")
    print(f"{'='*80}")
    
    try:
        enc = tiktoken.get_encoding(enc_name)
        print(f"Vocab size: {enc.n_vocab}")
        
        # Test special tokens
        test_tokens = [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|fim_suffix|>",
        ]
        
        print(f"\nSpecial tokens:")
        for token in test_tokens:
            try:
                token_id = enc.encode_single_token(token)
                print(f"  ✅ {token:20s} -> ID: {token_id}")
            except KeyError:
                # Try encoding normally to see how it's split
                encoded = enc.encode(token, allowed_special='all')
                print(f"  ❌ {token:20s} -> Split into {len(encoded)} tokens: {encoded[:5]}...")
        
        print()
    except Exception as e:
        print(f"  Error: {e}\n")

# Check if there's a ChatML-specific encoding
print(f"{'='*80}")
print("Checking for ChatML encoding...")
print(f"{'='*80}")

try:
    # Try to get a ChatML encoding
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    print(f"✅ Found encoding for gpt-3.5-turbo")
    print(f"   Name: {enc.name}")
    print(f"   Vocab size: {enc.n_vocab}")
    
    # Test ChatML tokens
    for token in ["<|im_start|>", "<|im_end|>"]:
        try:
            token_id = enc.encode_single_token(token)
            print(f"   ✅ {token} -> ID: {token_id}")
        except KeyError:
            encoded = enc.encode(token, allowed_special='all')
            print(f"   ❌ {token} -> Split into {len(encoded)} tokens")
except Exception as e:
    print(f"❌ Error: {e}")
