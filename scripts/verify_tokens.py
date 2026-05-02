import sys
import os

# Add the root directory to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer_utils import load_tokenizer

def verify_tokens():
    print("🔍 Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    test_text = "<|im_start|>thought\n"
    print(f"🔍 Encoding test text: {repr(test_text)}")
    
    encoded = tokenizer.encode(test_text)
    decoded_tokens = [tokenizer.decode([t]) for t in encoded]
    
    print(f"✅ Encoded length: {len(encoded)}")
    print(f"✅ Token IDs: {encoded}")
    print(f"✅ Decoded tokens: {decoded_tokens}")
    
    # Assertions as requested
    im_start_encoded = tokenizer.encode("<|im_start|>")
    print(f"🔍 Encoding '<|im_start|>': {im_start_encoded}")
    
    assert len(im_start_encoded) == 1, f"Expected length 1 for <|im_start|>, got {len(im_start_encoded)}"
    assert im_start_encoded[0] == 100264, f"Expected ID 100264 for <|im_start|>, got {im_start_encoded[0]}"
    
    im_end_encoded = tokenizer.encode("<|im_end|>")
    print(f"🔍 Encoding '<|im_end|>': {im_end_encoded}")
    assert len(im_end_encoded) == 1, f"Expected length 1 for <|im_end|>, got {len(im_end_encoded)}"
    assert im_end_encoded[0] == 100265, f"Expected ID 100265 for <|im_end|>, got {im_end_encoded[0]}"
    
    print("\n✨ ALL TOKEN VERIFICATIONS PASSED! ✨")

if __name__ == "__main__":
    try:
        verify_tokens()
    except AssertionError as e:
        print(f"❌ VERIFICATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ AN UNEXPECTED ERROR OCCURRED: {e}")
        sys.exit(1)
