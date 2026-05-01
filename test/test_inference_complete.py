"""
Complete inference testing - all scenarios.
"""
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_inference_test(description, command, expected_in_output=None, should_fail=False):
    """Run an inference test."""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"{'='*80}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout + result.stderr
        
        if should_fail:
            if result.returncode != 0:
                print(f"✅ Failed as expected")
                return True
            else:
                print(f"❌ Should have failed but succeeded")
                return False
        else:
            if result.returncode == 0:
                if expected_in_output:
                    if expected_in_output in output:
                        print(f"✅ Success - found expected output: '{expected_in_output}'")
                        return True
                    else:
                        print(f"❌ Success but missing expected output: '{expected_in_output}'")
                        print(f"Output preview: {output[:500]}")
                        return False
                else:
                    print(f"✅ Success")
                    return True
            else:
                print(f"❌ Failed with exit code {result.returncode}")
                print(f"Error: {result.stderr[:500]}")
                return False
                
    except subprocess.TimeoutExpired:
        print(f"❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def main():
    """Run all inference tests."""
    print("\n" + "="*80)
    print("COMPLETE INFERENCE TESTS")
    print("="*80)
    
    results = []
    
    # Test 1: Tiktoken checkpoint without config
    results.append((
        "Tiktoken checkpoint (no config)",
        run_inference_test(
            "Tiktoken checkpoint without config",
            "venv/bin/python3 src/inference.py --model checkpoints/best_model_step_2500.pt --prompt 'Hello' --max-tokens 20",
            expected_in_output="tiktoken"
        )
    ))
    
    # Test 2: Tiktoken checkpoint with config
    results.append((
        "Tiktoken checkpoint (with config)",
        run_inference_test(
            "Tiktoken checkpoint with config merge",
            "venv/bin/python3 src/inference.py --model checkpoints/best_model_step_2500.pt --config config/inference.yaml --prompt 'Hello' --max-tokens 20",
            expected_in_output="Merged config"
        )
    ))
    
    # Test 3: GPT-2 checkpoint without config
    results.append((
        "GPT-2 checkpoint (no config)",
        run_inference_test(
            "GPT-2 checkpoint without config",
            "venv/bin/python3 src/inference.py --model checkpoints/production/best_model.pt --prompt 'Hello' --max-tokens 20",
            expected_in_output="gpt2"
        )
    ))
    
    # Test 4: GPT-2 checkpoint with config
    results.append((
        "GPT-2 checkpoint (with config)",
        run_inference_test(
            "GPT-2 checkpoint with config merge",
            "venv/bin/python3 src/inference.py --model checkpoints/production/best_model.pt --config config/inference.yaml --prompt 'Hello' --max-tokens 20",
            expected_in_output="Merged config"
        )
    ))
    
    # Test 5: Custom temperature
    results.append((
        "Custom temperature",
        run_inference_test(
            "Custom temperature parameter",
            "venv/bin/python3 src/inference.py --model checkpoints/production/best_model.pt --prompt 'Hello' --temperature 0.5 --max-tokens 20",
            expected_in_output="Generated Text"
        )
    ))
    
    # Test 6: Multiple samples
    results.append((
        "Multiple samples",
        run_inference_test(
            "Generate multiple samples",
            "venv/bin/python3 src/inference.py --model checkpoints/production/best_model.pt --prompt 'Hello' --num-samples 2 --max-tokens 20",
            expected_in_output="Sample 2"
        )
    ))
    
    # Test 7: Invalid model path (should fail)
    results.append((
        "Invalid model path",
        run_inference_test(
            "Invalid model path should fail",
            "venv/bin/python3 src/inference.py --model nonexistent.pt --prompt 'Hello'",
            should_fail=True
        )
    ))
    
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
