#!/bin/bash
# Verification script for inference functionality

echo "================================================================================"
echo "INFERENCE VERIFICATION SCRIPT"
echo "================================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

# Function to run test
run_test() {
    local description="$1"
    local command="$2"
    local expected="$3"
    
    echo "--------------------------------------------------------------------------------"
    echo "TEST: $description"
    echo "Command: $command"
    echo ""
    
    output=$(eval "$command" 2>&1)
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        if [ -z "$expected" ] || echo "$output" | grep -q "$expected"; then
            echo -e "${GREEN}✅ PASSED${NC}"
            ((PASSED++))
        else
            echo -e "${RED}❌ FAILED${NC} - Expected output not found: $expected"
            echo "Output preview:"
            echo "$output" | head -20
            ((FAILED++))
        fi
    else
        echo -e "${RED}❌ FAILED${NC} - Exit code: $exit_code"
        echo "Error:"
        echo "$output" | tail -10
        ((FAILED++))
    fi
    echo ""
}

# Test 1: GPT-2 checkpoint without config
run_test \
    "GPT-2 checkpoint (no config)" \
    "venv/bin/python3 src/inference.py --model checkpoints/production/best_model.pt --prompt 'Hello' --max-tokens 20" \
    "Generated Text"

# Test 2: GPT-2 checkpoint with config
run_test \
    "GPT-2 checkpoint (with config merge)" \
    "venv/bin/python3 src/inference.py --model checkpoints/production/best_model.pt --config config/inference.yaml --prompt 'Hello' --max-tokens 20" \
    "Merged config"

# Test 3: Custom temperature
run_test \
    "Custom temperature parameter" \
    "venv/bin/python3 src/inference.py --model checkpoints/production/best_model.pt --prompt 'Hello' --temperature 0.5 --max-tokens 20" \
    "Generated Text"

# Test 4: Tiktoken checkpoint (if available)
if [ -f "checkpoints/best_model_step_2500.pt" ]; then
    echo -e "${YELLOW}⚠️  Tiktoken test may take 30-60 seconds on CPU...${NC}"
    run_test \
        "Tiktoken checkpoint (no config)" \
        "venv/bin/python3 src/inference.py --model checkpoints/best_model_step_2500.pt --prompt 'Hi' --max-tokens 10" \
        "Generated Text"
else
    echo "⚠️  Skipping tiktoken test - checkpoint not found"
fi

# Summary
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo "Total: $((PASSED + FAILED))"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some tests failed${NC}"
    exit 1
fi
