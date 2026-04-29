#!/bin/bash
# Upgrade PyTorch to latest version compatible with your system

echo "🔥 PyTorch Upgrade Script"
echo "=========================="

# Detect system
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected: macOS"
    
    # Check for Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        echo "Platform: Apple Silicon (M1/M2/M3)"
        echo ""
        echo "Installing PyTorch with MPS support..."
        pip install --upgrade torch torchvision torchaudio
    else
        echo "Platform: Intel Mac"
        echo ""
        echo "Installing PyTorch (CPU)..."
        pip install --upgrade torch torchvision torchaudio
    fi
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected: Linux"
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "Platform: Linux with NVIDIA GPU"
        echo ""
        echo "Installing PyTorch with CUDA 11.8..."
        pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "Platform: Linux (CPU only)"
        echo ""
        echo "Installing PyTorch (CPU)..."
        pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
else
    echo "Platform: Unknown"
    echo ""
    echo "Installing PyTorch (default)..."
    pip install --upgrade torch torchvision torchaudio
fi

echo ""
echo "✅ PyTorch upgrade complete!"
echo ""
echo "Verify installation:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"

echo ""
echo "Run environment check:"
echo "  python scripts/fix_environment.py"