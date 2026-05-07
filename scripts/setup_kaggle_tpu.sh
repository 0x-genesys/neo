#!/bin/bash
#
# Kaggle TPU Setup Script for torch_xla 2.9.0 + Python 3.12
# Ensures binary alignment and PJRT compatibility
#

set -e

echo "================================================================================"
echo "Kaggle TPU Environment Setup - torch_xla 2.9.0 + Python 3.12"
echo "================================================================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $PYTHON_VERSION"

if [ "$PYTHON_VERSION" != "3.12" ]; then
    echo "⚠️  Warning: This script is optimized for Python 3.12"
    echo "   Current version: $PYTHON_VERSION"
    echo "   Proceeding anyway..."
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# CRITICAL: Install torch_xla 2.9.0 ecosystem with binary alignment
echo ""
echo "================================================================================"
echo "Installing PyTorch + torch_xla 2.9.0 (PJRT-compatible)"
echo "================================================================================"
echo ""

# # Uninstall any existing torch packages to prevent conflicts
# echo "Removing any existing PyTorch installations..."
# pip uninstall -y torch torchvision torch_xla 2>/dev/null || true

# Install torch 2.9.0 with CUDA 12.8 support
echo ""
echo "Installing torch 2.9.0..."
# pip install --force-reinstall torch==2.9.0

# # Install torch_xla 2.9.0 with TPU support from libtpu-releases
# echo ""
# echo "Installing torch_xla[tpu]==2.9.0..."
# pip install --force-reinstall \
#     --extra-index-url https://storage.googleapis.com/libtpu-releases/index.html \
#     torch_xla[tpu]==2.9.0

# # Install torchvision 0.24.0 (compatible with torch 2.9.0)
# echo ""
# echo "Installing torchvision 0.24.0..."
# pip install --force-reinstall torchvision==0.24.0

# # Install fsspec with correct version for datasets compatibility
# echo ""
# echo "Installing fsspec 2026.2.0..."
# pip install --force-reinstall fsspec==2026.2.0

# echo ""
# echo "================================================================================"
# echo "Installing Project Dependencies"
# echo "================================================================================"
# echo ""

# # Install other requirements
# if [ -f "requirements.txt" ]; then
#     echo "Installing from requirements.txt..."
#     # Skip torch packages as they're already installed
#     grep -v "^torch" requirements.txt | grep -v "^#" | grep -v "^$" | \
#         xargs -I {} pip install {}
#     echo "✅ Dependencies installed"
# else
#     echo "⚠️  requirements.txt not found, skipping..."
# fi

pip install --force-reinstall --no-cache-dir \
    torch==2.9.0 \
    torch_xla[tpu]==2.9.0 \
    torchvision==0.24.0 \
    fsspec==2026.2.0 \
    -f https://storage.googleapis.com/libtpu-releases/index.html

# Verify installation
echo ""
echo "================================================================================"
echo "Verifying Installation"
echo "================================================================================"
echo ""

python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✅ torch: {torch.__version__}")
except ImportError as e:
    print(f"❌ torch: {e}")
    sys.exit(1)

try:
    import torch_xla
    print(f"✅ torch_xla: {torch_xla.__version__}")
except ImportError as e:
    print(f"❌ torch_xla: {e}")
    sys.exit(1)

try:
    import torch_xla.core.xla_model as xm
    print(f"✅ torch_xla.core.xla_model imported")
except ImportError as e:
    print(f"❌ torch_xla.core.xla_model: {e}")
    sys.exit(1)

try:
    import torchvision
    print(f"✅ torchvision: {torchvision.__version__}")
except ImportError as e:
    print(f"⚠️  torchvision: {e}")

try:
    import fsspec
    print(f"✅ fsspec: {fsspec.__version__}")
except ImportError as e:
    print(f"⚠️  fsspec: {e}")

print("\n" + "="*80)
print("✅ Installation Complete!")
print("="*80)
print("\nIMPORTANT: Before running training, ensure these environment variables are set:")
print("  export PJRT_DEVICE=TPU")
print("  export TPU_PROCESS_ADDRESSES=local")
print("  export TPU_NUM_DEVICES=8")
print("\nOr use the train.py script which sets them automatically.")
print("\nTo verify TPU availability, run:")
print("  python scripts/check_tpu.py")
EOF

echo ""
echo "================================================================================"
echo "Setup Complete!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "1. Verify TPU: python scripts/check_tpu.py"
echo "2. Start training: python train.py --config config/auto_training_117m_balanced.yaml --tpu"
echo ""
