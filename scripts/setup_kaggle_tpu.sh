#!/bin/bash
# Setup script for Kaggle TPU environment

echo "================================================================================"
echo "Kaggle TPU Setup"
echo "================================================================================"
echo ""

# Check if running on Kaggle
if [ ! -d "/kaggle" ]; then
    echo "⚠️  This script is designed for Kaggle environment"
    echo "   Current environment does not appear to be Kaggle"
    exit 1
fi

echo "✅ Kaggle environment detected"
echo ""

# Check if TPU is enabled
if [ ! -d "/dev/accel0" ] && [ ! -f "/sys/class/accel/accel0/device/chip" ]; then
    echo "❌ TPU not detected!"
    echo ""
    echo "To enable TPU in Kaggle:"
    echo "1. Go to notebook settings (gear icon)"
    echo "2. Under 'Accelerator', select 'TPU v3-8'"
    echo "3. Click 'Save'"
    echo "4. Restart the notebook"
    exit 1
fi

echo "✅ TPU hardware detected"
echo ""

# Install torch_xla
echo "📦 Installing torch_xla using official setup script..."
echo "   This may take a few minutes..."
echo ""

# Download and run the official PyTorch XLA setup script (as per Kaggle docs)
curl -s https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o /tmp/pytorch-xla-env-setup.py

# Run setup script with nightly version and required packages
python3 /tmp/pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev

if [ $? -eq 0 ]; then
    echo "✅ torch_xla installed successfully"
else
    echo "❌ Failed to install torch_xla"
    echo ""
    echo "Try manual installation:"
    echo "  curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py"
    echo "  python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev"
    exit 1
fi

echo ""

# Verify installation
echo "🔍 Verifying torch_xla installation..."
python3 -c "
import torch_xla
import torch_xla.core.xla_model as xm
print(f'✅ torch_xla version: {torch_xla.__version__}')
print(f'✅ TPU device: {xm.xla_device()}')
print(f'✅ TPU cores: {xm.xrt_world_size()}')
print(f'✅ TPU ordinal: {xm.get_ordinal()}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✅ Kaggle TPU Setup Complete!"
    echo "================================================================================"
    echo ""
    echo "You can now train with TPU:"
    echo "  python train.py --config config/auto_training_117m_balanced.yaml --tpu"
    echo ""
    echo "Or let auto-detection handle it:"
    echo "  python train.py --config config/auto_training_117m_balanced.yaml"
    echo ""
else
    echo ""
    echo "❌ torch_xla verification failed"
    echo "Please check the installation and try again"
    exit 1
fi
