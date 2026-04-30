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

# Note: We'll verify TPU after installing torch_xla
# Kaggle TPU might not expose /dev/accel0 until torch_xla is loaded
echo "📝 Note: TPU verification will happen after torch_xla installation"
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
echo "🔍 Verifying torch_xla installation and TPU availability..."
python3 -c "
import sys
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    print(f'✅ torch_xla version: {torch_xla.__version__}')
    
    try:
        device = xm.xla_device()
        print(f'✅ TPU device: {device}')
        print(f'✅ TPU cores: {xm.xrt_world_size()}')
        print(f'✅ TPU ordinal: {xm.get_ordinal()}')
    except Exception as e:
        print(f'❌ TPU not available: {e}')
        print('')
        print('To enable TPU in Kaggle:')
        print('1. Go to notebook settings (gear icon)')
        print('2. Under \"Accelerator\", select \"TPU v3-8\"')
        print('3. Click \"Save\"')
        print('4. Restart the notebook')
        print('')
        print('Note: Make sure you see \"TPU v3-8\" in the accelerator dropdown')
        sys.exit(1)
except ImportError as e:
    print(f'❌ torch_xla import failed: {e}')
    sys.exit(1)
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
    echo "================================================================================"
    echo "❌ TPU Setup Failed"
    echo "================================================================================"
    echo ""
    echo "Possible reasons:"
    echo "1. TPU not enabled in Kaggle notebook settings"
    echo "2. Notebook needs to be restarted after enabling TPU"
    echo "3. TPU quota exhausted (check Kaggle account limits)"
    echo ""
    echo "Please check the settings and try again."
    exit 1
fi
