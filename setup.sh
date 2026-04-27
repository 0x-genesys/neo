#!/bin/bash

# Production-Ready Transformer Training Setup Script
# Supports Mac (Intel/Apple Silicon), Linux, and Windows (via Git Bash)

set -e  # Exit on error

echo "=================================="
echo "🚀 Transformer Training Setup"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Windows;;
    MINGW*)     MACHINE=Windows;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo "Detected OS: ${MACHINE}"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: ${PYTHON_VERSION}"

# Check if Python 3.8+ is installed
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}❌ Python 3.8+ is required. Found: ${PYTHON_VERSION}${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python version OK${NC}"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo "Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install PyTorch based on platform
echo "Installing PyTorch..."
if [ "$MACHINE" = "Mac" ]; then
    # Check if Apple Silicon
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ]; then
        echo "Detected Apple Silicon (M1/M2/M3)"
        echo "Installing PyTorch with MPS support..."
        pip install torch torchvision torchaudio
    else
        echo "Detected Intel Mac"
        echo "Installing PyTorch..."
        pip install torch torchvision torchaudio
    fi
elif [ "$MACHINE" = "Linux" ]; then
    echo "Detected Linux"
    echo "Installing PyTorch with CUDA support (if available)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch (CPU version)..."
    pip install torch torchvision torchaudio
fi
echo -e "${GREEN}✅ PyTorch installed${NC}"
echo ""

# Install other requirements
echo "Installing other dependencies..."
pip install transformers datasets tokenizers sentencepiece
pip install tqdm pyyaml tensorboard wandb huggingface-hub
pip install numpy pandas scikit-learn matplotlib seaborn
pip install pytest black flake8
echo -e "${GREEN}✅ All dependencies installed${NC}"
echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p data
mkdir -p outputs
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Test device detection
echo "Testing device detection..."
python3 -c "from src.device_utils import select_device; select_device('auto', verbose=True)"
echo ""

# Check HuggingFace CLI
echo "Checking HuggingFace CLI..."
if command -v huggingface-cli &> /dev/null; then
    echo -e "${GREEN}✅ HuggingFace CLI installed${NC}"
    echo ""
    echo "To login to HuggingFace (for accessing gated datasets):"
    echo "  huggingface-cli login"
else
    echo -e "${YELLOW}⚠️  HuggingFace CLI not found in PATH${NC}"
    echo "You can still use the library, but CLI commands won't work"
fi
echo ""

# Print summary
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. (Optional) Login to HuggingFace:"
echo "   huggingface-cli login"
echo ""
echo "3. Test the setup:"
echo "   python src/device_utils.py"
echo ""
echo "4. Download a dataset (optional):"
echo "   python scripts/download_data.py --dataset tiny_shakespeare"
echo ""
echo "5. Start training:"
echo "   python train.py --config config/model_config.yaml"
echo ""
echo "6. Or use the quick start script:"
echo "   bash quick_start.sh"
echo ""
echo "=================================="
echo "📚 Documentation:"
echo "=================================="
echo "  • README.md - Project overview"
echo "  • DATASETS.md - Available datasets"
echo "  • GETTING_STARTED.md - Detailed guide"
echo "=================================="
