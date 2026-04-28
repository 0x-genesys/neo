#!/bin/bash

# Setup script for HuggingFace Hub integration

echo "=============================================================================="
echo "HUGGINGFACE HUB SETUP"
echo "=============================================================================="
echo ""

# Step 1: Install huggingface_hub
echo "Step 1: Installing huggingface_hub..."
pip install -q huggingface_hub

if [ $? -eq 0 ]; then
    echo "✅ huggingface_hub installed"
else
    echo "❌ Failed to install huggingface_hub"
    exit 1
fi

echo ""

# Step 2: Login to HuggingFace
echo "Step 2: Login to HuggingFace Hub"
echo ""
echo "You'll need your HuggingFace token from:"
echo "https://huggingface.co/settings/tokens"
echo ""

huggingface-cli login

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Logged in to HuggingFace Hub"
else
    echo ""
    echo "❌ Failed to login"
    exit 1
fi

echo ""

# Step 3: Verify repository access
echo "Step 3: Verifying repository access..."
echo ""

REPO_ID="0x-genesys/neo_weights_checkpoints"

python3 << EOF
from huggingface_hub import HfApi
import sys

try:
    api = HfApi()
    
    # Try to access the repo
    try:
        files = api.list_repo_files("$REPO_ID", repo_type="model")
        print(f"✅ Repository accessible: $REPO_ID")
        print(f"   Current files: {len(files)}")
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            print(f"⚠️  Repository not found: $REPO_ID")
            print(f"   Creating repository...")
            
            # Create the repository
            api.create_repo("$REPO_ID", repo_type="model", exist_ok=True)
            print(f"✅ Repository created: $REPO_ID")
        else:
            raise e
    
    print("")
    print("✅ Setup complete!")
    print("")
    print("Your training will now automatically upload checkpoints to:")
    print(f"https://huggingface.co/$REPO_ID")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================================================="
    echo "✅ HUGGINGFACE HUB SETUP COMPLETE"
    echo "=============================================================================="
    echo ""
    echo "Next steps:"
    echo "  1. Start training:"
    echo "     python train.py --config config/gpu_training_117m_1.5gb.yaml --multi-gpu"
    echo ""
    echo "  2. Monitor uploads:"
    echo "     https://huggingface.co/$REPO_ID"
    echo ""
    echo "  3. View documentation:"
    echo "     docs/HUGGINGFACE_HUB_INTEGRATION.md"
    echo ""
else
    echo ""
    echo "=============================================================================="
    echo "❌ SETUP FAILED"
    echo "=============================================================================="
    echo ""
    echo "Please check the error messages above and try again."
    echo ""
    exit 1
fi
