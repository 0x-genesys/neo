"""
Test checkpoint saving to ensure files are complete before upload.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_checkpoint_save():
    """Test that checkpoint saves completely before returning."""
    print("Testing checkpoint save integrity...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 100)
    )
    
    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'step': 1000,
        'epoch': 1
    }
    
    # Test 1: Standard torch.save
    print("\n1. Testing torch.save()...")
    test_path = Path("test_checkpoint_torch.pt")
    torch.save(checkpoint, test_path)
    size_torch = test_path.stat().st_size
    print(f"   Size: {size_torch / 1024:.2f} KB")
    test_path.unlink()
    
    # Test 2: XLA serialization (if available)
    try:
        import torch_xla.utils.serialization as xser
        import torch_xla.core.xla_model as xm
        
        print("\n2. Testing xser.save() without wait...")
        test_path = Path("test_checkpoint_xser_nowait.pt")
        xser.save(checkpoint, str(test_path))
        size_xser_nowait = test_path.stat().st_size
        print(f"   Size immediately: {size_xser_nowait / 1024:.2f} KB")
        test_path.unlink()
        
        print("\n3. Testing xser.save() with mark_step()...")
        test_path = Path("test_checkpoint_xser_mark.pt")
        xser.save(checkpoint, str(test_path))
        xm.mark_step()
        size_xser_mark = test_path.stat().st_size
        print(f"   Size after mark_step: {size_xser_mark / 1024:.2f} KB")
        test_path.unlink()
        
        print("\n4. Testing xser.save() with wait_device_ops()...")
        test_path = Path("test_checkpoint_xser_wait.pt")
        xser.save(checkpoint, str(test_path))
        xm.mark_step()
        xm.wait_device_ops()
        size_xser_wait = test_path.stat().st_size
        print(f"   Size after wait_device_ops: {size_xser_wait / 1024:.2f} KB")
        
        # Verify we can load it
        loaded = torch.load(test_path)
        assert 'model_state_dict' in loaded
        assert loaded['step'] == 1000
        print(f"   ✅ Checkpoint loads correctly")
        test_path.unlink()
        
        # Compare sizes
        print("\n5. Size comparison:")
        print(f"   torch.save:                {size_torch / 1024:.2f} KB")
        print(f"   xser.save (no wait):       {size_xser_nowait / 1024:.2f} KB")
        print(f"   xser.save (mark_step):     {size_xser_mark / 1024:.2f} KB")
        print(f"   xser.save (wait_device):   {size_xser_wait / 1024:.2f} KB")
        
        if size_xser_wait < size_torch * 0.9:
            print(f"\n   ⚠️  WARNING: xser.save produces smaller files!")
            print(f"   This might indicate incomplete saves.")
        else:
            print(f"\n   ✅ File sizes are consistent")
            
    except ImportError:
        print("\n   ⚠️  torch_xla not available, skipping XLA tests")
    
    print("\n✅ Test complete")

if __name__ == "__main__":
    test_checkpoint_save()
