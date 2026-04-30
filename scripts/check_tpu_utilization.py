"""
Check TPU utilization and verify all cores are being used.
"""

import torch
import time

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.runtime as xr
    
    print("="*80)
    print("TPU Utilization Check")
    print("="*80)
    
    # Check world size
    try:
        world_size = xr.world_size()
        print(f"World size (cores): {world_size}")
    except:
        print("World size: Unable to determine")
    
    # Check ordinal
    try:
        ordinal = xr.global_ordinal()
        print(f"Current ordinal (rank): {ordinal}")
    except:
        print("Ordinal: Unable to determine")
    
    # Get device
    device = xm.xla_device()
    print(f"XLA device: {device}")
    
    # Check if we're in a spawned process
    import os
    print(f"Process ID: {os.getpid()}")
    print(f"TPU_PROCESS_ADDRESSES: {os.environ.get('TPU_PROCESS_ADDRESSES', 'Not set')}")
    
    # Test data parallelism
    print("\n" + "="*80)
    print("Testing Data Parallelism")
    print("="*80)
    
    # Create dummy data
    batch_size = 32
    seq_len = 128
    vocab_size = 1000
    
    dummy_data = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dummy_data),
        batch_size=8,
        shuffle=False
    )
    
    # Test with ParallelLoader
    print(f"\nOriginal loader batches: {len(dummy_loader)}")
    
    para_loader = pl.ParallelLoader(dummy_loader, [device])
    per_device = para_loader.per_device_loader(device)
    
    batch_count = 0
    for batch in per_device:
        batch_count += 1
        if batch_count == 1:
            print(f"First batch shape: {batch[0].shape}")
    
    print(f"ParallelLoader batches processed: {batch_count}")
    
    # Check if data is being distributed
    if batch_count == len(dummy_loader):
        print("\n⚠️  WARNING: ParallelLoader processed same number of batches as original")
        print("   This suggests data is NOT being distributed across cores")
        print("   You may be using only 1 core!")
    else:
        print(f"\n✅ Data distribution detected")
        print(f"   Original: {len(dummy_loader)} batches")
        print(f"   Per-core: {batch_count} batches")
        print(f"   Cores utilized: ~{len(dummy_loader) / batch_count:.1f}")
    
    print("\n" + "="*80)
    print("Recommendation")
    print("="*80)
    
    try:
        if xr.world_size() == 1:
            print("⚠️  World size is 1 - you're using single-process mode")
            print("   This uses all 8 cores via ParallelLoader but may be slower")
            print("   Consider using xmp.spawn for true multi-process training")
        else:
            print(f"✅ World size is {xr.world_size()} - multi-process mode active")
    except:
        print("Unable to determine training mode")
    
except ImportError:
    print("torch_xla not installed")
