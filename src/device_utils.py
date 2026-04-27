"""
Device detection and management utilities for cross-platform support.
Handles CUDA (NVIDIA), MPS (Apple Silicon/Intel), and CPU.
"""
import torch
import platform
import subprocess


def get_device_info():
    """Get detailed information about available compute devices."""
    info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        'cpu_count': torch.get_num_threads(),
    }
    
    # CUDA info
    if info['cuda_available']:
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_capability'] = torch.cuda.get_device_capability(0)
        
        # Memory info
        if torch.cuda.is_available():
            info['cuda_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
    
    # MPS info (Apple Silicon)
    if info['mps_available']:
        info['mps_built'] = torch.backends.mps.is_built()
        # Try to get more Mac-specific info
        try:
            if platform.system() == 'Darwin':
                # Check if Apple Silicon
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                info['cpu_brand'] = result.stdout.strip()
                
                # Check for Apple Silicon
                if 'Apple' in info['cpu_brand'] or info['architecture'] == 'arm64':
                    info['apple_silicon'] = True
                else:
                    info['apple_silicon'] = False
        except:
            pass
    
    return info


def select_device(preferred_device='auto', verbose=True):
    """
    Select the best available device for training/inference.
    
    Args:
        preferred_device: 'auto', 'cuda', 'mps', or 'cpu'
        verbose: Print device information
        
    Returns:
        torch.device: Selected device
    """
    device_info = get_device_info()
    
    if preferred_device == 'auto':
        # Priority: CUDA > MPS > CPU
        if device_info['cuda_available']:
            device = torch.device('cuda')
            device_type = 'CUDA'
        elif device_info['mps_available']:
            device = torch.device('mps')
            device_type = 'MPS'
        else:
            device = torch.device('cpu')
            device_type = 'CPU'
    else:
        # Use specified device
        if preferred_device == 'cuda':
            if not device_info['cuda_available']:
                print("⚠️  CUDA requested but not available. Falling back to auto-selection.")
                return select_device('auto', verbose)
            device = torch.device('cuda')
            device_type = 'CUDA'
        elif preferred_device == 'mps':
            if not device_info['mps_available']:
                print("⚠️  MPS requested but not available. Falling back to auto-selection.")
                return select_device('auto', verbose)
            device = torch.device('mps')
            device_type = 'MPS'
        elif preferred_device == 'cpu':
            device = torch.device('cpu')
            device_type = 'CPU'
        else:
            raise ValueError(f"Unknown device: {preferred_device}. Use 'auto', 'cuda', 'mps', or 'cpu'")
    
    if verbose:
        print("\n" + "="*80)
        print("🖥️  Device Information")
        print("="*80)
        print(f"Platform: {device_info['platform']} ({device_info['architecture']})")
        print(f"Processor: {device_info.get('processor', 'Unknown')}")
        print(f"Python: {device_info['python_version']}")
        print(f"PyTorch: {device_info['pytorch_version']}")
        print(f"CPU Threads: {device_info['cpu_count']}")
        print()
        
        if device_type == 'CUDA':
            print(f"✅ Using CUDA GPU")
            print(f"   Device: {device_info['cuda_device_name']}")
            print(f"   CUDA Version: {device_info['cuda_version']}")
            print(f"   Compute Capability: {device_info['cuda_capability']}")
            if 'cuda_memory_total' in device_info:
                print(f"   Memory: {device_info['cuda_memory_total']:.2f} GB")
        elif device_type == 'MPS':
            print(f"✅ Using Apple Metal Performance Shaders (MPS)")
            if device_info.get('apple_silicon'):
                print(f"   Apple Silicon Detected: {device_info.get('cpu_brand', 'Unknown')}")
            else:
                print(f"   Intel Mac with MPS support")
            print(f"   Note: MPS provides GPU acceleration on Mac")
        else:
            print(f"⚠️  Using CPU (slower training)")
            print(f"   Consider using a GPU for faster training")
        
        print("="*80 + "\n")
    
    return device


def get_optimal_workers(device):
    """
    Get optimal number of data loader workers based on device.
    
    Args:
        device: torch.device
        
    Returns:
        int: Number of workers
    """
    if device.type == 'cuda':
        # CUDA can handle more workers
        return min(8, torch.get_num_threads())
    elif device.type == 'mps':
        # MPS works better with fewer workers
        return min(4, torch.get_num_threads() // 2)
    else:
        # CPU
        return min(4, torch.get_num_threads() // 2)


def check_mixed_precision_support(device):
    """
    Check if mixed precision training is supported on the device.
    
    Args:
        device: torch.device
        
    Returns:
        bool: True if mixed precision is supported
    """
    if device.type == 'cuda':
        # Check CUDA capability (need 7.0+ for good FP16 support)
        capability = torch.cuda.get_device_capability(device)
        return capability[0] >= 7
    elif device.type == 'mps':
        # MPS supports mixed precision but may have issues
        # Return False for now to avoid potential bugs
        return False
    else:
        # CPU doesn't benefit from mixed precision
        return False


def optimize_for_device(model, device, compile_model=False):
    """
    Apply device-specific optimizations to the model.
    
    Args:
        model: PyTorch model
        device: torch.device
        compile_model: Whether to use torch.compile (PyTorch 2.0+)
        
    Returns:
        Optimized model
    """
    model = model.to(device)
    
    # Device-specific optimizations
    if device.type == 'cuda':
        # Enable cuDNN benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Enable TF32 for Ampere GPUs (3x speedup)
        if torch.cuda.get_device_capability(device)[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✅ Enabled TF32 for faster training on Ampere GPU")
    
    elif device.type == 'mps':
        # MPS-specific settings
        # Currently no special optimizations needed
        pass
    
    # Compile model (PyTorch 2.0+)
    if compile_model and hasattr(torch, 'compile'):
        try:
            print("🔧 Compiling model with torch.compile()...")
            model = torch.compile(model)
            print("✅ Model compiled successfully")
        except Exception as e:
            print(f"⚠️  Failed to compile model: {e}")
            print("   Continuing without compilation")
    
    return model


def print_device_recommendations(device):
    """Print recommendations for optimal training on the selected device."""
    print("\n" + "="*80)
    print("💡 Training Recommendations")
    print("="*80)
    
    if device.type == 'cuda':
        print("✅ CUDA GPU detected - optimal for training")
        print("   • Use mixed precision (AMP) for faster training")
        print("   • Increase batch size to utilize GPU memory")
        print("   • Enable gradient checkpointing for larger models")
        print("   • Use torch.compile() for additional speedup (PyTorch 2.0+)")
    
    elif device.type == 'mps':
        print("✅ MPS (Apple Silicon/Intel) detected")
        print("   • MPS provides good acceleration on Mac")
        print("   • Start with smaller batch sizes (MPS has different memory management)")
        print("   • Disable mixed precision if you encounter issues")
        print("   • Use fewer data loader workers (2-4)")
        print("   • Some operations may fall back to CPU (this is normal)")
        
        if platform.machine() == 'arm64':
            print("   • Apple Silicon detected - expect good performance!")
        else:
            print("   • Intel Mac - performance may vary")
    
    else:
        print("⚠️  CPU training - will be slower")
        print("   • Use smaller models and batch sizes")
        print("   • Consider using a cloud GPU (Colab, AWS, etc.)")
        print("   • Reduce context length to speed up training")
        print("   • Use gradient accumulation instead of large batches")
    
    print("="*80 + "\n")


def test_device_performance(device, size=1024):
    """
    Run a simple performance test on the device.
    
    Args:
        device: torch.device
        size: Matrix size for test
        
    Returns:
        float: Time in seconds
    """
    import time
    
    print(f"\n🧪 Running performance test on {device}...")
    
    # Warmup
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    _ = torch.matmul(a, b)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    
    # Actual test
    start = time.time()
    for _ in range(10):
        c = torch.matmul(a, b)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    
    elapsed = time.time() - start
    
    print(f"   Matrix multiplication ({size}x{size}): {elapsed:.4f}s for 10 iterations")
    print(f"   Performance: {(10 * size**3 * 2) / elapsed / 1e9:.2f} GFLOPS")
    
    return elapsed


if __name__ == '__main__':
    # Test device detection
    print("Testing device detection and selection...\n")
    
    # Get device info
    info = get_device_info()
    print("Device Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Select device
    device = select_device('auto', verbose=True)
    
    # Print recommendations
    print_device_recommendations(device)
    
    # Test performance
    test_device_performance(device, size=512)
