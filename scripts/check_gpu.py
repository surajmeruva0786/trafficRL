#!/usr/bin/env python3
"""
GPU/CUDA Verification Script
Checks if PyTorch can access GPU and provides detailed information.
"""

import torch
import sys


def check_cuda_availability():
    """Check CUDA availability and print detailed information."""
    print("=" * 70)
    print("GPU/CUDA Verification for TrafficRL")
    print("=" * 70)
    print()
    
    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print("✓ GPU acceleration is ENABLED")
        print()
        
        # CUDA version
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print()
        
        # GPU information
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        print()
        
        for i in range(num_gpus):
            print(f"GPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Multi-processor count: {props.multi_processor_count}")
            print()
        
        # Current device
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {current_device} ({torch.cuda.get_device_name(current_device)})")
        print()
        
        # Memory usage
        print("Memory usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print()
        
        # Test GPU computation
        print("Testing GPU computation...")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("✓ GPU computation test PASSED")
            print(f"  Result shape: {z.shape}")
            print(f"  Result device: {z.device}")
        except Exception as e:
            print(f"✗ GPU computation test FAILED: {e}")
        print()
        
        print("=" * 70)
        print("✓ Your system is ready for GPU-accelerated training!")
        print("=" * 70)
        return True
        
    else:
        print("✗ GPU acceleration is DISABLED")
        print()
        print("Possible reasons:")
        print("  1. CUDA is not installed on your system")
        print("  2. PyTorch was installed without CUDA support")
        print("  3. GPU drivers are not properly installed")
        print()
        print("To enable GPU acceleration:")
        print("  1. Install NVIDIA CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)")
        print("  2. Install CUDA-enabled PyTorch:")
        print("     pip uninstall torch")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print()
        print("=" * 70)
        print("⚠️  Training will be VERY slow on CPU!")
        print("=" * 70)
        return False


def main():
    """Main function."""
    cuda_ok = check_cuda_availability()
    
    if not cuda_ok:
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(1)


if __name__ == "__main__":
    main()
