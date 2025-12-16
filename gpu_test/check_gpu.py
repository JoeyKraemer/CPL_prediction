#!/usr/bin/env python3
"""
Check GPU availability for PyTorch with ROCm
"""

import sys

import torch

print("=" * 60)
print("GPU DETECTION FOR NEURALPROPHET")
print("=" * 60)

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch with ROCm: {'rocm' in torch.__version__.lower()}")

# Check CUDA availability (for NVIDIA)
cuda_available = torch.cuda.is_available()
print(f"CUDA available (NVIDIA): {cuda_available}")

hip_available = hasattr(torch, "hip") and torch.hip.is_available()
print(f"HIP available (AMD): {hip_available}")

# Check device count
device_count = (
    torch.cuda.device_count()
    if cuda_available
    else (torch.hip.device_count() if hip_available else 0)
)
print(f"GPU device count: {device_count}")

# Get device info
if device_count > 0:
    if cuda_available:
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    elif hip_available:
        for i in range(device_count):
            print(f"GPU {i}: {torch.hip.get_device_name(i)}")
else:
    print("No GPUs detected - running on CPU")

print(
    f"Current device: {torch.cuda.current_device() if cuda_available else torch.hip.current_device() if hip_available else 'CPU'}"
)
print(f"Current device: {torch.cuda.current_device() if cuda_available else torch.hip.current_device() if hip_available else 'CPU'}")

# Test if we can move a tensor to GPU
try:
    if cuda_available:
        test_tensor = torch.tensor([1.0]).cuda()
        print(f"Successfully created CUDA tensor: {test_tensor.device}")
    elif hip_available:
        test_tensor = torch.tensor([1.0]).hip()
        print(f"Successfully created HIP tensor: {test_tensor.device}")
    else:
        test_tensor = torch.tensor([1.0])
        print(f"Running on CPU: {test_tensor.device}")
except Exception as e:
    print(f"GPU tensor creation failed: {e}")

print("=" * 60)
print("RECOMMENDATIONS FOR NEURALPROPHET")
print("=" * 60)

if device_count > 0:
    print("✅ GPU detected! NeuralProphet should automatically use it.")
    print("📝 No code changes needed - PyTorch Lightning handles GPU selection.")
    print("💡 For explicit GPU control, you can set:")
    print("   - accelerator='gpu' in PyTorch Lightning Trainer")
    print("   - devices='auto' for automatic device selection")
else:
    print("❌ No GPU detected. Running on CPU.")
    print("🔧 Check your ROCm installation and GPU drivers.")
    print("💡 Your AMD Radeon 890M should be supported by ROCm 6.1")

print("=" * 60)
