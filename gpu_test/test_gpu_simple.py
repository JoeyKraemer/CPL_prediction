#!/usr/bin/env python3
"""
Simple GPU test for NeuralProphet
"""

import os
import sys

import torch

# Set environment variables for ROCm
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["AMD_SERIALIZE_KERNEL"] = "0"


def test_basic_gpu():
    """Test basic GPU functionality"""
    print("=" * 60)
    print("BASIC GPU TEST")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

        # Test simple tensor operations
        try:
            # Clear cache
            torch.cuda.empty_cache()

            # Create tensor on GPU
            x = torch.randn(100, 100, device="cuda")
            y = torch.randn(100, 100, device="cuda")
            z = x + y

            print(f"✅ Basic GPU operations work!")
            print(f"   Tensor device: {z.device}")
            print(f"   Tensor shape: {z.shape}")

            # Check memory
            allocated = torch.cuda.memory_allocated() / 1024**2
            cached = torch.cuda.memory_reserved() / 1024**2
            print(f"   Memory allocated: {allocated:.1f} MB")
            print(f"   Memory cached: {cached:.1f} MB")

            return True

        except Exception as e:
            print(f"❌ GPU operations failed: {e}")
            return False
    else:
        print("No GPU detected")
        return False


def test_neuralprophet_import():
    """Test if NeuralProphet can be imported"""
    print("\n" + "=" * 60)
    print("NEURALPROPHET IMPORT TEST")
    print("=" * 60)

    try:
        from neuralprophet import NeuralProphet

        print("✅ NeuralProphet imported successfully")

        # Test basic model creation
        model = NeuralProphet()
        print("✅ NeuralProphet model created successfully")

        return True

    except Exception as e:
        print(f"❌ NeuralProphet import failed: {e}")
        return False


if __name__ == "__main__":
    success1 = test_basic_gpu()
    success2 = test_neuralprophet_import()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL BASIC TESTS PASSED!")
        print("✅ Your system is ready for GPU-accelerated NeuralProphet")
        print("🚀 Try running your main scripts:")
        print("   python prophet/neural_prophet_chronological_training.py fast")
    elif success1:
        print("🎉 GPU TEST PASSED!")
        print("⚠️  NeuralProphet import failed - check installation")
    else:
        print("💻 BASIC TESTS FAILED")
        print("❌ Check your ROCm and PyTorch installation")
    print("=" * 60)
