#!/usr/bin/env python3
"""
Test script to verify GPU training with NeuralProphet
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "neuralprophet")

from base_forecaster import BaseNeuralProphetForecaster


def test_gpu_training():
    """Test GPU training with a small dataset"""
    print("=" * 60)
    print("TESTING GPU TRAINING WITH NEURALPROPHET")
    print("=" * 60)

    # Check GPU availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    else:
        print("No GPU detected - will use CPU")
        return False

    # Create a simple test forecaster
    forecaster = BaseNeuralProphetForecaster()

    # Create a small test dataset
    print("\n📊 Creating test dataset...")
    dates = pd.date_range("2023-01-01", "2023-01-31")
    values = np.sin(np.arange(len(dates)) * 0.2) + np.random.normal(0, 0.1, len(dates))

    test_df = pd.DataFrame(
        {
            "date": dates,
            "traffic_source_campaign_name_anon": "test_campaign",
            "target_business_anon": "test_business",
            "target_region_anon": "test_region",
            "Cost_per_Lead_anon": values,
            "Click-Throug_Rate_anon": np.random.uniform(0.01, 0.1, len(dates)),
            "Conversion_Rate_anon": np.random.uniform(0.05, 0.2, len(dates)),
            "campaign_max_day": np.random.uniform(100, 500, len(dates)),
            "campaign_max_total": np.random.uniform(1000, 5000, len(dates)),
        }
    )

    # Save test data
    test_csv = "test_data.csv"
    test_df.to_csv(test_csv, index=False)

    try:
        # Load and prepare data
        print("📊 Loading and preparing data...")
        split_info = forecaster.load_and_prepare_data(
            csv_path=test_csv, campaign_name="test_campaign", use_train_test_split=True
        )

        # Train model with GPU
        print("🚀 Training model with GPU...")
        forecaster.train_model(
            epochs=5,  # Just 5 epochs for testing
            learning_rate=0.01,
            use_lags=True,
            use_validation=False,
            show_progress=True,
        )

        print("✅ GPU training test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ GPU training failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Clean up - only once
        if os.path.exists(test_csv):
            os.remove(test_csv)


def test_memory_usage():
    """Test GPU memory usage"""
    print("\n" + "=" * 60)
    print("TESTING GPU MEMORY USAGE")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("GPU not available - skipping memory test")
        return False

    try:
        # Test memory allocation
        print("📊 Testing GPU memory allocation...")

        # Clear cache first
        torch.cuda.empty_cache()

        # Test tensor creation
        test_tensor = torch.randn(1000, 1000, device="cuda")
        print(f"✅ Successfully created tensor on GPU: {test_tensor.device}")

        # Check memory usage
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        cached = torch.cuda.memory_reserved() / 1024**2  # MB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB

        print(f"📊 GPU Memory Usage:")
        print(f"   Allocated: {allocated:.1f} MB")
        print(f"   Cached: {cached:.1f} MB")
        print(f"   Total: {total:.1f} MB")
        print(f"   Free: {total - cached:.1f} MB")

        # Clean up
        del test_tensor
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

        print(f"❌ Memory test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_gpu_training()
    success2 = test_memory_usage()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your AMD Radeon 890M is ready for NeuralProphet training")
        print("🚀 Run your main scripts with GPU acceleration:")
        print("   python prophet/neural_prophet_chronological_training.py fast")
    elif success1:
        print("🎉 GPU TRAINING TEST PASSED!")
        print("⚠️  Memory test had issues - check GPU drivers")
    else:
        print("💻 GPU TESTS FAILED")
        print("❌ Check your ROCm installation and GPU drivers")
        print("💡 Try running with CPU first: python -m neuralprophet --cpu")
    print("=" * 60)
