#!/usr/bin/env python3
"""
Simple NeuralProphet CPL Forecasting
Basic NeuralProphet without external regressors for initial comparison
"""

from basic_forecaster import main

if __name__ == "__main__":
    import sys

    # Check command line arguments for mode
    mode = "fast"  # default
    if len(sys.argv) > 1:
        if sys.argv[1] in ["fast", "production"]:
            mode = sys.argv[1]
        else:
            print(
                "Usage: python neural_prophet_chronological_training.py [fast|production]"
            )
            print("  fast       - 50 epochs, quick exploration (default)")
            print("  production - 200 epochs max, early stopping enabled")
            sys.exit(1)

    main(mode=mode)
