#!/usr/bin/env python3
"""
Basic Forecaster Module - Simple NeuralProphet without MICE
"""

from base_forecaster import BaseNeuralProphetForecaster
from base_forecaster import main as base_main


class SimpleNeuralProphetForecaster(BaseNeuralProphetForecaster):
    """Simple NeuralProphet forecaster without MICE imputation"""

    def load_and_prepare_data(
        self,
        csv_path="data/Dataset_AM_final.csv",
        campaign_name=None,
        use_train_test_split=True,
    ):
        """Load and prepare CPL data without MICE imputation"""
        return super().load_and_prepare_data(
            csv_path, campaign_name, use_train_test_split
        )


def main(mode="fast"):
    """Main function for basic forecaster"""
    return base_main(
        mode=mode, forecaster_class=SimpleNeuralProphetForecaster, model_type="NP"
    )


if __name__ == "__main__":
    import sys

    # Check command line arguments for mode
    mode = "fast"  # default
    if len(sys.argv) > 1:
        if sys.argv[1] in ["fast", "production"]:
            mode = sys.argv[1]
        else:
            print("Usage: python basic_forecaster.py [fast|production]")
            print("  fast       - 50 epochs, quick exploration (default)")
            print("  production - 200 epochs max, early stopping enabled")
            sys.exit(1)

    main(mode=mode)
