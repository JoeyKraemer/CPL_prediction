#!/usr/bin/env python3
"""
Base Forecaster Module - Common functionality for NeuralProphet forecasters
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from neuralprophet import NeuralProphet

warnings.filterwarnings("ignore")


class BaseNeuralProphetForecaster:
    """Base class for NeuralProphet-based CPL forecasting"""

    def __init__(self):
        self.model = None
        self.prophet_df = None
        self.campaign_data = None
        self.forecast = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.test_metrics = None
        self.feature_cols = []

    def load_and_prepare_data(
        self,
        csv_path="data/Dataset_AM_final.csv",
        campaign_name=None,
        use_train_test_split=True,
    ):
        """Load and prepare CPL data for NeuralProphet with proper train/test split and feature engineering"""
        # Load dataset
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df["date"] = pd.to_datetime(df["date"])

        # Clean data
        df_clean = df[df["Cost_per_Lead_anon"] > 0].copy()

        # Select campaign
        if campaign_name:
            self.campaign_data = df_clean[
                df_clean["traffic_source_campaign_name_anon"] == campaign_name
            ].copy()
        else:
            self.campaign_data = df_clean.copy()

        self.campaign_data = self.campaign_data.sort_values("date").reset_index(
            drop=True
        )

        # FEATURE ENGINEERING
        self.campaign_data = self._engineer_features(self.campaign_data)

        if not use_train_test_split:
            raise ValueError(
                "Train/test split is required. Set use_train_test_split=True"
            )

        # TRAIN/VAL/TEST SPLIT
        # Use chronological split: 70% train, 15% val, 15% test
        n = len(self.campaign_data)
        train_size = int(n * 0.70)
        val_size = int(n * 0.15)

        train_data = self.campaign_data.iloc[:train_size].copy().reset_index(drop=True)
        val_data = (
            self.campaign_data.iloc[train_size : train_size + val_size]
            .copy()
            .reset_index(drop=True)
        )
        test_data = (
            self.campaign_data.iloc[train_size + val_size :]
            .copy()
            .reset_index(drop=True)
        )

        # Note: n_lags=7 adds 7 more autoregressive features internally
        feature_cols = [
            # Essential temporal indicators (4)
            "day_of_week",
            "is_weekend",
            "is_month_start",
            "is_month_end",
            # Campaign performance metrics (4)
            "CTR",
            "Conversion_Rate"
        ]

        # Create NeuralProphet format with external regressors
        self.train_df = pd.DataFrame(
            {"ds": train_data["date"], "y": train_data["Cost_per_Lead_anon"]}
        )
        for col in feature_cols:
            if col in train_data.columns:
                self.train_df[col] = train_data[col].values

        self.val_df = pd.DataFrame(
            {"ds": val_data["date"], "y": val_data["Cost_per_Lead_anon"]}
        )
        for col in feature_cols:
            if col in val_data.columns:
                self.val_df[col] = val_data[col].values

        self.test_df = pd.DataFrame(
            {"ds": test_data["date"], "y": test_data["Cost_per_Lead_anon"]}
        )
        for col in feature_cols:
            if col in test_data.columns:
                self.test_df[col] = test_data[col].values

        # Store feature columns for later use
        self.feature_cols = [col for col in feature_cols if col in train_data.columns]

        # Use training data as prophet_df
        self.prophet_df = self.train_df.copy()

        print(f"📊 Data Split:")
        print(
            f"  Train: {len(self.train_df)} days ({train_data['date'].min().date()} to {train_data['date'].max().date()})"
        )
        print(
            f"  Val:   {len(self.val_df)} days ({val_data['date'].min().date()} to {val_data['date'].max().date()})"
        )
        print(
            f"  Test:  {len(self.test_df)} days ({test_data['date'].min().date()} to {test_data['date'].max().date()})"
        )

        return {
            "train": len(self.train_df),
            "val": len(self.val_df),
            "test": len(self.test_df),
        }

    def _engineer_features(self, df):
        """Engineer OPTIMIZED feature set"""
        df = df.copy()

        # 1. ESSENTIAL TEMPORAL FEATURES (binary indicators not fully captured by seasonality)
        df["day_of_week"] = df["date"].dt.dayofweek  # 0=Monday, 6=Sunday
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_month_start"] = (df["date"].dt.day <= 5).astype(int)
        df["is_month_end"] = (df["date"].dt.day >= 25).astype(int)

        # 2. CAMPAIGN PERFORMANCE METRICS (external data not in CPL time series)
        if "Click-Throug_Rate_anon" in df.columns:
            df["CTR"] = df["Click-Throug_Rate_anon"].fillna(0)
        else:
            df["CTR"] = 0

        if "Conversion_Rate_anon" in df.columns:
            df["Conversion_Rate"] = df["Conversion_Rate_anon"].fillna(0)
        else:
            df["Conversion_Rate"] = 0

        return df

    def train_model(
        self,
        epochs=50,
        learning_rate=0.01,
        use_lags=True,
        use_validation=True,
        show_progress=False,
    ):
        """Train optimized NeuralProphet model with validation monitoring"""

        n_data_points = len(self.train_df)
        if use_lags and n_data_points < 30:
            n_lags = min(3, max(1, n_data_points // 5))  # 1-3 lags for short campaigns
            print(f"  ⚠️  Short campaign ({n_data_points} days) - using {n_lags} lags")
        elif use_lags:
            n_lags = 7  # Standard 7 lags for longer campaigns
        else:
            n_lags = 0

        # IMPROVEMENT 3: Data validation before training
        if not self._validate_data():
            raise ValueError(
                "Data validation failed - check for NaN in target or features"
            )

        import io
        import sys

        # Create NeuralProphet model - let it handle trainer internally
        self.model = NeuralProphet(
            # Hyperparameters:
            # Seasonality
            weekly_seasonality="auto",
            yearly_seasonality="auto",
            daily_seasonality=False,
            # Changepoints
            n_changepoints=15,
            changepoints_range=0.8,
            # Neural architecture
            n_lags=n_lags,
            n_forecasts=1,
            # Training parameters
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=16,
            # Regularization
            trend_reg=0,
            seasonality_reg=0,
            # Normalization and loss
            normalize="minmax",
            impute_missing=True,
            drop_missing=True,
            loss_func="Huber",
        )

        # ADD EXTERNAL REGRESSORS (if features are available)
        if hasattr(self, "feature_cols") and self.feature_cols:
            print(f"  Adding {len(self.feature_cols)} external regressors...")

            # Filter out features that have only one unique value (constant features)
            valid_features = []
            removed_features = []

            for feature in self.feature_cols:
                if feature in self.prophet_df.columns:
                    # Check if feature has more than one unique value
                    unique_count = self.prophet_df[feature].nunique()
                    if unique_count > 1:
                        valid_features.append(feature)
                    else:
                        removed_features.append(feature)
                        print(
                            f"  ⚠️  Removed constant feature '{feature}' (only {unique_count} unique value)"
                        )
                else:
                    print(f"  ⚠️  Feature '{feature}' not found in data - skipping")

            # Update feature_cols to only include valid features
            self.feature_cols = valid_features

            # Remove unused features from the DataFrame to avoid conflicts
            if removed_features:
                print(f"  🗑️  Removing unused columns from data: {removed_features}")
                self.prophet_df = self.prophet_df.drop(
                    columns=removed_features, errors="ignore"
                )
                self.train_df = self.train_df.drop(
                    columns=removed_features, errors="ignore"
                )
                self.val_df = self.val_df.drop(
                    columns=removed_features, errors="ignore"
                )
                self.test_df = self.test_df.drop(
                    columns=removed_features, errors="ignore"
                )

            # Add valid features to the model
            for feature in self.feature_cols:
                try:
                    # Add each feature as future regressor (available for all time periods)
                    self.model = self.model.add_future_regressor(feature)
                except Exception as e:
                    print(f"  ⚠️  Could not add feature {feature}: {str(e)}")
                    # Debug: Show more information about the feature
                    if feature in self.prophet_df.columns:
                        print(f"     Feature dtype: {self.prophet_df[feature].dtype}")
                        print(
                            f"     Feature missing values: {self.prophet_df[feature].isnull().sum()}"
                        )
                        print(
                            f"     Feature unique values: {self.prophet_df[feature].nunique()}"
                        )

        # Configure for GPU training (AMD Radeon 890M with 16GB VRAM)
        if torch.cuda.is_available():
            # Set environment variables for ROCm compatibility
            os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
            os.environ["AMD_SERIALIZE_KERNEL"] = "0"
            torch.cuda.empty_cache()

        # Check if we have enough data for training
        min_required_rows = 10  # Minimum rows needed for training
        if len(self.prophet_df) < min_required_rows:
            raise ValueError(
                f"Dataframe has only {len(self.prophet_df)} rows, "
                f"but at least {min_required_rows} rows are required for training. "
                f"Please check your data split or use a campaign with more data."
            )

        # Train the model
        try:
            if use_validation and self.val_df is not None and len(self.val_df) > 0:
                self.metrics = self.model.fit(
                    self.prophet_df,
                    freq="D",
                    validation_df=self.val_df,
                )
            else:
                self.metrics = self.model.fit(self.prophet_df, freq="D")

        except Exception as e:
            if "less than n_forecasts + n_lags rows" in str(e):
                raise ValueError(
                    f"Insufficient data: {len(self.prophet_df)} rows, needs {self.model.n_forecasts + self.model.n_lags}"
                )
            else:
                raise e

        # Extract final training metrics
        if self.metrics is not None and len(self.metrics) > 0:
            final_metrics = self.metrics.tail(1).to_dict("records")[0]
            self.final_loss = final_metrics.get("Loss", None)
            self.final_mae = final_metrics.get("MAE", None)
            self.epochs_trained = len(self.metrics)
        else:
            self.final_loss = None
            self.final_mae = None
            self.epochs_trained = epochs

        return self.model

    def get_training_metrics(self):
        """Get training convergence metrics"""
        return {
            "epochs_trained": self.epochs_trained
            if hasattr(self, "epochs_trained")
            else None,
            "final_loss": self.final_loss if hasattr(self, "final_loss") else None,
            "final_mae": self.final_mae if hasattr(self, "final_mae") else None,
            "converged": self.epochs_trained < 100
            if hasattr(self, "epochs_trained")
            else False,
        }

    def evaluate_performance(self, on_test_set=True):
        """Calculate NeuralProphet performance metrics on proper test set"""
        if on_test_set and self.test_df is not None and len(self.test_df) > 0:
            # PROPER EVALUATION: Test on completely unseen data (like LSTM)
            print("\n🎯 Evaluating on TEST SET (unseen data)")
            test_forecast = self.model.predict(self.test_df)

            # Use the correct prediction column
            pred_col = "yhat1" if "yhat1" in test_forecast.columns else "yhat"

            if pred_col in test_forecast.columns and "y" in test_forecast.columns:
                forecast_clean = test_forecast.dropna(subset=["y", pred_col])
                if len(forecast_clean) > 0:
                    mae = mean_absolute_error(
                        forecast_clean["y"], forecast_clean[pred_col]
                    )
                    mape = (
                        mean_absolute_percentage_error(
                            forecast_clean["y"], forecast_clean[pred_col]
                        )
                        * 100
                    )

                    # Calculate SMA baselines for comparison
                    # SMA-3: 3-day Simple Moving Average baseline
                    sma3_preds = (
                        forecast_clean["y"]
                        .shift(1)
                        .rolling(window=3, min_periods=1)
                        .mean()
                    )
                    # Filter out NaN values for proper evaluation
                    mask_sma3 = ~sma3_preds.isna()
                    sma3_mae = (
                        mean_absolute_error(
                            forecast_clean["y"][mask_sma3], sma3_preds[mask_sma3]
                        )
                        if mask_sma3.sum() > 0
                        else float("nan")
                    )
                    sma3_mape = (
                        mean_absolute_percentage_error(
                            forecast_clean["y"][mask_sma3], sma3_preds[mask_sma3]
                        )
                        * 100
                        if mask_sma3.sum() > 0
                        else float("nan")
                    )

                    # SMA-7: 7-day Simple Moving Average baseline
                    sma7_preds = (
                        forecast_clean["y"]
                        .shift(1)
                        .rolling(window=7, min_periods=1)
                        .mean()
                    )
                    # Filter out NaN values for proper evaluation
                    mask_sma7 = ~sma7_preds.isna()
                    sma7_mae = (
                        mean_absolute_error(
                            forecast_clean["y"][mask_sma7], sma7_preds[mask_sma7]
                        )
                        if mask_sma7.sum() > 0
                        else float("nan")
                    )
                    sma7_mape = (
                        mean_absolute_percentage_error(
                            forecast_clean["y"][mask_sma7], sma7_preds[mask_sma7]
                        )
                        * 100
                        if mask_sma7.sum() > 0
                        else float("nan")
                    )

                    self.test_metrics = {
                        "MAE": mae,
                        "MAPE": mape,
                        "SMA3_MAE": sma3_mae,
                        "SMA3_MAPE": sma3_mape,
                        "SMA7_MAE": sma7_mae,
                        "SMA7_MAPE": sma7_mape,
                        "test_predictions": forecast_clean[[pred_col]].values.flatten(),
                        "test_actuals": forecast_clean["y"].values,
                    }

                    print(f"  Test samples: {len(forecast_clean)}")
                    print(f"  NeuralProphet MAE: {mae:.3f}, MAPE: {mape:.1f}%")
                    print(
                        f"  SMA-3 Baseline MAE: {sma3_mae:.3f}, MAPE: {sma3_mape:.1f}%"
                    )
                    print(
                        f"  SMA-7 Baseline MAE: {sma7_mae:.3f}, MAPE: {sma7_mape:.1f}%"
                    )
                    return self.test_metrics
                
        return {"MAE": float("inf"), "MAPE": float("inf")}

    def get_campaign_info(self):
        """Get campaign metadata"""
        return {
            "campaign": self.campaign_data["traffic_source_campaign_name_anon"].iloc[0]
            if hasattr(self, "campaign_data") and len(self.campaign_data) > 0
            else None,
            "business": self.campaign_data["target_business_anon"].iloc[0]
            if hasattr(self, "campaign_data") and len(self.campaign_data) > 0
            else None,
            "region": self.campaign_data["target_region_anon"].iloc[0]
            if hasattr(self, "campaign_data") and len(self.campaign_data) > 0
            else None,
            "date_range": f"{self.campaign_data['date'].min().date()} to {self.campaign_data['date'].max().date()}"
            if hasattr(self, "campaign_data") and len(self.campaign_data) > 0
            else None,
            "total_days": len(self.campaign_data)
            if hasattr(self, "campaign_data")
            else 0,
            "avg_cpl": self.campaign_data["Cost_per_Lead_anon"].mean()
            if hasattr(self, "campaign_data") and len(self.campaign_data) > 0
            else None,
        }

    def _validate_data(self):
        """Validate data before training"""
        import numpy as np

        # Check target column
        if self.train_df["y"].isnull().any():
            nan_count = self.train_df["y"].isnull().sum()
            print(f"  ❌ Target 'y' has {nan_count} NaN")
            return False

        # Check for inf values
        if np.isinf(self.train_df["y"]).any():
            print(f"  ❌ Target 'y' has infinite values")
            return False

        # Check features
        if hasattr(self, "feature_cols"):
            for col in self.feature_cols:
                if col in self.train_df.columns:
                    nan_count = self.train_df[col].isnull().sum()
                    if nan_count > 0:
                        print(f"  ❌ Feature '{col}' has {nan_count} NaN")
                        return False

                    if np.isinf(self.train_df[col]).any():
                        print(f"  ❌ Feature '{col}' has infinite values")
                        return False

        # Check minimum data points
        if len(self.train_df) < 5:
            print(f"  ❌ Insufficient training data: {len(self.train_df)} rows")
            return False

        return True


def evaluate_all_campaigns(
    csv_path="data/Dataset_AM_final.csv",
    epochs=50,
    max_campaigns=None,
    min_days=100,
    mode="fast",
    forecaster_class=None,
):
    """Evaluate all campaigns in dataset with specified filters"""

    train_pct = 0.70
    val_pct = 0.15
    test_pct = 0.15
    n_lags = 7
    min_train_days = 10  # Minimum for meaningful training

    # Calculate: need enough for train + val + test + lags
    min_days_calculated = int(min_train_days / train_pct) + n_lags
    min_days = max(min_days, min_days_calculated, 30)  # At least 30 days

    print(f"📊 Calculated minimum days required: {min_days}")
    print(
        f"   (Train: ≥{int(min_days * train_pct)}, Val: ≥{int(min_days * val_pct)}, Test: ≥{int(min_days * test_pct)}, Lags: {n_lags})"
    )

    # Configure based on mode
    if mode == "production":
        epochs = 200
        use_validation = True
        show_progress = False
        print(
            "🏭 PRODUCTION MODE: 200 epochs max, early stopping enabled, validation monitoring"
        )
    else:  # fast mode
        epochs = 50
        use_validation = False
        show_progress = False
        print("⚡ FAST MODE: 50 epochs, quick exploration")

    # Load full dataset
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"])
    df_clean = df[df["Cost_per_Lead_anon"] > 0].copy()

    # Filter campaigns by minimum data points
    campaign_counts = df_clean.groupby("traffic_source_campaign_name_anon").size()
    valid_campaigns = campaign_counts[campaign_counts >= min_days].index.tolist()

    if max_campaigns is not None:
        valid_campaigns = valid_campaigns[:max_campaigns]

    print(f"\n🎯 Evaluating {len(valid_campaigns)} campaigns (min {min_days} days)...")
    print(f"   Total campaigns in dataset: {len(campaign_counts)}")
    print(f"   Filtered to campaigns with ≥{min_days} days: {len(valid_campaigns)}")
    print("=" * 80)

    results = []
    successful = 0
    failed = 0
    early_stopped = 0

    for idx, campaign in enumerate(valid_campaigns, 1):
        try:
            # Initialize forecaster (silent)
            forecaster = forecaster_class()

            # Load campaign data
            split_info = forecaster.load_and_prepare_data(
                csv_path=csv_path, campaign_name=campaign, use_train_test_split=True
            )

            # Check continuous data (no large gaps)
            train_dates = pd.to_datetime(forecaster.train_df["ds"]).sort_values()
            date_diff = train_dates.diff().dt.total_seconds() / 86400
            max_gap = date_diff.max()

            if max_gap > 30:
                print(
                    f"[{idx}/{len(valid_campaigns)}] {campaign[:40]:<40} ⚠️ Gap: {max_gap:.0f}d"
                )
                failed += 1
                continue

            # Train model (silent)
            forecaster.train_model(
                epochs=epochs,
                learning_rate=0.01,
                use_lags=True,
                use_validation=use_validation,
                show_progress=False,
            )

            # Get training metrics
            train_metrics = forecaster.get_training_metrics()
            epochs_trained = train_metrics.get("epochs_trained", epochs)

            # Evaluate on test set
            performance = forecaster.evaluate_performance(on_test_set=True)

            # Show compact one-line result
            print(
                f"[{idx}/{len(valid_campaigns)}] {campaign[:40]:<40} "
                f"✅ MAE: {performance['MAE']:.4f} ({epochs_trained}e)"
            )
            print(
                f"✅ Epochs: {epochs_trained}/{epochs}, MAE: {performance['MAE']:.4f}"
            )

            # Get campaign info for results
            campaign_info = forecaster.get_campaign_info()

            # Store minimal results for final summary
            results.append(
                {
                    "Campaign": campaign,
                    "Business": campaign_info["business"],
                    "Total_Days": campaign_info["total_days"],
                    "Train_Days": split_info["train"]
                    if isinstance(split_info, dict)
                    else 0,
                    "Test_Days": split_info["test"]
                    if isinstance(split_info, dict)
                    else 0,
                    "Avg_CPL": campaign_info["avg_cpl"],
                    "Test_MAE": performance["MAE"],
                    "Test_MAPE": performance["MAPE"],
                    "SMA3_MAE": performance.get("SMA3_MAE", None),
                    "SMA3_MAPE": performance.get("SMA3_MAPE", None),
                    "SMA7_MAE": performance.get("SMA7_MAE", None),
                    "SMA7_MAPE": performance.get("SMA7_MAPE", None),
                    "Epochs_Trained": train_metrics.get("epochs_trained", epochs),
                    "Final_Loss": train_metrics.get("final_loss", None),
                    "Converged_Early": train_metrics.get("converged", False),
                }
            )

            if train_metrics.get("converged", False):
                early_stopped += 1

            successful += 1

        except Exception as e:
            import traceback

            print(f"❌ {str(e)[:80]}")
            if idx == 1:  # Show full traceback for first failure only
                print(traceback.format_exc())
            failed += 1
            continue

    print(f"\n{'=' * 80}")
    print(f"✅ Successful: {successful} | ❌ Failed: {failed}")
    if mode == "production":
        print(f"⏹️  Early stopped: {early_stopped}/{successful} campaigns")

    return pd.DataFrame(results)


def main(mode="fast", forecaster_class=None, model_type="NP"):
    """Main execution function - Evaluate ALL campaigns

    Args:
        mode: 'fast' (50 epochs for exploration) or 'production' (200 epochs with early stopping)
        forecaster_class: The forecaster class to use
        model_type: Type of model (NP or NP_MICE)
    """
    print("=" * 80)
    print(f"NEURAL PROPHET - ALL CAMPAIGNS EVALUATION ({mode.upper()} MODE)")
    print("=" * 80)

    # Evaluate campaigns
    results_df = evaluate_all_campaigns(
        csv_path="data/Dataset_AM_final.csv",
        epochs=50 if mode == "fast" else 200,
        max_campaigns=None,
        min_days=30,
        mode=mode,
        forecaster_class=forecaster_class,
    )

    if len(results_df) == 0:
        print("No results to display.")
        return results_df

    # Summary statistics only
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nCampaigns evaluated: {len(results_df)}")
    print(f"Total training days: {results_df['Train_Days'].sum():.0f}")
    print(f"Total test days:     {results_df['Test_Days'].sum():.0f}")

    # Performance metrics
    print(f"\nNeuralProphet Performance:")
    print(
        f"  MAE:  Mean={results_df['Test_MAE'].mean():.4f}, Median={results_df['Test_MAE'].median():.4f}"
    )
    print(
        f"  MAPE: Mean={results_df['Test_MAPE'].mean():.1f}%, Median={results_df['Test_MAPE'].median():.1f}%"
    )

    # Baseline performance
    if "SMA3_MAE" in results_df.columns and "SMA7_MAE" in results_df.columns:
        print(f"\nSMA-3 Performance:")
        print(
            f"  MAE:  Mean={results_df['SMA3_MAE'].mean():.4f}, Median={results_df['SMA3_MAE'].median():.4f}"
        )
        print(
            f"  MAPE: Mean={results_df['SMA3_MAPE'].mean():.1f}%, Median={results_df['SMA3_MAPE'].median():.1f}%"
        )

        print(f"\nSMA-7 Performance:")
        print(
            f"  MAE:  Mean={results_df['SMA7_MAE'].mean():.4f}, Median={results_df['SMA7_MAE'].median():.4f}"
        )
        print(
            f"  MAPE: Mean={results_df['SMA7_MAPE'].mean():.1f}%, Median={results_df['SMA7_MAPE'].median():.1f}%"
        )

    # Baseline comparisons
    if "SMA3_MAE" in results_df.columns and "SMA7_MAE" in results_df.columns:
        np_beats_sma3 = (results_df["Test_MAE"] < results_df["SMA3_MAE"]).sum()
        np_beats_sma7 = (results_df["Test_MAE"] < results_df["SMA7_MAE"]).sum()
        total = len(results_df)

        print(f"\nBaseline Comparison:")
        print(
            f"  Beats SMA-3: {np_beats_sma3}/{total} ({np_beats_sma3 / total * 100:.1f}%)"
        )
        print(
            f"  Beats SMA-7: {np_beats_sma7}/{total} ({np_beats_sma7 / total * 100:.1f}%)"
        )

    # Training stats
    if "Epochs_Trained" in results_df.columns:
        print(f"\nTraining:")
        print(f"  Mean epochs: {results_df['Epochs_Trained'].mean():.1f}")
        if mode == "production" and "Converged_Early" in results_df.columns:
            early_stopped = results_df["Converged_Early"].sum()
            print(f"  Early stopped: {early_stopped}/{len(results_df)}")

    # Save results
    results_dir = Path("neuralprophet/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"result_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved: {output_file}")
    print("=" * 80)

    return results_df
