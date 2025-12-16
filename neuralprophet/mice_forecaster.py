#!/usr/bin/env python3
"""
MICE Forecaster Module - Extends base forecaster with MICE imputation
"""

import numpy as np
import pandas as pd
from base_forecaster import BaseNeuralProphetForecaster
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class MICENeuralProphetForecaster(BaseNeuralProphetForecaster):
    """NeuralProphet forecaster with MICE imputation"""

    def load_and_prepare_data(
        self,
        csv_path="data/Dataset_AM_final.csv",
        campaign_name=None,
        use_train_test_split=True,
        use_mice=True,
    ):
        """Load and prepare CPL data with MICE imputation"""
        # Call parent method to load basic data
        result = super().load_and_prepare_data(
            csv_path, campaign_name, use_train_test_split
        )

        # Apply MICE imputation if enabled (silent mode)
        if use_mice:
            try:
                self.campaign_data = self._apply_mice_imputation(self.campaign_data)
            except Exception as e:
                pass  # Silent fallback to original data

            # Update train/val/test splits with imputed data
            n = len(self.campaign_data)
            train_size = int(n * 0.70)
            val_size = int(n * 0.15)

            train_data = (
                self.campaign_data.iloc[:train_size].copy().reset_index(drop=True)
            )
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

            # Update DataFrames
            self.train_df = pd.DataFrame(
                {"ds": train_data["date"], "y": train_data["Cost_per_Lead_anon"]}
            )
            for col in self.feature_cols:
                if col in train_data.columns:
                    self.train_df[col] = train_data[col].values

            self.val_df = pd.DataFrame(
                {"ds": val_data["date"], "y": val_data["Cost_per_Lead_anon"]}
            )
            for col in self.feature_cols:
                if col in val_data.columns:
                    self.val_df[col] = val_data[col].values

            self.test_df = pd.DataFrame(
                {"ds": test_data["date"], "y": test_data["Cost_per_Lead_anon"]}
            )
            for col in self.feature_cols:
                if col in test_data.columns:
                    self.test_df[col] = test_data[col].values

            self.prophet_df = self.train_df.copy()

        return result

    def _apply_mice_imputation(self, df):
        """Apply MICE imputation to handle missing data"""
        try:
            # Make a copy to avoid modifying original
            df = df.copy()

            # Identify numeric columns that need imputation
            numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

            # Skip if no numeric columns or no missing values
            if len(numeric_cols) == 0:
                return df

            cols_to_drop = []
            for col in numeric_cols:
                missing_pct = df[col].isnull().sum() / len(df)
                if missing_pct > 0.8:
                    cols_to_drop.append(col)
                    print(f"  🗑️  Dropping '{col}' ({missing_pct*100:.1f}% missing)")
            
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

            # Check for missing values
            missing_mask = df[numeric_cols].isnull().any()
            cols_with_missing = numeric_cols[missing_mask]

            if len(cols_with_missing) == 0:
                return df

            # Check for columns with all missing values
            cols_all_missing = []
            cols_for_mice = []

            for col in cols_with_missing:
                if df[col].isnull().all():
                    cols_all_missing.append(col)
                else:
                    cols_for_mice.append(col)

            # Drop columns that are all NaN
            if cols_all_missing:
                df = df.drop(columns=cols_all_missing)

            # If no columns left for MICE, skip it
            if len(cols_for_mice) == 0 or len(df) < 2:
                return df

            # Apply MICE imputation with error handling
            imputer = IterativeImputer(max_iter=10, random_state=42, verbose=0)

            try:
                # Convert to numpy array and back to ensure consistent shapes
                imputed_values = imputer.fit_transform(df[cols_for_mice])

                # Check that shapes match
                if imputed_values.shape[0] == df.shape[0]:
                    # Assign imputed values directly
                    for col_idx, col in enumerate(cols_for_mice):
                        df[col] = imputed_values[:, col_idx]
                    
                    # CRITICAL: Final cleanup - any remaining NaN after MICE
                    remaining_cols = df.select_dtypes(include=["float64", "int64"]).columns
                    remaining_nan = df[remaining_cols].isnull().sum().sum()
                    if remaining_nan > 0:
                        df[remaining_cols] = df[remaining_cols].fillna(method='bfill').fillna(method='ffill').fillna(0)
                else:
                    df[cols_for_mice] = df[cols_for_mice].fillna(method='bfill').fillna(method='ffill').fillna(0)

            except Exception as e:
                df[cols_for_mice] = df[cols_for_mice].fillna(method='bfill').fillna(method='ffill').fillna(0)

            return df

        except Exception as e:
            # Silent fallback
            numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
            df[numeric_cols] = df[numeric_cols].fillna(method='bfill').fillna(method='ffill').fillna(0)
            return df

    def _fill_date_gaps(self, df):
        """Fill gaps in date series"""
        # Create complete date range
        min_date = df["date"].min()
        max_date = df["date"].max()
        complete_dates = pd.date_range(start=min_date, end=max_date, freq="D")

        # Reindex with complete dates
        df_complete = df.set_index("date").reindex(complete_dates).reset_index()
        df_complete.rename(columns={"index": "date"}, inplace=True)

        return df_complete


def main(mode="fast"):
    """Main function for MICE forecaster"""
    from base_forecaster import main as base_main

    return base_main(
        mode=mode, forecaster_class=MICENeuralProphetForecaster, model_type="NP_MICE"
    )


if __name__ == "__main__":
    import sys

    # Check command line arguments for mode
    mode = "fast"  # default
    if len(sys.argv) > 1:
        if sys.argv[1] in ["fast", "production"]:
            mode = sys.argv[1]
        else:
            print("Usage: python mice_forecaster.py [fast|production]")
            print("  fast       - 50 epochs, quick exploration (default)")
            print("  production - 200 epochs max, early stopping enabled")
            sys.exit(1)

    main(mode=mode)
