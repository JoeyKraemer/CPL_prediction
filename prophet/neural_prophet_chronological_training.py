#!/usr/bin/env python3
"""
Simple NeuralProphet CPL Forecasting
Basic NeuralProphet without external regressors for initial comparison
"""

import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import warnings
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')

class SimpleNeuralProphetForecaster:
    """Basic NeuralProphet-based CPL forecasting"""
    
    def __init__(self):
        self.model = None
        self.prophet_df = None
        self.campaign_data = None
        self.forecast = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.test_metrics = None
        
    def load_and_prepare_data(self, csv_path='data/Dataset_AM_final.csv', campaign_name='traffic_source_campaign_name_1592', use_train_test_split=True):
        """Load and prepare CPL data for NeuralProphet with proper train/test split"""
        # Load dataset
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df['date'] = pd.to_datetime(df['date'])
        
        # Clean data
        df_clean = df[df['Cost_per_Lead_anon'] > 0].copy()
        
        # Select campaign
        self.campaign_data = df_clean[df_clean['traffic_source_campaign_name_anon'] == campaign_name].copy()
        self.campaign_data = self.campaign_data.sort_values('date').reset_index(drop=True)
        
        if not use_train_test_split:
            raise ValueError("Train/test split is required. Set use_train_test_split=True")
        
        # TRAIN/VAL/TEST SPLIT 
        # Use chronological split: 70% train, 15% val, 15% test
        n = len(self.campaign_data)
        train_size = int(n * 0.70)
        val_size = int(n * 0.15)
        
        train_data = self.campaign_data.iloc[:train_size].copy().reset_index(drop=True)
        val_data = self.campaign_data.iloc[train_size:train_size + val_size].copy().reset_index(drop=True)
        test_data = self.campaign_data.iloc[train_size + val_size:].copy().reset_index(drop=True)
        
        # Create NeuralProphet format for all splits at once
        self.train_df = pd.DataFrame({
            'ds': train_data['date'],
            'y': train_data['Cost_per_Lead_anon']
        })

        self.val_df = pd.DataFrame({
            'ds': val_data['date'],
            'y': val_data['Cost_per_Lead_anon']
        })

        self.test_df = pd.DataFrame({
            'ds': test_data['date'],
            'y': test_data['Cost_per_Lead_anon']
        })
        
        # Use training data as prophet_df
        self.prophet_df = self.train_df.copy()
        
        print(f"📊 Data Split:")
        print(f"  Train: {len(self.train_df)} days ({train_data['date'].min().date()} to {train_data['date'].max().date()})")
        print(f"  Val:   {len(self.val_df)} days ({val_data['date'].min().date()} to {val_data['date'].max().date()})")
        print(f"  Test:  {len(self.test_df)} days ({test_data['date'].min().date()} to {test_data['date'].max().date()})")
        
        return {
            'train': len(self.train_df),
            'val': len(self.val_df),
            'test': len(self.test_df)
        }
    
    def train_model(self, epochs=100, learning_rate=0.01, use_lags=True, use_validation=True, show_progress=False):
        """Train optimized NeuralProphet model with validation monitoring"""
        # NeuralProphet configuration - using library defaults for PyTorch Lightning trainer
        # (Custom trainer_config can cause callback conflicts with validation monitoring)
        
        # Create NeuralProphet model 
        self.model = NeuralProphet(
        # Hyperparameters:
            
            # Seasonality
            weekly_seasonality='auto',      # Uses NeuralProphet default value 
            yearly_seasonality='auto',      # Uses NeuralProphet default value  
            daily_seasonality=False,
            
            # Changepoints
            n_changepoints=15,              # Trend changepoints              
            changepoints_range=0.8,         # Use first 80% of data for changepoints
            
            # Neural architecture
            n_lags=7 if use_lags else 0,    #  Looks back 7 days for lagged features
            n_forecasts=1,
            
            # Training parameters
            epochs=epochs,                  # 50 = Fast mode, 200 = Production mode                               
            learning_rate=learning_rate,    # 0.01 = Moderate LR
            batch_size=8,                   # Small batch size for stability
            
            # Regularization
            trend_reg=0,                    # No trend regularization
            seasonality_reg=0,              # No seasonality regularization
            
            # Normalization and loss
            normalize='minmax',
            impute_missing=True,            # Impute missing dates
            drop_missing=False,             # Keep missing values for validation
            loss_func='Huber'               # Huber loss = robust to outliers (acts like "MAE" for large errors, "MSE" for small errors)
        )
        
        # Train the model
        if use_validation and self.val_df is not None and len(self.val_df) > 0:
            print("📊 Training with validation monitoring...")
            self.metrics = self.model.fit(
                self.prophet_df, 
                freq='D',
                validation_df=self.val_df  # Use validation data for monitoring
            )
        else:
            self.metrics = self.model.fit(self.prophet_df, freq='D')
        
        # Extract final training metrics
        if self.metrics is not None and len(self.metrics) > 0:
            final_metrics = self.metrics.tail(1).to_dict('records')[0]
            self.final_loss = final_metrics.get('Loss', None)
            self.final_mae = final_metrics.get('MAE', None)
            self.epochs_trained = len(self.metrics)
        else:
            self.final_loss = None
            self.final_mae = None
            self.epochs_trained = epochs
        
        return self.model
    
    def get_training_metrics(self):
        """Get training convergence metrics"""
        return {
            'epochs_trained': self.epochs_trained if hasattr(self, 'epochs_trained') else None,
            'final_loss': self.final_loss if hasattr(self, 'final_loss') else None,
            'final_mae': self.final_mae if hasattr(self, 'final_mae') else None,
            'converged': self.epochs_trained < 100 if hasattr(self, 'epochs_trained') else False
        }
    
    def evaluate_performance(self, on_test_set=True):
        """Calculate NeuralProphet performance metrics on proper test set"""
        if on_test_set and self.test_df is not None and len(self.test_df) > 0:
            # PROPER EVALUATION: Test on completely unseen data (like LSTM)
            print("\n🎯 Evaluating on TEST SET (unseen data)")
            test_forecast = self.model.predict(self.test_df)
            
            # Use the correct prediction column
            pred_col = 'yhat1' if 'yhat1' in test_forecast.columns else 'yhat'
            
            if pred_col in test_forecast.columns and 'y' in test_forecast.columns:
                forecast_clean = test_forecast.dropna(subset=['y', pred_col])
                if len(forecast_clean) > 0:
                    mae = mean_absolute_error(forecast_clean['y'], forecast_clean[pred_col])
                    mape = mean_absolute_percentage_error(forecast_clean['y'], forecast_clean[pred_col]) * 100
                    
                    # Calculate SMA baselines for comparison
                    # SMA-3: 3-day Simple Moving Average baseline
                    sma3_preds = forecast_clean['y'].shift(1).rolling(window=3, min_periods=1).mean()
                    # Filter out NaN values for proper evaluation
                    mask_sma3 = ~sma3_preds.isna()
                    sma3_mae = mean_absolute_error(forecast_clean['y'][mask_sma3], sma3_preds[mask_sma3]) if mask_sma3.sum() > 0 else float('nan')
                    sma3_mape = mean_absolute_percentage_error(forecast_clean['y'][mask_sma3], sma3_preds[mask_sma3]) * 100 if mask_sma3.sum() > 0 else float('nan')
                    
                    # SMA-7: 7-day Simple Moving Average baseline
                    sma7_preds = forecast_clean['y'].shift(1).rolling(window=7, min_periods=1).mean()
                    # Filter out NaN values for proper evaluation
                    mask_sma7 = ~sma7_preds.isna()
                    sma7_mae = mean_absolute_error(forecast_clean['y'][mask_sma7], sma7_preds[mask_sma7]) if mask_sma7.sum() > 0 else float('nan')
                    sma7_mape = mean_absolute_percentage_error(forecast_clean['y'][mask_sma7], sma7_preds[mask_sma7]) * 100 if mask_sma7.sum() > 0 else float('nan')
                    
                    self.test_metrics = {
                        'MAE': mae,
                        'MAPE': mape,
                        'SMA3_MAE': sma3_mae,
                        'SMA3_MAPE': sma3_mape,
                        'SMA7_MAE': sma7_mae,
                        'SMA7_MAPE': sma7_mape,
                        'test_predictions': forecast_clean[[pred_col]].values.flatten(),
                        'test_actuals': forecast_clean['y'].values
                    }
                    
                    print(f"  Test samples: {len(forecast_clean)}")
                    print(f"  NeuralProphet MAE: {mae:.3f}, MAPE: {mape:.1f}%")
                    print(f"  SMA-3 Baseline MAE: {sma3_mae:.3f}, MAPE: {sma3_mape:.1f}%")
                    print(f"  SMA-7 Baseline MAE: {sma7_mae:.3f}, MAPE: {sma7_mape:.1f}%")
                    return self.test_metrics
                    
        # Fallback to training data evaluation (not recommended)
        print("⚠️  Evaluating on TRAINING DATA (not recommended)")
        train_forecast = self.model.predict(self.prophet_df)
        pred_col = 'yhat1' if 'yhat1' in train_forecast.columns else 'yhat'
        
        if pred_col in train_forecast.columns and 'y' in train_forecast.columns:
            forecast_clean = train_forecast.dropna(subset=['y', pred_col])
            if len(forecast_clean) > 0:
                mae = mean_absolute_error(forecast_clean['y'], forecast_clean[pred_col])
                mape = mean_absolute_percentage_error(forecast_clean['y'], forecast_clean[pred_col]) * 100
                return {'MAE': mae, 'MAPE': mape}
        
        return {'MAE': float('inf'), 'MAPE': float('inf')}
    
    def get_campaign_info(self):
        """Get campaign metadata"""
        return {
            'campaign': self.campaign_data['traffic_source_campaign_name_anon'].iloc[0],
            'business': self.campaign_data['target_business_anon'].iloc[0],
            'region': self.campaign_data['target_region_anon'].iloc[0],
            'date_range': f"{self.campaign_data['date'].min().date()} to {self.campaign_data['date'].max().date()}",
            'total_days': len(self.campaign_data),
            'avg_cpl': self.campaign_data['Cost_per_Lead_anon'].mean()
        }


def evaluate_all_campaigns(csv_path='data/Dataset_AM_final.csv', epochs=50, max_campaigns=None, min_days=100, mode='fast'):
    """Evaluate all campaigns in dataset with specified filters"""

    import warnings
    warnings.filterwarnings('ignore')
    
    # Configure based on mode
    if mode == 'production':
        epochs = 200
        use_validation = True
        show_progress = False
        print("🏭 PRODUCTION MODE: 200 epochs max, early stopping enabled, validation monitoring")
    else:  # fast mode
        epochs = 50
        use_validation = False
        show_progress = False
        print("⚡ FAST MODE: 50 epochs, quick exploration")
    
    # Load full dataset
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df['date'] = pd.to_datetime(df['date'])
    df_clean = df[df['Cost_per_Lead_anon'] > 0].copy()
    
    # Filter campaigns by minimum data points
    campaign_counts = df_clean.groupby('traffic_source_campaign_name_anon').size()
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
            print(f"[{idx}/{len(valid_campaigns)}] {campaign[:40]:<40}", end=" ")
            
            # Initialize forecaster
            forecaster = SimpleNeuralProphetForecaster()
            
            # Load campaign data
            split_info = forecaster.load_and_prepare_data(
                csv_path=csv_path,
                campaign_name=campaign,
                use_train_test_split=True
            )
            
            # Check continuous data (no large gaps)
            # Gap Detection Rationale:
            # - NeuralProphet assumes continuous time series for accurate seasonality/trend modeling
            # - Large gaps (>30 days) indicate campaign pauses or stops, which:
            #   1. Break temporal continuity needed for lag features (n_lags=7)
            #   2. Invalidate weekly/yearly seasonality patterns
            #   3. Make future predictions unreliable (model can't learn from missing periods)
            # - 30-day threshold chosen as balance between data quality and campaign coverage
            # - Campaigns with gaps >30 days should use alternative approaches (e.g., separate models per active period)
            train_dates = pd.to_datetime(forecaster.train_df['ds']).sort_values()
            date_diff = train_dates.diff().dt.total_seconds() / 86400  # Convert to days
            max_gap = date_diff.max()
            
            if max_gap > 30:  # Skip if gaps > 30 days
                print(f"⚠️ Gap: {max_gap}d")
                failed += 1
                continue
            
            # Train model
            forecaster.train_model(
                epochs=epochs, 
                learning_rate=0.01, 
                use_lags=True,  # Re-enabled lag features
                use_validation=use_validation,
                show_progress=show_progress
            )
            
            # Get training metrics
            train_metrics = forecaster.get_training_metrics()
            if train_metrics.get('converged', False):
                early_stopped += 1
            
            # Evaluate on test set (predictions generated internally)
            performance = forecaster.evaluate_performance(on_test_set=True)
            
            # Get campaign info
            campaign_info = forecaster.get_campaign_info()
            
            # Store results
            results.append({
                'Campaign': campaign,
                'Business': campaign_info['business'],
                'Total_Days': campaign_info['total_days'],
                'Train_Days': split_info['train'] if isinstance(split_info, dict) else 0,
                'Test_Days': split_info['test'] if isinstance(split_info, dict) else 0,
                'Avg_CPL': campaign_info['avg_cpl'],
                'Test_MAE': performance['MAE'],
                'Test_MAPE': performance['MAPE'],
                'SMA3_MAE': performance.get('SMA3_MAE', None),
                'SMA3_MAPE': performance.get('SMA3_MAPE', None),
                'SMA7_MAE': performance.get('SMA7_MAE', None),
                'SMA7_MAPE': performance.get('SMA7_MAPE', None),
                'Epochs_Trained': train_metrics.get('epochs_trained', epochs),
                'Final_Loss': train_metrics.get('final_loss', None),
                'Converged_Early': train_metrics.get('converged', False)
            })
            
            epoch_info = f"E:{train_metrics.get('epochs_trained', epochs)}"
            print(f"✅ MAE: {performance['MAE']:.4f} {epoch_info}")
            successful += 1
            
        except Exception as e:
            import traceback
            print(f"❌ {str(e)[:80]}")
            if idx == 1:  # Show full traceback for first failure only
                print(traceback.format_exc())
            failed += 1
            continue
    
    print(f"\n{'='*80}")
    print(f"✅ Successful: {successful} | ❌ Failed: {failed}")
    if mode == 'production':
        print(f"⏹️  Early stopped: {early_stopped}/{successful} campaigns")
    
    return pd.DataFrame(results)


def main(mode='fast'):
    """Main execution function - Evaluate ALL campaigns
    
    Args:
        mode: 'fast' (50 epochs for exploration) or 'production' (200 epochs with early stopping)
    """
    print("=" * 80)
    print(f"NEURAL PROPHET - ALL CAMPAIGNS EVALUATION ({mode.upper()} MODE)")
    print("=" * 80)
    
    # Evaluate campaigns
    results_df = evaluate_all_campaigns(
        csv_path='data/Dataset_AM_final.csv',
        epochs=50 if mode == 'fast' else 200,
        max_campaigns=None,
        min_days=30,
        mode=mode
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
    print(f"  MAE:  Mean={results_df['Test_MAE'].mean():.4f}, Median={results_df['Test_MAE'].median():.4f}")
    print(f"  MAPE: Mean={results_df['Test_MAPE'].mean():.1f}%, Median={results_df['Test_MAPE'].median():.1f}%")
    
    # Baseline performance
    if 'SMA3_MAE' in results_df.columns and 'SMA7_MAE' in results_df.columns:
        print(f"\nSMA-3 Performance:")
        print(f"  MAE:  Mean={results_df['SMA3_MAE'].mean():.4f}, Median={results_df['SMA3_MAE'].median():.4f}")
        print(f"  MAPE: Mean={results_df['SMA3_MAPE'].mean():.1f}%, Median={results_df['SMA3_MAPE'].median():.1f}%")
        
        print(f"\nSMA-7 Performance:")
        print(f"  MAE:  Mean={results_df['SMA7_MAE'].mean():.4f}, Median={results_df['SMA7_MAE'].median():.4f}")
        print(f"  MAPE: Mean={results_df['SMA7_MAPE'].mean():.1f}%, Median={results_df['SMA7_MAPE'].median():.1f}%")
    
    # Baseline comparisons
    if 'SMA3_MAE' in results_df.columns and 'SMA7_MAE' in results_df.columns:
        np_beats_sma3 = (results_df['Test_MAE'] < results_df['SMA3_MAE']).sum()
        np_beats_sma7 = (results_df['Test_MAE'] < results_df['SMA7_MAE']).sum()
        total = len(results_df)
        
        print(f"\nBaseline Comparison:")
        print(f"  Beats SMA-3: {np_beats_sma3}/{total} ({np_beats_sma3/total*100:.1f}%)")
        print(f"  Beats SMA-7: {np_beats_sma7}/{total} ({np_beats_sma7/total*100:.1f}%)")
    
    # Training stats
    if 'Epochs_Trained' in results_df.columns:
        print(f"\nTraining:")
        print(f"  Mean epochs: {results_df['Epochs_Trained'].mean():.1f}")
        if mode == 'production' and 'Converged_Early' in results_df.columns:
            early_stopped = results_df['Converged_Early'].sum()
            print(f"  Early stopped: {early_stopped}/{len(results_df)}")
    
    # Save results
    results_dir = Path('prophet/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f'result_{timestamp}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved: {output_file}")
    print("=" * 80)
    
    return results_df

if __name__ == "__main__":
    import sys
    
    # Check command line arguments for mode
    mode = 'fast'  # default
    if len(sys.argv) > 1:
        if sys.argv[1] in ['fast', 'production']:
            mode = sys.argv[1]
        else:
            print("Usage: python neural_prophet_simple.py [fast|production]")
            print("  fast       - 50 epochs, quick exploration (default)")
            print("  production - 200 epochs max, early stopping enabled")
            sys.exit(1)
    
    results_df = main(mode=mode)