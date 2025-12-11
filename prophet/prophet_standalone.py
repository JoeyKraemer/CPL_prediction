#!/usr/bin/env python3
"""
Prophet CPL Forecasting - Production Version
Standalone Prophet model for CPL prediction with SMA benchmarking
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

class CPLProphetForecaster:
    """Prophet-based CPL forecasting with performance benchmarking"""
    
    def __init__(self):
        self.model = None
        self.prophet_df = None
        self.campaign_data = None
        self.forecast = None
        
    def load_and_prepare_data(self, csv_path='../data/Dataset_AM_final.csv', campaign_name='traffic_source_campaign_name_1592'):
        """Load and prepare CPL data for Prophet"""
        # Load dataset
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df['date'] = pd.to_datetime(df['date'])
        
        # Clean data
        df_clean = df[df['Cost_per_Lead_anon'] > 0].copy()
        
        # Select campaign
        self.campaign_data = df_clean[df_clean['traffic_source_campaign_name_anon'] == campaign_name].copy()
        self.campaign_data = self.campaign_data.sort_values('date').reset_index(drop=True)
        
        # Create Prophet format
        self.prophet_df = pd.DataFrame({
            'ds': self.campaign_data['date'],
            'y': self.campaign_data['Cost_per_Lead_anon']
        })
        
        return len(self.campaign_data)
    
    def train_model(self):
        """Train Prophet model"""
        self.model = Prophet(
            daily_seasonality='auto',
            weekly_seasonality='auto',
            yearly_seasonality='auto',
            seasonality_mode='additive'
        )
        
        self.model.fit(self.prophet_df)
        return self.model
    
    def make_predictions(self, forecast_days=7):
        """Generate predictions for specified number of days"""
        future = self.model.make_future_dataframe(periods=forecast_days)
        self.forecast = self.model.predict(future)
        return self.forecast
    
    def evaluate_performance(self):
        """Calculate Prophet performance metrics"""
        historical_predictions = self.forecast.head(len(self.prophet_df))['yhat']
        
        mae = mean_absolute_error(self.prophet_df['y'], historical_predictions)
        mape = mean_absolute_percentage_error(self.prophet_df['y'], historical_predictions) * 100
        
        return {'MAE': mae, 'MAPE': mape}
    
    def compare_with_sma(self):
        """Compare Prophet with Simple Moving Average benchmarks"""
        recent_data = self.prophet_df.tail(30)
        sma_3 = recent_data['y'].tail(3).mean()
        sma_7 = recent_data['y'].tail(7).mean()
        
        last_actual = self.prophet_df['y'].iloc[-1]
        prophet_pred = self.forecast.iloc[len(self.prophet_df)-1]['yhat']
        
        errors = {
            '3-day SMA': abs(last_actual - sma_3),
            '7-day SMA': abs(last_actual - sma_7),
            'Prophet': abs(last_actual - prophet_pred)
        }
        
        best_method = min(errors.items(), key=lambda x: x[1])
        
        return {
            'predictions': {'3-day SMA': sma_3, '7-day SMA': sma_7, 'Prophet': prophet_pred, 'Actual': last_actual},
            'errors': errors,
            'best_method': best_method
        }
    
    def get_forecast_summary(self):
        """Get summary of future predictions"""
        future_pred = self.forecast.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        summary = pd.DataFrame({
            'Date': future_pred['ds'].dt.date,
            'Predicted_CPL': future_pred['yhat'].values,
            'Lower_Bound': future_pred['yhat_lower'].values,
            'Upper_Bound': future_pred['yhat_upper'].values
        })
        
        return summary
    
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
    
    # Baseline metrics
    if 'SMA3_MAE' in results_df.columns and 'SMA7_MAE' in results_df.columns:
        print(f"\nSMA-3 Baseline:")
        print(f"  MAE:  Mean={results_df['SMA3_MAE'].mean():.4f}, Median={results_df['SMA3_MAE'].median():.4f}")
        print(f"  MAPE: Mean={results_df['SMA3_MAPE'].mean():.1f}%, Median={results_df['SMA3_MAPE'].median():.1f}%")
        
        print(f"\nSMA-7 Baseline:")
        print(f"  MAE:  Mean={results_df['SMA7_MAE'].mean():.4f}, Median={results_df['SMA7_MAE'].median():.4f}")
        print(f"  MAPE: Mean={results_df['SMA7_MAPE'].mean():.1f}%, Median={results_df['SMA7_MAPE'].median():.1f}%")
        
        # Win rates
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
    forecaster = main()