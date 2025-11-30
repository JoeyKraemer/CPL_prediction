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
        
    def load_and_prepare_data(self, csv_path='data/Dataset_AM_final.csv', campaign_name='traffic_source_campaign_name_1592'):
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

def main():
    """Main execution function"""
    print("CPL Prophet Forecasting")
    print("=" * 50)
    
    # Initialize forecaster
    forecaster = CPLProphetForecaster()
    
    # Load and prepare data
    print("Loading data...")
    days_loaded = forecaster.load_and_prepare_data()
    print(f"Loaded {days_loaded} days of CPL data")
    
    # Train model
    print("Training Prophet model...")
    forecaster.train_model()
    print("Model training complete")
    
    # Make predictions
    print("Generating predictions...")
    forecaster.make_predictions(forecast_days=7)
    
    # Evaluate performance
    performance = forecaster.evaluate_performance()
    print(f"\nProphet Performance:")
    print(f"  MAE: {performance['MAE']:.4f}")
    print(f"  MAPE: {performance['MAPE']:.1f}%")
    
    # Compare with SMA
    sma_comparison = forecaster.compare_with_sma()
    print(f"\nSMA Comparison:")
    print(f"  3-day SMA error: {sma_comparison['errors']['3-day SMA']:.4f}")
    print(f"  7-day SMA error: {sma_comparison['errors']['7-day SMA']:.4f}")
    print(f"  Prophet error: {sma_comparison['errors']['Prophet']:.4f}")
    print(f"  Best method: {sma_comparison['best_method'][0]} ({sma_comparison['best_method'][1]:.4f})")
    
    # Show campaign info
    campaign_info = forecaster.get_campaign_info()
    print(f"\nCampaign Info:")
    print(f"  Campaign: {campaign_info['campaign']}")
    print(f"  Business: {campaign_info['business']}")
    print(f"  Date Range: {campaign_info['date_range']}")
    print(f"  Average CPL: {campaign_info['avg_cpl']:.4f}")
    
    # Show future predictions
    forecast_summary = forecaster.get_forecast_summary()
    print(f"\n7-Day Forecast:")
    print(forecast_summary.to_string(index=False))
    
    return forecaster

if __name__ == "__main__":
    forecaster = main()