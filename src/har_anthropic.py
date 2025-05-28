import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import warnings
from sklearn.linear_model import HuberRegressor
from scipy import stats
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HARConfig:
    """Configuration for HAR model parameters"""
    daily_window: int = 1
    weekly_window: int = 5  # Business days in a week
    monthly_window: int = 21  # Business days in a month
    min_observations: int = 50  # Minimum obs for regression
    outlier_threshold: float = 3.0  # Z-score threshold
    use_robust_regression: bool = True
    parallel_processing: bool = True
    max_workers: Optional[int] = None

class HARCrossSectionalModel:
    """
    Production-grade HAR (Heterogeneous Autoregressive) Cross-Sectional Volatility Model
    
    Features:
    - Robust regression with outlier handling
    - Vectorized operations where possible
    - Comprehensive error handling and validation
    - Parallel processing for large datasets
    - Statistical diagnostics and model validation
    """
    
    def __init__(self, config: HARConfig = None):
        self.config = config or HARConfig()
        self.vol_matrix = None
        self.features = None
        self.results = None
        self.diagnostics = {}
        
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate input data with comprehensive checks"""
        try:
            df = pd.read_csv(file_path, parse_dates=['date'])
        except Exception as e:
            raise ValueError(f"Failed to load data from {file_path}: {e}")
            
        # Validation checks
        required_cols = ['date', 'ticker', 'annualized_vol_30min']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Data quality checks
        if df['annualized_vol_30min'].isna().sum() / len(df) > 0.5:
            warnings.warn("More than 50% of volatility data is missing")
            
        # Remove obvious outliers (vol > 500% or < 0)
        outlier_mask = (df['annualized_vol_30min'] > 5.0) | (df['annualized_vol_30min'] < 0)
        if outlier_mask.sum() > 0:
            logger.warning(f"Removing {outlier_mask.sum()} extreme outlier observations")
            df = df[~outlier_mask].copy()
            
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
        """Prepare and align data to NYSE calendar"""
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        # NYSE calendar alignment
        nyse = mcal.get_calendar('NYSE')
        global_dates = nyse.valid_days(
            start_date=df['date'].min(), 
            end_date=df['date'].max()
        ).tz_convert(None).normalize()
        
        # Create complete matrix
        vol_matrix = df.pivot(
            index='date', 
            columns='ticker', 
            values='annualized_vol_30min'
        ).reindex(global_dates)
        
        # Log data completeness
        completeness = (1 - vol_matrix.isna().sum().sum() / vol_matrix.size) * 100
        logger.info(f"Data completeness: {completeness:.1f}%")
        
        return vol_matrix, global_dates
    
    def create_har_features(self, vol_matrix: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create HAR features with proper handling of missing data"""
        features = {}
        
        # Lag 1 (daily)
        features['har_1d'] = vol_matrix.shift(self.config.daily_window)
        
        # Weekly average (past 5 days)
        features['har_1w'] = vol_matrix.shift(1).rolling(
            window=self.config.weekly_window, 
            min_periods=max(1, self.config.weekly_window // 2)
        ).mean()
        
        # Monthly average (past 21 days)  
        features['har_1m'] = vol_matrix.shift(1).rolling(
            window=self.config.monthly_window,
            min_periods=max(1, self.config.monthly_window // 2)
        ).mean()
        
        # Market volatility (cross-sectional mean)
        features['market_vol'] = vol_matrix.mean(axis=1, skipna=True).reindex(vol_matrix.index)
        features['market_vol'] = features['market_vol'].shift(1)
        
        # Volatility of volatility (realized vol of daily vols)
        features['vol_of_vol'] = vol_matrix.rolling(
            window=self.config.weekly_window
        ).std().shift(1)
        
        return features
    
    def _run_cross_sectional_regression(self, date_chunk: Tuple[pd.Timestamp, ...]) -> List[Dict]:
        """Run regression for a chunk of dates (for parallel processing)"""
        chunk_results = []
        
        for date in date_chunk:
            try:
                result = self._single_date_regression(date)
                if result is not None:
                    chunk_results.append(result)
            except Exception as e:
                logger.warning(f"Regression failed for date {date}: {e}")
                
        return chunk_results
    
    def _single_date_regression(self, date: pd.Timestamp) -> Optional[Dict]:
        """Run regression for a single date"""
        if date not in self.vol_matrix.index:
            return None
            
        y = self.vol_matrix.loc[date].values
        
        # Construct feature matrix
        X_components = [np.ones_like(y)]  # Intercept
        feature_names = ['intercept']
        
        for name, feature_df in self.features.items():
            if date in feature_df.index:
                X_components.append(feature_df.loc[date].values)
                feature_names.append(name)
        
        if len(X_components) == 1:  # Only intercept
            return None
            
        X = np.column_stack(X_components)
        
        # Remove NaN observations
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        
        if valid_mask.sum() < self.config.min_observations:
            return None
            
        X_valid, y_valid = X[valid_mask], y[valid_mask]
        
        # Outlier detection using Z-score
        if self.config.outlier_threshold > 0:
            z_scores = np.abs(stats.zscore(y_valid, nan_policy='omit'))
            outlier_mask = z_scores < self.config.outlier_threshold
            X_valid, y_valid = X_valid[outlier_mask], y_valid[outlier_mask]
        
        if len(y_valid) < self.config.min_observations:
            return None
        
        # Regression
        if self.config.use_robust_regression:
            model = HuberRegressor(epsilon=1.35, alpha=0.0001)
            model.fit(X_valid, y_valid)
            coefficients = model.coef_
            if hasattr(model, 'intercept_'):
                coefficients = np.concatenate([[model.intercept_], coefficients[1:]])
            predictions = model.predict(X_valid)
        else:
            try:
                coefficients, residuals, rank, s = np.linalg.lstsq(X_valid, y_valid, rcond=None)
                predictions = X_valid @ coefficients
            except np.linalg.LinAlgError:
                return None
        
        # Calculate metrics
        residuals = y_valid - predictions
        rmse = np.sqrt(np.mean(residuals**2))
        r_squared = 1 - np.sum(residuals**2) / np.sum((y_valid - np.mean(y_valid))**2)
        
        # Build result dictionary
        result = {
            'date': date,
            'n_obs': len(y_valid),
            'rmse': rmse,
            'r_squared': r_squared,
            'mean_vol': np.mean(y_valid)
        }
        
        # Add coefficients
        for i, name in enumerate(feature_names):
            if i < len(coefficients):
                result[f'beta_{name}'] = coefficients[i]
        
        return result
    
    def fit(self, file_path: str) -> 'HARCrossSectionalModel':
        """Fit the HAR cross-sectional model"""
        logger.info("Loading and preparing data...")
        df = self.load_and_validate_data(file_path)
        self.vol_matrix, global_dates = self.prepare_data(df)
        
        logger.info("Creating HAR features...")
        self.features = self.create_har_features(self.vol_matrix)
        
        logger.info(f"Running cross-sectional regressions for {len(global_dates)} dates...")
        
        if self.config.parallel_processing and len(global_dates) > 100:
            # Parallel processing for large datasets
            date_chunks = np.array_split(global_dates, 
                                       self.config.max_workers or min(8, len(global_dates)//10))
            
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [executor.submit(self._run_cross_sectional_regression, chunk) 
                          for chunk in date_chunks]
                results = []
                for future in futures:
                    results.extend(future.result())
        else:
            # Sequential processing
            results = self._run_cross_sectional_regression(tuple(global_dates))
        
        self.results = pd.DataFrame(results).set_index('date').sort_index()
        
        # Calculate diagnostics
        self._calculate_diagnostics()
        
        logger.info(f"Model fitted successfully. {len(self.results)} successful regressions.")
        return self
    
    def _calculate_diagnostics(self):
        """Calculate model diagnostics and validation metrics"""
        if self.results is None or len(self.results) == 0:
            return
            
        self.diagnostics = {
            'avg_r_squared': self.results['r_squared'].mean(),
            'avg_rmse': self.results['rmse'].mean(),
            'avg_observations': self.results['n_obs'].mean(),
            'coefficient_stability': {
                col: self.results[col].std() / abs(self.results[col].mean())
                for col in self.results.columns if col.startswith('beta_')
            },
            'successful_regressions': len(self.results),
            'total_dates': len(self.vol_matrix)
        }
    
    def get_coefficients_summary(self) -> pd.DataFrame:
        """Get summary statistics of regression coefficients"""
        if self.results is None:
            raise ValueError("Model must be fitted first")
            
        coef_cols = [col for col in self.results.columns if col.startswith('beta_')]
        summary = self.results[coef_cols].describe()
        
        # Add stability metrics
        stability = pd.Series({
            col: abs(self.results[col].std() / self.results[col].mean()) 
            for col in coef_cols
        }, name='coefficient_of_variation')
        
        return pd.concat([summary, stability.to_frame().T])
    
    def plot_results(self, figsize: Tuple[int, int] = (15, 10)):
        """Create comprehensive plots of model results"""
        if self.results is None:
            raise ValueError("Model must be fitted first")
            
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Coefficient evolution
        coef_cols = [col for col in self.results.columns if col.startswith('beta_')]
        self.results[coef_cols].plot(ax=axes[0,0], title="HAR Coefficients Over Time")
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Model fit quality
        self.results[['r_squared', 'rmse']].plot(ax=axes[0,1], secondary_y=['rmse'],
                                               title="Model Fit Quality")
        
        # Observation count
        self.results['n_obs'].plot(ax=axes[1,0], title="Number of Observations per Regression")
        
        # Coefficient stability (rolling std)
        rolling_std = self.results[coef_cols].rolling(window=21).std()
        rolling_std.plot(ax=axes[1,1], title="Coefficient Stability (21-day Rolling Std)")
        
        plt.tight_layout()
        return fig
    
    def get_diagnostics(self) -> Dict:
        """Return model diagnostics"""
        return self.diagnostics

# Usage example
def main():
    """Example usage of the improved HAR model"""
    
    # Configure model
    config = HARConfig(
        min_observations=50,
        use_robust_regression=True,
        parallel_processing=True,
        outlier_threshold=3.0
    )
    
    # Fit model
    model = HARCrossSectionalModel(config)
    model.fit("../output/all_vols.csv")
    
    # Get results
    print("Model Diagnostics:")
    for key, value in model.get_diagnostics().items():
        print(f"{key}: {value}")
    
    print("\nCoefficient Summary:")
    print(model.get_coefficients_summary())
    
    # Plot results
    fig = model.plot_results()
    
    return model

if __name__ == "__main__":
    model = main()