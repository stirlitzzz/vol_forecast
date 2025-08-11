# vol_forecast

End-to-end pipeline for forecasting equity volatility and generating trading signals.

## Pipeline

1. **Data scraping** – download raw data.
   - `src/load_stock_aggregate_prices.py` downloads minute stock aggregates from Polygon.
   - `src/ivol_scraper.py` pulls option surfaces from IVolatility and writes cleaned features.
2. **Feature generation** – compute realized and derived features.
   - `src/compute_all_vols.py` calculates realized volatility across multiple horizons.
   - `src/features.py` provides HAR feature utilities used by later stages.
3. **Forecasting** – estimate future realized volatility using HAR regression in `src/forecast.py`.
4. **Signaling** – compare forecasts to implied volatility and size positions using `src/signals.py`.
5. **Model training** – train classifiers (e.g., RandomForest) on generated signals with helpers in `src/model_train.py` and `src/training.py`.

## Configuration

Paths and hyperparameters are defined in `config.json` (or the files under `configs/`). Update these to match local data locations:

```json
{
  "realized_path": "data/iv_data.parquet",
  "implied_path": "data/iv_implied.parquet",
  "price_path": "data/underlying_prices.parquet",
  "model_path": "models/best_model.pkl"
}
```

A YAML alternative exists at `configs/config.yaml` for experiment-specific settings (features, model params, thresholds, etc.).

## Dependencies

- Python ≥ 3.9
- pandas, numpy, scikit-learn
- boto3, ivolatility, pandas-market-calendars
- tqdm, python-dotenv
- shap, seaborn, joblib (for diagnostics)

Install with:

```bash
pip install pandas numpy scikit-learn boto3 ivolatility pandas-market-calendars tqdm python-dotenv shap seaborn joblib
```

## Usage

**Download stock data**

```bash
python src/load_stock_aggregate_prices.py --start_date 2022-01-01 --end_date 2023-12-31 --tickers_file data/SPX_constituents_2023_filled.csv
```

**Scrape options and build features**

```bash
python src/ivol_scraper.py --ticker_file data/SPX_tickers.txt --start_date 2023-01-01 --end_date 2023-06-30 --filtered_file data/options_filtered.csv --features_file data/iv_features.csv
```

**Compute realized volatility**

```bash
python src/compute_all_vols.py --input_dir data/filtered_data --output_file output/all_vols.csv --intervals 1min 5min 15min 30min --estimator std
```

**Forecast and generate signals**

```bash
python - <<'PY'
import json, pandas as pd
from src.forecast import build_forecast_pipeline
from src.signals import generate_signals_and_positions
config = json.load(open('config.json'))
realized = pd.read_parquet(config['realized_path'])
implied = pd.read_parquet(config['implied_path'])
# Example: assume no earnings adjustments
mask = pd.DataFrame(False, index=realized.index, columns=realized.columns)
forecast = build_forecast_pipeline(realized, mask, config)
signals, positions = generate_signals_and_positions(forecast, implied)
PY
```

**Train model**

```bash
python - <<'PY'
import json, pandas as pd
from src.model_train import split_train_test_multiple_with_validation
from src.training import train_model_with_validation
config = json.load(open('config.json'))
features = pd.read_parquet('data/features.parquet')
labels = features.pop('target')
(train_X, train_y), (val_X, val_y), _ = split_train_test_multiple_with_validation(features, labels)
model = train_model_with_validation(train_X.values, train_y.values, val_X.values, val_y.values)
PY
```
