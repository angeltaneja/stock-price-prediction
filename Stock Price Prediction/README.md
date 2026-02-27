# Equity Forecast Dashboard

A web application that forecasts equity (stock) share prices for the next 1–30 trading days using a neural network (scikit-learn MLPRegressor). Data is fetched via Yahoo Finance; forecasts are served through a simple Flask API and a dark-themed dashboard. No TensorFlow required (avoids Windows DLL issues).

## What it does

- **Train models**: `build_models.py` downloads historical close prices for a set of tickers, trains a two-layer LSTM, and saves each model and its price normalizer.
- **Run forecasts**: `forecast_app.py` loads the saved models, optionally uses cached market data, and exposes a REST API and HTML dashboard to run forecasts for a chosen ticker and horizon.

## Setup

1. Install dependencies (Python 3.8+):

   ```bash
   pip install -r requirements.txt
   ```
   (The `-r` flag tells pip to read the list of packages from `requirements.txt`.)

2. Train the networks (saves to `saved_models/` and `normalizers/`):

   ```bash
   python build_models.py
   ```

3. Start the web app:

   ```bash
   python forecast_app.py
   ```

4. Open **http://localhost:5000** in a browser and use the dashboard to pick a ticker and forecast horizon (days).

## Project layout

- **forecast_app.py** – Flask server; routes: `/` (dashboard), `/forecast` (POST), `/tickers` (GET).
- **build_models.py** – Downloads data, trains MLP models with scikit-learn, writes `saved_models/<TICKER>_model.pkl` and `normalizers/<TICKER>_normalizer.pkl`.
- **templates/dashboard.html** – Single-page UI to run and view forecasts.
- **data_cache/** – Cached CSV files for each ticker (created when the app fetches data).
- **saved_models/** – Trained scikit-learn models (.pkl).
- **normalizers/** – Joblib-saved MinMaxScaler objects (.pkl).

## Supported tickers

TSLA, AAPL, GOOGL, MSFT, AMZN, META, NVDA, NFLX, AMD, INTC.

## API

- **POST /forecast**  
  Body: `{"ticker": "AAPL", "horizon": 10}`  
  Returns: `latest_price`, `forecasts` (list of `{step, date, value}`), and `success`.

- **GET /tickers**  
  Returns: `{"tickers": [...], "success": true}`.

## Notes

- Forecasts are for **trading days** (business days). Horizon is in calendar days but only weekdays are used for dates.
- If you have existing CSV data in another folder, you can copy it into `data_cache/` with filenames like `AAPL.csv` (Date index, `Close` column) to avoid re-downloading.
