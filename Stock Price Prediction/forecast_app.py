"""
Equity Forecast Web Application
Serves share price forecasts using pre-trained neural networks (scikit-learn, no TensorFlow).
"""
from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from datetime import datetime, timedelta
import time
from pandas.tseries.offsets import BDay
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Folder configuration
DATA_CACHE_DIR = "data_cache"
SAVED_MODELS_DIR = "saved_models"
NORMALIZERS_DIR = "normalizers"
for folder in [DATA_CACHE_DIR, SAVED_MODELS_DIR, NORMALIZERS_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Supported tickers
TICKERS = ["TSLA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "NFLX", "AMD", "INTC"]
WINDOW_SIZE = 60
MAX_ATTEMPTS = 5
BACKOFF_SECONDS = 15


def get_market_data(ticker, from_date, to_date):
    """Retrieve equity data with retries; uses local cache when API is rate-limited."""
    cache_path = os.path.join(DATA_CACHE_DIR, f"{ticker}.csv")

    for attempt in range(MAX_ATTEMPTS):
        try:
            frame = yf.download(ticker, start=from_date, end=to_date, progress=False)
            if frame.empty:
                raise ValueError("Empty response from data source")
            frame.to_csv(cache_path, index=True)
            print(f"Data retrieved and stored for {ticker}")
            return frame
        except Exception as e:
            print(f"Attempt {attempt + 1}/{MAX_ATTEMPTS} failed for {ticker}: {str(e)}")
            if "Too Many Requests" in str(e) and attempt < MAX_ATTEMPTS - 1:
                time.sleep(BACKOFF_SECONDS * (2 ** attempt))
            continue

    if os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path, index_col='Date', parse_dates=True)
            if not cached.empty and 'Close' in cached.columns:
                print(f"Using cached data for {ticker}")
                return cached
        except Exception as e:
            print(f"Cache read error for {ticker}: {e}")
    print(f"Could not load data for {ticker} after {MAX_ATTEMPTS} attempts")
    return None


def load_trained_artifacts(ticker):
    """Load the saved model and price normalizer for the given ticker."""
    model_path = os.path.join(SAVED_MODELS_DIR, f"{ticker}_model.pkl")
    norm_path = os.path.join(NORMALIZERS_DIR, f"{ticker}_normalizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(norm_path):
        print(f"Missing model or normalizer for {ticker}")
        return None, None

    try:
        model = joblib.load(model_path)
        normalizer = joblib.load(norm_path)
        return model, normalizer
    except Exception as e:
        print(f"Error loading artifacts for {ticker}: {e}")
        return None, None


def run_forecast(ticker, horizon_days=10):
    """Generate share price forecasts for the next horizon_days trading days."""
    model, normalizer = load_trained_artifacts(ticker)
    if not model or not normalizer:
        return {'success': False, 'error': f'No trained model for {ticker}'}

    to_date = datetime.now()
    from_date = to_date - timedelta(days=30 + horizon_days + WINDOW_SIZE)
    series = get_market_data(ticker, from_date, to_date)

    if series is None or series.empty or 'Close' not in series.columns:
        return {'success': False, 'error': f'Could not load data for {ticker}. Check connection or try again later.'}

    close_only = series[['Close']].copy()
    scaled = normalizer.transform(close_only)

    forecasts = []
    batch = scaled[-WINDOW_SIZE:].flatten()

    for _ in range(horizon_days):
        X = np.reshape(batch[-WINDOW_SIZE:], (1, WINDOW_SIZE))
        pred = model.predict(X)
        forecasts.append(float(pred[0]))
        batch = np.append(batch, pred[0])

    forecasts = np.array(forecasts).reshape(-1, 1)
    forecasts = normalizer.inverse_transform(forecasts)

    last_trade_date = close_only.index[-1]
    forecast_dates = [last_trade_date + BDay(i) for i in range(1, horizon_days + 1)]
    output = [
        {'step': i + 1, 'date': d.strftime('%Y-%m-%d'), 'value': round(float(v), 2)}
        for i, (d, v) in enumerate(zip(forecast_dates, forecasts.flatten()))
    ]

    latest_price = float(close_only['Close'].iloc[-1])
    return {
        'ticker': ticker,
        'latest_price': round(latest_price, 2),
        'forecasts': output,
        'success': True
    }


@app.route('/')
def home():
    """Serve the main dashboard with available tickers."""
    return render_template('dashboard.html', tickers=TICKERS)


@app.route('/forecast', methods=['POST'])
def forecast():
    """Handle forecast requests via POST."""
    try:
        payload = request.get_json()
        ticker = (payload.get('ticker') or '').upper()
        horizon = int(payload.get('horizon', 10))

        if ticker not in TICKERS:
            return jsonify({'success': False, 'error': f'Ticker {ticker} not supported. Use: {", ".join(TICKERS)}'})
        if not 1 <= horizon <= 30:
            return jsonify({'success': False, 'error': 'Forecast horizon must be between 1 and 30 days'})

        result = run_forecast(ticker, horizon)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})


@app.route('/tickers')
def list_tickers():
    """Return the list of supported equity tickers."""
    return jsonify({'tickers': TICKERS, 'success': True})


if __name__ == '__main__':
    print("Starting Equity Forecast application...")
    if not any(os.path.exists(os.path.join(SAVED_MODELS_DIR, f"{t}_model.pkl")) for t in TICKERS):
        print("No saved models found. Run build_models.py first.")
    if not any(os.path.exists(os.path.join(NORMALIZERS_DIR, f"{t}_normalizer.pkl")) for t in TICKERS):
        print("No normalizers found. Run build_models.py first.")
    app.run(debug=True, host='0.0.0.0', port=5000)
