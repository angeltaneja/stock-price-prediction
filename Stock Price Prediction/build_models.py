"""
Build and train neural networks for equity price forecasting.
Uses scikit-learn MLPRegressor (no TensorFlow). Saves models and normalizers for the web app.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Configuration
TICKER_LIST = ["TSLA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "NFLX", "AMD", "INTC"]
TRAIN_START = "2020-01-01"
TRAIN_END = "2025-04-01"
SEQUENCE_LENGTH = 60
MAX_ITER = 200
HIDDEN_LAYERS = (100, 50)

os.makedirs('saved_models', exist_ok=True)
os.makedirs('normalizers', exist_ok=True)


def build_network():
    """Build a multi-layer perceptron for sequence-to-value regression."""
    return MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation='relu',
        solver='adam',
        max_iter=MAX_ITER,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )


def build_sequences(raw_values, seq_len):
    """Create overlapping input/output sequences from a 1D series."""
    X, y = [], []
    for i in range(seq_len, len(raw_values)):
        X.append(raw_values[i - seq_len:i, 0])
        y.append(raw_values[i, 0])
    return np.array(X), np.array(y)


def train_for_ticker(ticker, start_dt, end_dt, seq_len):
    """Download data, normalize, and train the model for one ticker."""
    try:
        print(f"\n{'='*50}")
        print(f"Training for {ticker}")
        print(f"{'='*50}")

        print(f"Downloading data for {ticker}...")
        try:
            raw = yf.download(ticker, start=start_dt, end=end_dt, progress=False)
        except Exception as e:
            print(f"Download error for {ticker}: {e}")
            time.sleep(60)
            return None

        if raw.empty or 'Close' not in raw:
            print(f"No data available for {ticker}")
            return None

        close_df = raw[['Close']].copy()
        print(f"Rows: {close_df.shape[0]}, Range: {close_df.index[0].strftime('%Y-%m-%d')} to {close_df.index[-1].strftime('%Y-%m-%d')}")

        normalizer = MinMaxScaler(feature_range=(0, 1))
        scaled = normalizer.fit_transform(close_df)

        X, y = build_sequences(scaled, seq_len)

        if len(X) == 0:
            print(f"Not enough history for {ticker} (need at least {seq_len} days)")
            return None

        split = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        print(f"Train size: {len(X_tr)}, Test size: {len(X_te)}")

        print("Building and training network...")
        net = build_network()
        net.fit(X_tr, y_tr)

        train_loss = np.mean((net.predict(X_tr) - y_tr) ** 2)
        test_loss = np.mean((net.predict(X_te) - y_te) ** 2)
        print(f"Train MSE: {train_loss:.6f}, Test MSE: {test_loss:.6f}")

        model_path = f'saved_models/{ticker}_model.pkl'
        norm_path = f'normalizers/{ticker}_normalizer.pkl'

        joblib.dump(net, model_path)
        joblib.dump(normalizer, norm_path)

        print(f"Model saved: {model_path}")
        print(f"Normalizer saved: {norm_path}")

        return {
            'ticker': ticker,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'rows': len(close_df),
            'train_samples': len(X_tr),
            'test_samples': len(X_te),
            'model_path': model_path,
            'norm_path': norm_path
        }

    except Exception as e:
        print(f"Error training {ticker}: {str(e)}")
        return None


def main():
    """Train models for all configured tickers and print a summary."""
    print("Equity Forecast Model Builder (scikit-learn)")
    print(f"Tickers: {', '.join(TICKER_LIST)}")
    print(f"Date range: {TRAIN_START} to {TRAIN_END}")
    print(f"Sequence length: {SEQUENCE_LENGTH} days")

    outcomes = []
    ok_tickers = []
    fail_tickers = []

    for ticker in TICKER_LIST:
        res = train_for_ticker(ticker, TRAIN_START, TRAIN_END, SEQUENCE_LENGTH)
        if res:
            outcomes.append(res)
            ok_tickers.append(ticker)
        else:
            fail_tickers.append(ticker)
        print("Pausing to avoid rate limits...")
        time.sleep(5)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Trained: {len(ok_tickers)}, Failed: {len(fail_tickers)}")
    if ok_tickers:
        print(f"OK: {', '.join(ok_tickers)}")
    if fail_tickers:
        print(f"Failed: {', '.join(fail_tickers)}")

    if outcomes:
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"{'Ticker':<8} {'Train MSE':<12} {'Test MSE':<12} {'Rows':<10} {'Train N':<10}")
        print("-" * 80)
        for r in outcomes:
            print(f"{r['ticker']:<8} {r['train_loss']:<12.6f} {r['test_loss']:<12.6f} {r['rows']:<10} {r['train_samples']:<10}")

    print("\nDone. Models and normalizers are in saved_models/ and normalizers/.")
    return outcomes


if __name__ == "__main__":
    main()
