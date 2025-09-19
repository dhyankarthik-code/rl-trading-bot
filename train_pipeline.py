import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

from agent import train_model, load_model
from data import fetch_historical_stock, fetch_sentiment, preprocess_data, add_advanced_features
from walk_forward import walk_forward_splits, summarize_performance
from env import TradingEnv

# Configuration
TICKER = os.environ.get('TRAIN_TICKER', 'AAPL')
START = os.environ.get('TRAIN_START', '2020-01-01')
END = os.environ.get('TRAIN_END', '2024-12-31')
INTERVAL = os.environ.get('TRAIN_INTERVAL', '1d')
OUTPUT_DIR = 'training_runs'
TIMESTEP_STAGES = [100_000, 300_000, 600_000, 1_000_000]
WALK_TRAIN_DAYS = 365 * 2
WALK_VAL_DAYS = 90
WALK_STEP = 90
WINDOW_SIZE = 50
METRICS_FILE = 'metrics_log.json'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_windows(df: pd.DataFrame, sentiment_score: float):
    train_windows, test_windows = preprocess_data(df, sentiment_score)
    return train_windows, test_windows


def stage_tag(steps):
    return f"stage_{steps//1000}k"


def evaluate_model(model_path: str, val_windows):
    if not val_windows:
        return {}
    env = TradingEnv(val_windows)
    model = load_model() if model_path == 'rl_model.zip' else load_model()
    obs, _ = env.reset()
    equity_curve = []
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        equity_curve.append(info['equity'])
    return summarize_performance(equity_curve)


def append_metrics(stage: str, split_idx: int, metrics: dict):
    entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'stage': stage,
        'split': split_idx,
        **metrics
    }
    existing = []
    if os.path.exists(METRICS_FILE):
        try:
            existing = json.load(open(METRICS_FILE, 'r'))
        except Exception:
            existing = []
    existing.append(entry)
    json.dump(existing, open(METRICS_FILE, 'w'), indent=2)
    print(f"[Metrics] Recorded stage={stage} split={split_idx}: {metrics}")


def main():
    print("[Pipeline] Fetching historical data...")
    df = fetch_historical_stock(TICKER, START, END, INTERVAL)
    if df is None or df.empty:
        print("Empty historical dataset. Abort.")
        return
    df = add_advanced_features(df)
    sentiment = fetch_sentiment(TICKER)

    print("[Pipeline] Starting walk-forward splits...")
    splits = list(walk_forward_splits(df, train_period=WALK_TRAIN_DAYS, val_period=WALK_VAL_DAYS, step=WALK_STEP))
    if not splits:
        print("No splits generated. Abort.")
        return

    for steps in TIMESTEP_STAGES:
        tag = stage_tag(steps)
        print(f"\n[Stage {tag}] Training with target timesteps={steps}")
        for idx, split in enumerate(splits):
            train_df = split['train_df']
            val_df = split['val_df']
            # Build windows from combined train_df
            combined = pd.concat([train_df])
            train_windows, _ = build_windows(combined, sentiment)
            if not train_windows:
                print(f"[Stage {tag}] Split {idx} produced no train windows, skipping")
                continue
            model = train_model(train_windows, total_timesteps=steps, checkpoint_interval=steps//2, model_path='rl_model.zip')
            # Validation windows (fresh build to avoid leakage)
            val_windows, _ = build_windows(val_df, sentiment)
            metrics = evaluate_model('rl_model.zip', val_windows)
            append_metrics(tag, idx, metrics)

    print("[Pipeline] Complete. Metrics stored in metrics_log.json")


if __name__ == '__main__':
    main()
