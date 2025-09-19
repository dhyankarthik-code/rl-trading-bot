import numpy as np
import pandas as pd
from typing import Generator, Tuple, List, Dict, Any

"""Walk-Forward Splitting Utilities

This module provides rolling (walk-forward) train/validation splits for time series.

Core design:
  - Expanding or rolling window strategies
  - Generates (train_indices, val_indices) or sliced DataFrames
  - Can adapt to windowed data (list of windows) already produced

Example usage (for raw OHLC dataframe before window assembly):

for split in walk_forward_splits(df, train_period=365*2, val_period=90, step=90):
    train_df, val_df = split['train_df'], split['val_df']
    # build windows -> train model -> evaluate

For pre-windowed arrays (list of np.ndarray):

for wf in walk_forward_windows(windows, train_windows=2000, val_windows=500, step=500):
    train_windows_list = wf['train']
    val_windows_list = wf['val']

"""

def walk_forward_splits(df: pd.DataFrame,
                         train_period: int,
                         val_period: int,
                         step: int,
                         min_observations: int = 1000) -> Generator[Dict[str, Any], None, None]:
    """Generate rolling walk-forward (expanding) splits on a DataFrame.

    Args:
        df: Time-indexed dataframe.
        train_period: Number of rows for training slice.
        val_period: Number of rows for validation slice.
        step: Rows to advance each iteration.
        min_observations: Minimum total required rows.
    Yields:
        dict with train_df, val_df, start_idx, end_idx
    """
    if len(df) < min_observations:
        return
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_period
        val_end = train_end + val_period
        if val_end > len(df):
            break
        train_df = df.iloc[train_start:train_end]
        val_df = df.iloc[train_end:val_end]
        yield {
            'train_df': train_df,
            'val_df': val_df,
            'train_range': (train_start, train_end),
            'val_range': (train_end, val_end)
        }
        start += step


def walk_forward_windows(windows: List[np.ndarray],
                          train_windows: int,
                          val_windows: int,
                          step: int) -> Generator[Dict[str, Any], None, None]:
    """Walk-forward over precomputed windows.

    Args:
        windows: List of window arrays.
        train_windows: Count of windows for training.
        val_windows: Count of windows for validation.
        step: Advance count.
    Yields:
        dict with train (list), val (list), indices metadata.
    """
    total = len(windows)
    start = 0
    while True:
        t_end = start + train_windows
        v_end = t_end + val_windows
        if v_end > total:
            break
        yield {
            'train': windows[start:t_end],
            'val': windows[t_end:v_end],
            'train_range': (start, t_end),
            'val_range': (t_end, v_end)
        }
        start += step


def summarize_performance(equity_curve: List[float]) -> Dict[str, float]:
    """Compute basic performance metrics from an equity curve.

    Args:
        equity_curve: Sequence of equity values over time.
    Returns:
        dict with sharpe, sortino, max_drawdown, returns_total.
    """
    if len(equity_curve) < 5:
        return {}
    eq = np.array(equity_curve)
    returns = np.diff(eq) / eq[:-1]
    avg = returns.mean()
    std = returns.std() + 1e-9
    sharpe = (avg / std) * np.sqrt(252)  # daily-ish assumption
    downside = returns[returns < 0]
    sortino = (avg / (downside.std() + 1e-9)) * np.sqrt(252) if len(downside) > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    max_dd = dd.max()
    total_ret = eq[-1] / eq[0] - 1
    return {
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_drawdown': float(max_dd),
        'total_return': float(total_ret)
    }
