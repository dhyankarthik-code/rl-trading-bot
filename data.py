import yfinance as yf
import ccxt
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import talib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

def fetch_stock_data(ticker, start, end):
    """
    Fetch historical stock data using yfinance.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with OHLCV data.
    """
    try:
        df = yf.download(ticker, start=start, end=end)
        df.columns = df.columns.droplevel(1)  # Flatten MultiIndex
        return df
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def fetch_crypto_data(pair, timeframe='1d', limit=1000):
    """
    Fetch historical crypto data using ccxt from Binance.

    Args:
        pair (str): Trading pair (e.g., 'BTC/USDT').
        timeframe (str): Timeframe for candles (e.g., '1d').
        limit (int): Number of candles to fetch.

    Returns:
        pd.DataFrame: DataFrame with OHLCV data.
    """
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching crypto data: {e}")
        return pd.DataFrame()

def fetch_sentiment(query):
    """
    Simulate fetching sentiment from X (Twitter) data using semantic search.
    For demo, use dummy posts and score with VADER.

    Args:
        query (str): Search query for sentiment analysis.

    Returns:
        float: Average compound sentiment score.
    """
    # Dummy posts for simulation
    dummy_posts = [
        "The market is bullish today!",
        "Crypto prices are crashing.",
        "Neutral news about stocks.",
        "Excited about new tech stocks.",
        "Worried about economic downturn."
    ]
    scores = [sia.polarity_scores(post)['compound'] for post in dummy_posts]
    return np.mean(scores)

def preprocess_data(df, sentiment_score):
    """
    Preprocess the data: clean, normalize, add features, create windows.

    Args:
        df (pd.DataFrame): Raw OHLCV data.
        sentiment_score (float): Sentiment score to add as feature.

    Returns:
        tuple: (train_data, test_data) as lists of windows.
    """
    try:
        # Clean data
        df = df.dropna()

        # # Normalize columns
        # scaler = MinMaxScaler()
        # df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])

        # Add features
        df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'].values)
        df['upper'], df['middle'], df['lower'] = talib.BBANDS(df['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['sentiment'] = sentiment_score

        # Fill NaN from indicators
        df = df.fillna(method='bfill').fillna(method='ffill')

        # # Normalize all columns
        # scaler = MinMaxScaler()
        # df[df.columns] = scaler.fit_transform(df[df.columns])

        # Create sliding windows of 50 steps
        window_size = 50
        windows = []
        for i in range(len(df) - window_size):
            window = df.iloc[i:i+window_size].values
            windows.append(window)

        # Split into train and test
        split_idx = int(0.8 * len(windows))
        train_windows = windows[:split_idx]
        test_windows = windows[split_idx:]

        return train_windows, test_windows
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return [], []

# Example usage
if __name__ == "__main__":
    # Fetch data
    stock_df = fetch_stock_data('AAPL', '2020-01-01', '2023-01-01')
    if not stock_df.empty:
        sentiment = fetch_sentiment('AAPL stock')
        train_data, test_data = preprocess_data(stock_df, sentiment)
        print(f"Train windows: {len(train_data)}, Test windows: {len(test_data)}")
        # Save example
        stock_df.to_csv('data.csv')