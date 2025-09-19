import yfinance as yf
import ccxt
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import talib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import nltk
import datetime

# Download NLTK data if needed
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

def fetch_live_data(ticker, is_crypto=False, is_forex=False, is_futures=False):
    """
    Fetch live/today's intraday data for stocks, crypto, forex, or futures.

    Args:
        ticker (str): Ticker symbol (e.g., 'AAPL', 'BTC/USDT', 'EUR/USD').
        is_crypto (bool): True for crypto.
        is_forex (bool): True for forex.
        is_futures (bool): True for futures.

    Returns:
        pd.DataFrame: DataFrame with OHLCV data for today.
    """
    try:
        today = datetime.date.today()
        if is_futures or is_crypto:
            exchange = ccxt.binance()
            # For futures, use perpetual futures
            if is_futures:
                ticker = ticker + ':USDT' if not ticker.endswith(':USDT') else ticker
            since = int((datetime.datetime.combine(today, datetime.time.min) - datetime.timedelta(days=1)).timestamp() * 1000)
            ohlcv = exchange.fetch_ohlcv(ticker, '1m', since=since, limit=1440)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        elif is_forex:
            # Placeholder: Use ccxt for forex if available, else yfinance approximation
            exchange = ccxt.binance()  # Assuming crypto proxy for forex
            ticker = ticker.replace('/', '') + 'T'  # e.g., EURUSD -> EURUSDT
            since = int((datetime.datetime.combine(today, datetime.time.min) - datetime.timedelta(days=1)).timestamp() * 1000)
            ohlcv = exchange.fetch_ohlcv(ticker, '1m', since=since, limit=1440)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        else:
            # Stocks
            df = yf.download(ticker, period='1d', interval='1m')
            if df is not None and not df.empty:
                df.columns = df.columns.droplevel(1)  # Flatten MultiIndex
            return df
    except Exception as e:
        print(f"Error fetching live data: {e}")
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

# --- Advanced / Extended Dataset & Feature Engineering (Phase Upgrade) ---

def fetch_historical_stock(ticker: str, start: str, end: str, interval: str = '1d') -> pd.DataFrame:
    """Fetch multi-year historical OHLCV for a stock using yfinance.

    Args:
        ticker (str): Ticker symbol.
        start (str): 'YYYY-MM-DD'.
        end (str): 'YYYY-MM-DD'.
        interval (str): yfinance interval.
    Returns:
        pd.DataFrame: OHLCV dataframe.
    """
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval)
        if df is not None and not df.empty and isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
    except Exception as e:
        print(f"Error fetching historical stock data: {e}")
        return pd.DataFrame()

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add an expanded feature set for RL model.

    Features include:
      - Log returns & rolling returns
      - Volatility (rolling std)
      - ATR (Average True Range)
      - Price momentum (ROC)
      - Bollinger Band width
      - RSI, MACD (if not already)
      - Time of day (sin/cos encoding for intraday if granular)
    """
    try:
        df = df.copy()
        if 'Close' not in df.columns:
            return df
        # Returns
        df['ret_1'] = df['Close'].pct_change()
        df['log_ret'] = np.log(df['Close']).diff()
        df['ret_5'] = df['Close'].pct_change(5)
        df['ret_10'] = df['Close'].pct_change(10)
        # Volatility
        df['vol_10'] = df['ret_1'].rolling(10).std()
        df['vol_20'] = df['ret_1'].rolling(20).std()
        # ATR
        if {'High','Low','Close'}.issubset(df.columns):
            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values
            df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        # Momentum
        df['roc_10'] = talib.ROC(df['Close'].values, timeperiod=10)
        df['roc_20'] = talib.ROC(df['Close'].values, timeperiod=20)
        # Existing indicators safeguard
        if 'RSI' not in df.columns:
            df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
        if 'MACD' not in df.columns:
            macd, macd_signal, macd_hist = talib.MACD(df['Close'].values)
            df['MACD'] = macd
            df['MACD_signal'] = macd_signal
            df['MACD_hist'] = macd_hist
        # Bollinger bandwidth
        upper, middle, lower = talib.BBANDS(df['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle
        # Time encodings (if intraday index)
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        return df
    except Exception as e:
        print(f"Error adding advanced features: {e}")
        return df

# Example usage
if __name__ == "__main__":
    # Fetch live data
    stock_df = fetch_live_data('AAPL', is_crypto=False)
    if stock_df is not None and not stock_df.empty:
        sentiment = fetch_sentiment('AAPL stock')
        train_data, test_data = preprocess_data(stock_df, sentiment)
        print(f"Live windows: {len(train_data) + len(test_data)}")
        # Save example
        stock_df.to_csv('live_data.csv')