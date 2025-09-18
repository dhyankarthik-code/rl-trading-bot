from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from agent import load_model, predict_action
import yfinance as yf
import shap
import talib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from alpaca_trade_api import REST  # Placeholder, need keys
import ccxt
import requests
from datetime import datetime

app = FastAPI()

# Load model at startup
model = load_model()

# Broker integrations (placeholders, need API keys)
# alpaca_api = REST('API_KEY', 'SECRET_KEY', base_url='https://paper-api.alpaca.markets')
binance_exchange = ccxt.binance()

class PredictRequest(BaseModel):
    ticker: str

@app.get('/hub')
def get_hub():
    """
    Return dashboard hub JSON with links to all products.
    """
    return {
        "title": "Supercharts Hub",
        "products": {
            "supercharts": "/supercharts",
            "screeners": "/screener",
            "calendar": "/calendar",
            "news": "/news",
            "portfolio": "/portfolio",
            "options": "/options"
        },
        "integrations": {
            "brokers": ["Alpaca", "Binance"],
            "markets": ["Stocks", "Crypto", "Futures", "Forex", "Indices"]
        }
    }

@app.get('/news')
def get_news(query: str = "trading"):
    """
    Fetch real-time news using NewsAPI free tier, with mock X sentiment.
    """
    try:
        api_key = "YOUR_NEWSAPI_KEY"  # Placeholder, get from newsapi.org
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            # Add mock X sentiment
            sentiments = ["Bullish", "Bearish", "Neutral"]
            for article in articles:
                article["sentiment"] = np.random.choice(sentiments)
            return {"articles": articles}
        else:
            return {"error": "NewsAPI limit reached or invalid key"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/calendar')
def get_calendar():
    """
    Fetch economic calendar events (mock/scraped from free sources).
    """
    # Placeholder: In real, scrape from investing.com or use free API
    events = [
        {"date": "2025-09-19", "event": "Fed Interest Rate Decision", "impact": "High"},
        {"date": "2025-09-20", "event": "Apple Earnings", "impact": "Medium"}
    ]
    return {"events": events}

@app.post('/trade/alpaca')
def trade_alpaca(symbol: str, qty: int, side: str):
    """
    Place order via Alpaca (paper trading).
    """
    try:
        # Placeholder: alpaca_api.submit_order(symbol=symbol, qty=qty, side=side, type='market', time_in_force='gtc')
        return {"status": "Order placed (mock)", "symbol": symbol, "qty": qty, "side": side}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/trade/binance')
def trade_binance(symbol: str, qty: float, side: str):
    """
    Place order via Binance (testnet).
    """
    try:
        # Placeholder: binance_exchange.create_order(symbol, 'market', side, qty)
        return {"status": "Order placed (mock)", "symbol": symbol, "qty": qty, "side": side}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict')
def predict(request: PredictRequest):
    """
    Predict buy/sell/hold action for a given ticker.

    Args:
        request: Request with ticker.

    Returns:
        dict: Action and SHAP explanation.
    """
    try:
        # Fetch live data (last 50 days for window)
        df = yf.download(request.ticker, period='60d')
        if df.empty:
            raise HTTPException(status_code=400, detail="Invalid ticker")

        # Preprocess similar to data.py
        df = df.dropna()
        scaler = MinMaxScaler()
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'])
        df['upper'], df['middle'], df['lower'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['sentiment'] = 0.0  # Placeholder sentiment
        df = df.fillna(method='bfill').fillna(method='ffill')

        # Get last window
        window = df.iloc[-50:].values.astype(np.float32)
        if window.shape[0] < 50:
            raise HTTPException(status_code=400, detail="Not enough data")

        # Predict
        action = predict_action(model, window)

        # SHAP explanation (simplified)
        def model_predict(obs):
            return predict_action(model, obs.reshape(1, 50, -1))

        explainer = shap.KernelExplainer(model_predict, window.reshape(1, -1))
        shap_values = explainer.shap_values(window.reshape(1, -1))
        shap_html = shap.plots.text(shap_values[0], show=False)  # Placeholder for HTML

        return {'action': int(action), 'explanation': str(shap_html)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Alpaca integration for paper trading (placeholder)
# api = REST('API_KEY', 'SECRET_KEY', base_url='https://paper-api.alpaca.markets')
# To place order: api.submit_order(symbol='AAPL', qty=1, side='buy', type='market', time_in_force='gtc')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)