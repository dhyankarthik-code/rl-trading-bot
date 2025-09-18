from data import fetch_live_data, fetch_sentiment, preprocess_data
from agent import load_model, predict_action

# Fetch live data for AAPL
df = fetch_live_data('AAPL', is_crypto=False)
print(f"Live AAPL data shape: {df.shape}")

# Fetch sentiment
sentiment = fetch_sentiment('AAPL')
print(f"Sentiment score: {sentiment}")

# Preprocess
train, test = preprocess_data(df, sentiment)
print(f"Train windows: {len(train)}, Test windows: {len(test)}")

if test:
    # Load model
    model = load_model()
    print("Model loaded")

    # Predict on last window
    obs = test[-1]
    action = predict_action(model, obs)
    print(f"Predicted Action for AAPL: {action} (0: Sell, 1: Hold, 2: Buy)")
else:
    print("Not enough data for prediction")