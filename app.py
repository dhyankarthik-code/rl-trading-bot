import streamlit as st
import requests
import plotly.graph_objects as go
import firebase_admin
from firebase_admin import credentials, auth
import stripe
import json
import numpy as np
from agent import load_model, predict_action
from data import fetch_live_data, fetch_sentiment, preprocess_data
from datetime import datetime, timedelta

# Firebase setup (placeholder, need key.json)
# cred = credentials.Certificate('key.json')
# firebase_admin.initialize_app(cred)

# Stripe setup (placeholder, need keys)
# stripe.api_key = 'sk_test_...'

# Load model at startup
model = load_model()
st.write("✅ Model loaded successfully!")

st.title('RL Trading Bot Dashboard')

# Auth placeholder
# user = auth.get_user_by_email('user@example.com')  # Replace with actual auth

# Sidebar for inputs
ticker = st.sidebar.selectbox('Select Ticker', ['AAPL', 'GOOGL', 'BTC-USD'])

if st.sidebar.button('Predict'):
    try:
        # Determine if crypto and adjust ticker
        is_crypto = 'USD' in ticker
        if is_crypto:
            ticker_ccxt = ticker.replace('-USD', '/USDT')  # e.g., BTC-USD -> BTC/USDT
        else:
            ticker_ccxt = ticker
        df = fetch_live_data(ticker_ccxt, is_crypto)
        if df is None or df.empty:
            st.error("Failed to fetch live data for ticker")
        else:
            sentiment = fetch_sentiment(ticker)
            train_data, test_data = preprocess_data(df, sentiment)
            if test_data:
                # Use last window for prediction
                obs = test_data[-1]
                action = predict_action(model, obs)
                st.write(f"Predicted Action: {action} (0: Sell, 1: Hold, 2: Buy)")
                st.write("Note: SHAP explanations removed for deployment simplicity.")
                
                # Placeholder plot
                fig = go.Figure(data=[go.Bar(x=['Feature1', 'Feature2'], y=[0.1, 0.2])])
                st.plotly_chart(fig)
            else:
                st.error("Not enough live data for prediction (need at least 50 data points)")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Monetization
if st.button('Subscribe for Premium'):
    # Create Stripe session
    # session = stripe.checkout.Session.create(
    #     payment_method_types=['card'],
    #     line_items=[{
    #         'price_data': {
    #             'currency': 'usd',
    #             'product_data': {'name': 'Premium Access'},
    #             'unit_amount': 999,
    #         },
    #         'quantity': 1,
    #     }],
    #     mode='payment',
    #     success_url='http://localhost:8501/success',
    #     cancel_url='http://localhost:8501/cancel',
    # )
    # st.write(f"Redirect to: {session.url}")
    st.write("Subscription feature (Stripe integration placeholder)")