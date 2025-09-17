# Reinforcement Market Analysis - RL Trading Bot

This project implements a Market Predictive Reinforcement Learning Trading Bot using free tools on an RTX 3050 laptop with CUDA support.

## Goals
- Predict buy/sell/hold actions for stocks/crypto with 70%+ backtest accuracy and Sharpe >1.0.
- Integrate explainable AI via SHAP for model interpretability.
- Incorporate social sentiment from X (Twitter) data.
- Build a complete pipeline from data acquisition to deployment.

## Project Structure
- `data.py`: Data fetching, preprocessing, and feature engineering.
- `env.py`: Custom Gym environment for trading simulations.
- `agent.py`: RL model training, inference, SHAP explainability, and backtesting.
- `api.py`: FastAPI backend for real-time predictions.
- `app.py`: Streamlit frontend dashboard with authentication and monetization.
- `Dockerfile`: Containerization for deployment.
- `requirements.txt`: Python dependencies.

## Key Features
- Uses Stable-Baselines3 for RL with PPO algorithm.
- CUDA optimization for RTX 3050.
- Sentiment analysis using VADER on simulated X data.
- SHAP visualizations for explainability.
- Backtesting with Backtrader.
- Free APIs: yfinance, ccxt, Alpaca (paper trading).
- Deployment via Docker and Streamlit Cloud.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run data acquisition: Execute functions in `data.py`
3. Train model: Run `agent.py`
4. Start API: `uvicorn api:app --reload`
5. Launch app: `streamlit run app.py`

## Prompt for Copilot
This codebase is generated using a detailed prompt for VS Code Copilot (Grok Code Fast 1 Preview mode) to ensure modular, optimized code with best practices.