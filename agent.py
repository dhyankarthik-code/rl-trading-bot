from stable_baselines3 import PPO
import torch
import numpy as np
import backtrader as bt
from env import TradingEnv
import matplotlib.pyplot as plt

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True

def train_model(train_data, total_timesteps=200000):
    """
    Train the PPO model on the trading environment.

    Args:
        train_data (list): Training data windows.
        total_timesteps (int): Number of training steps.

    Returns:
        PPO: Trained model.
    """
    env = TradingEnv(train_data)
    model = PPO('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log='./logs/')
    model.learn(total_timesteps=10000)
    model.save('rl_model.zip')
    return model

def load_model():
    """
    Load the trained model.

    Returns:
        PPO: Loaded model.
    """
    return PPO.load('rl_model.zip')

def predict_action(model, obs):
    """
    Predict action from observation.

    Args:
        model: Trained PPO model.
        obs (np.array): Observation window.

    Returns:
        int: Predicted action.
    """
    action, _ = model.predict(obs)
    return action

def explain_with_shap(model, data_sample, test_obs):
    """
    Generate SHAP explanations for the model predictions.

    Args:
        model: Trained model.
        data_sample (np.array): Sample data for explainer.
        test_obs (np.array): Test observations.
    """
    def model_predict(obs):
        # Wrap predict to return probabilities or values for SHAP
        actions, _ = model.predict(obs)
        return actions  # Or use policy probabilities if available

    explainer = shap.KernelExplainer(model_predict, data_sample)
    shap_values = explainer.shap_values(test_obs[:10])  # Limit for demo
    shap.summary_plot(shap_values, test_obs[:10])
    plt.savefig('shap_summary.png')
    print("SHAP summary plot saved.")

class RLStrategy(bt.Strategy):
    """
    Backtrader strategy using the trained RL model.
    """
    def __init__(self):
        self.model = load_model()
        self.dataclose = self.datas[0].close

    def next(self):
        # Get current observation (simplified, need to build window)
        # For demo, use dummy obs
        obs = np.random.rand(50, 12).astype(np.float32)  # Replace with actual window
        action = predict_action(self.model, obs)

        if action == 2:  # Buy
            self.buy()
        elif action == 0:  # Sell
            self.sell()

def backtest_model(test_df):
    """
    Backtest the model using Backtrader.

    Args:
        test_df (pd.DataFrame): Test data DataFrame.
    """
    cerebro = bt.Cerebro()
    cerebro.addstrategy(RLStrategy)

    # Add data feed
    data = bt.feeds.PandasData(dataname=test_df)
    cerebro.adddata(data)

    # Run backtest
    cerebro.run()
    cerebro.plot()

# Example usage
if __name__ == "__main__":
    # Assume train_data and test_data from data.py
    from data import preprocess_data, fetch_stock_data, fetch_sentiment

    stock_df = fetch_stock_data('AAPL', '2020-01-01', '2023-01-01')
    sentiment = fetch_sentiment('AAPL')
    train_data, test_data = preprocess_data(stock_df, sentiment)

    if train_data:
        model = train_model(train_data)
        # Explain
        # data_sample = np.array(train_data[:10])
        # test_obs = np.array(test_data[:10])
        # explain_with_shap(model, data_sample, test_obs)

        # Backtest (need test_df, assume stock_df test portion)
        # test_df = stock_df[int(0.8 * len(stock_df)):]
        # backtest_model(test_df)