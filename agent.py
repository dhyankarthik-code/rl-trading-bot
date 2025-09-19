from stable_baselines3 import PPO
import torch
import numpy as np
import backtrader as bt
from env import TradingEnv
import matplotlib.pyplot as plt
import os
from typing import List
import shap  # used in explain_with_shap

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True

def train_model(train_data: List, total_timesteps: int = 200000, checkpoint_interval: int = 50000, model_path: str = 'rl_model.zip'):
    """Train the PPO model honoring total_timesteps with optional checkpointing.

    Args:
        train_data (List): Training data windows.
        total_timesteps (int): Total timesteps to learn.
        checkpoint_interval (int): Interval to save intermediate checkpoints.
        model_path (str): Final model save path.
    Returns:
        PPO: Trained model.
    """
    env = TradingEnv(train_data)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PPO('MlpPolicy', env, verbose=1, device=device, tensorboard_log='./logs/')
    steps_done = 0
    while steps_done < total_timesteps:
        remaining = total_timesteps - steps_done
        next_chunk = min(checkpoint_interval, remaining)
        model.learn(total_timesteps=next_chunk, reset_num_timesteps=False)
        steps_done += next_chunk
        ckpt_name = f"checkpoint_{steps_done}.zip"
        model.save(ckpt_name)
        print(f"[Checkpoint] Saved {ckpt_name}")
    model.save(model_path)
    print(f"[Training] Completed {total_timesteps} timesteps. Final model saved to {model_path}.")
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
    print("Direct execution is for future extended training scripts. Use dedicated training script to avoid accidental retraining.")