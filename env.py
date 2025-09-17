import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TradingEnv(gym.Env):
    """
    Custom Gym environment for reinforcement learning trading simulation.
    The agent learns to make buy/sell/hold decisions based on market data windows.
    """
    def __init__(self, data, window_size=50, initial_balance=10000):
        """
        Initialize the trading environment.

        Args:
            data (list): List of numpy arrays, each representing a window of features.
            window_size (int): Size of the observation window.
            initial_balance (float): Starting balance for the portfolio.
        """
        super(TradingEnv, self).__init__()
        self.data = data
        self.window_size = window_size
        self.num_features = data[0].shape[1] if data else 12  # Assuming 12 features
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.transaction_cost = 0.001  # 0.1% per transaction

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: sell, 1: hold, 2: buy
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.window_size, self.num_features), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.

        Returns:
            tuple: (observation, info)
        """
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action (int): Action to take (0: sell, 1: hold, 2: buy).

        Returns:
            tuple: (observation, reward, done, info)
        """
        # Get current price
        current_price = self.data[self.current_step][-1, 3]  # Close price of last step in window

        # Execute action
        if action == 0:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price * (1 - self.transaction_cost)
                self.shares_held = 0
        elif action == 2:  # Buy
            shares_to_buy = self.balance // (current_price * (1 + self.transaction_cost))
            if shares_to_buy > 0:
                self.balance -= shares_to_buy * current_price * (1 + self.transaction_cost)
                self.shares_held += shares_to_buy

        # Calculate portfolio value
        portfolio_value = self.balance + self.shares_held * current_price

        # Reward: Change in portfolio value (simplified, could include Sharpe)
        prev_portfolio = self.balance + self.shares_held * (self.data[max(0, self.current_step-1)][-1, 3] if self.current_step > 0 else current_price)
        reward = portfolio_value - prev_portfolio

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # Get next observation
        obs = self._get_observation() if not done else np.zeros((self.window_size, self.num_features))

        return obs, reward, done, False, {}

    def _get_observation(self):
        """
        Get the current observation window.

        Returns:
            np.array: Current window of features.
        """
        if self.current_step < len(self.data):
            return self.data[self.current_step].astype(np.float32)
        else:
            return np.zeros((self.window_size, self.num_features), dtype=np.float32)

    def render(self, mode='human'):
        """
        Render the environment (optional).
        """
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Shares: {self.shares_held}, Portfolio: {self.balance + self.shares_held * self.data[self.current_step][-1, 3]:.2f}")

# Example usage
if __name__ == "__main__":
    # Dummy data for testing
    dummy_data = [np.random.rand(50, 12) for _ in range(100)]
    env = TradingEnv(dummy_data)
    obs = env.reset()
    obs, _ = obs
    print(f"Observation shape: {obs.shape}")
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    print(f"Reward: {reward}, Done: {done}")