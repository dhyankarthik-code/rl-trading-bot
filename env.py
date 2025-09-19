import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TradingEnv(gym.Env):
    """Trading environment with risk-adjusted reward and basic risk controls.

    Reward components:
      base_return: (equity_t - equity_{t-1}) / equity_{t-1}
      - cost_penalty: transaction costs applied when trades occur
      - volatility_penalty: lambda_vol * rolling_std(window equity returns)
      - drawdown_penalty: lambda_dd * current_drawdown
      - trade_frequency_penalty: lambda_trades if over-trading
      - inactivity_penalty: optional penalty for not trading when conditions favorable (placeholder)

    Future extensions:
      - Regime awareness
      - Position sizing based on volatility
      - Adaptive risk budget
    """
    def __init__(self,
                 data,
                 window_size: int = 50,
                 initial_balance: float = 10000,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 lambda_vol: float = 0.1,
                 lambda_dd: float = 0.2,
                 lambda_trades: float = 0.001,
                 rolling_vol_window: int = 20,
                 max_drawdown_stop: float = 0.4):
        super().__init__()
        self.data = data
        self.window_size = window_size
        self.num_features = data[0].shape[1] if data else 12
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.last_equity = initial_balance
        self.equity_curve = []
        self.returns_window = []
        self.trade_count = 0
        self.buy_price = None

        # Risk & cost parameters
        self.commission = commission
        self.slippage = slippage
        self.lambda_vol = lambda_vol
        self.lambda_dd = lambda_dd
        self.lambda_trades = lambda_trades
        self.rolling_vol_window = rolling_vol_window
        self.max_drawdown_stop = max_drawdown_stop
        self.equity_peak = initial_balance

        # Gym spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.window_size, self.num_features), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.last_equity = self.initial_balance
        self.equity_curve = [self.initial_balance]
        self.returns_window = []
        self.trade_count = 0
        self.equity_peak = self.initial_balance
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        # Current close price (assumes column 3 is close)
        current_price = float(self.data[self.current_step][-1, 3])

        # Track starting equity
        starting_equity = self.balance + self.shares_held * current_price

        # Execute action with basic position sizing (all-in/out). Placeholder for advanced sizing.
        trade_executed = False
        if action == 0:  # Sell
            if self.shares_held > 0:
                proceeds_price = current_price * (1 - self.slippage)
                proceeds = self.shares_held * proceeds_price
                cost = proceeds * self.commission
                self.balance += proceeds - cost
                self.shares_held = 0
                trade_executed = True
        elif action == 2:  # Buy
            if self.balance > 0:
                exec_price = current_price * (1 + self.slippage)
                shares_to_buy = int(self.balance / (exec_price * (1 + self.commission)))
                if shares_to_buy > 0:
                    spend = shares_to_buy * exec_price
                    cost = spend * self.commission
                    total = spend + cost
                    self.balance -= total
                    self.shares_held += shares_to_buy
                    trade_executed = True

        if trade_executed:
            self.trade_count += 1

        # Updated equity
        equity = self.balance + self.shares_held * current_price
        self.equity_curve.append(equity)
        if equity > self.equity_peak:
            self.equity_peak = equity

        # Base return
        base_return = (equity - starting_equity) / max(starting_equity, 1e-9)
        self.returns_window.append(base_return)
        if len(self.returns_window) > self.rolling_vol_window:
            self.returns_window.pop(0)

        # Rolling volatility
        rolling_vol = np.std(self.returns_window) if len(self.returns_window) >= 2 else 0.0
        # Drawdown
        drawdown = (self.equity_peak - equity) / max(self.equity_peak, 1e-9)

        # Penalties
        vol_penalty = self.lambda_vol * rolling_vol
        dd_penalty = self.lambda_dd * drawdown
        trade_freq_penalty = self.lambda_trades * self.trade_count / (self.current_step + 1)

        reward = base_return - vol_penalty - dd_penalty - trade_freq_penalty

        # Episode termination conditions
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1 or drawdown >= self.max_drawdown_stop
        truncated = False

        obs = self._get_observation() if not done else np.zeros((self.window_size, self.num_features), dtype=np.float32)

        info = {
            'equity': equity,
            'base_return': base_return,
            'rolling_vol': rolling_vol,
            'drawdown': drawdown,
            'trade_count': self.trade_count,
            'reward_components': {
                'base': base_return,
                'vol_penalty': -vol_penalty,
                'dd_penalty': -dd_penalty,
                'trade_freq_penalty': -trade_freq_penalty
            }
        }
        return obs, reward, done, truncated, info

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
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    print(f"Reward: {reward}, Done: {done}")