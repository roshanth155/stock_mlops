import numpy as np
import gymnasium as gym
from gymnasium import spaces


class StockTradingEnv(gym.Env):
    """
    Custom Gym environment for single-asset stock trading.

    State  : [norm_price, norm_portfolio_value, position, rsi, ma_ratio, volatility, cash_ratio]
    Action : 0 = Hold, 1 = Buy (10% of cash), 2 = Sell (10% of holdings)
    Reward : Sharpe-ratio-based risk-adjusted return per step
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, prices: np.ndarray, initial_cash: float = 10_000.0, window: int = 20):
        super().__init__()
        assert len(prices) > window + 1, "Need more price data than window size"

        self.prices        = prices.astype(np.float32)
        self.initial_cash  = initial_cash
        self.window        = window
        self.n_steps       = len(prices) - window - 1

        # Discrete action space: Hold / Buy / Sell
        self.action_space = spaces.Discrete(3)

        # Observation: 7 continuous features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        self.reset()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_rsi(self, idx: int, period: int = 14) -> float:
        if idx < period:
            return 50.0
        deltas = np.diff(self.prices[idx - period: idx + 1])
        gains  = deltas[deltas > 0].mean() if (deltas > 0).any() else 0.0
        losses = -deltas[deltas < 0].mean() if (deltas < 0).any() else 1e-9
        rs     = gains / (losses + 1e-9)
        return float(100 - 100 / (1 + rs))

    def _compute_volatility(self, idx: int) -> float:
        if idx < self.window:
            return 0.0
        returns = np.diff(self.prices[idx - self.window: idx + 1]) / \
                  (self.prices[idx - self.window: idx] + 1e-9)
        return float(np.std(returns))

    def _get_obs(self) -> np.ndarray:
        idx   = self.current_step + self.window
        price = self.prices[idx]
        ma    = self.prices[idx - self.window: idx].mean()

        norm_price      = price / self.prices[self.window]          # relative to start
        norm_port       = self.portfolio_value / self.initial_cash  # relative to initial
        position        = float(self.shares_held > 0)               # 0 or 1
        rsi             = self._compute_rsi(idx) / 100.0            # 0-1
        ma_ratio        = price / (ma + 1e-9)                       # >1 = above MA
        volatility      = self._compute_volatility(idx)
        cash_ratio      = self.cash / (self.portfolio_value + 1e-9)

        return np.array([norm_price, norm_port, position,
                         rsi, ma_ratio, volatility, cash_ratio],
                        dtype=np.float32)

    @property
    def portfolio_value(self) -> float:
        idx = self.current_step + self.window
        return self.cash + self.shares_held * self.prices[idx]

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step  = 0
        self.cash          = float(self.initial_cash)
        self.shares_held   = 0.0
        self.prev_value    = self.initial_cash
        self.returns_hist  = []
        return self._get_obs(), {}

    def step(self, action: int):
        idx   = self.current_step + self.window
        price = float(self.prices[idx])

        # Execute action
        if action == 1:  # Buy 10% of cash
            invest       = self.cash * 0.10
            bought       = invest / (price + 1e-9)
            self.shares_held += bought
            self.cash        -= invest

        elif action == 2:  # Sell 10% of holdings
            sell_shares      = self.shares_held * 0.10
            self.cash        += sell_shares * price
            self.shares_held -= sell_shares

        # Reward: step return normalised by rolling std (Sharpe proxy)
        cur_value   = self.portfolio_value
        step_return = (cur_value - self.prev_value) / (self.prev_value + 1e-9)
        self.returns_hist.append(step_return)
        self.prev_value = cur_value

        if len(self.returns_hist) >= 10:
            mu    = np.mean(self.returns_hist[-20:])
            sigma = np.std(self.returns_hist[-20:]) + 1e-9
            reward = float(mu / sigma)          # Sharpe proxy
        else:
            reward = float(step_return)

        self.current_step += 1
        done = self.current_step >= self.n_steps

        info = {
            "portfolio_value": cur_value,
            "cash":            self.cash,
            "shares_held":     self.shares_held,
            "step_return":     step_return,
        }
        return self._get_obs(), reward, done, False, info

    def render(self, mode="human"):
        idx = self.current_step + self.window
        print(f"Step {self.current_step:4d} | "
              f"Price: {self.prices[idx]:8.2f} | "
              f"Portfolio: {self.portfolio_value:10.2f} | "
              f"Cash: {self.cash:10.2f} | "
              f"Shares: {self.shares_held:.4f}")
