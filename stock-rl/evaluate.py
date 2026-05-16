"""
evaluate.py — Compare DQN policy vs Buy-and-Hold baseline
Usage: python evaluate.py --config configs/dqn_v1.yaml --model models/policy_v1.pt
"""

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import yfinance as yf
from sim.stock_env import StockTradingEnv
from agent.dqn_agent import DQNAgent


# ------------------------------------------------------------------
def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_prices(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    return df["Close"].values.flatten().astype(np.float32)


def sharpe(returns, rf=0.0):
    r = np.array(returns)
    if r.std() < 1e-9:
        return 0.0
    return float((r.mean() - rf) / r.std() * np.sqrt(252))


def max_drawdown(portfolio_vals):
    vals  = np.array(portfolio_vals)
    peak  = np.maximum.accumulate(vals)
    dd    = (vals - peak) / (peak + 1e-9)
    return float(dd.min())


# ------------------------------------------------------------------
# Run DQN policy (greedy — no exploration)
# ------------------------------------------------------------------
def run_dqn(env, agent):
    obs, _ = env.reset()
    done   = False
    port_vals, step_rets, rewards = [], [], []

    while not done:
        action = agent.select_action(obs)   # epsilon is low after load
        obs, rew, done, _, info = env.step(action)
        port_vals.append(info["portfolio_value"])
        step_rets.append(info["step_return"])
        rewards.append(rew)

    return port_vals, step_rets, rewards


# ------------------------------------------------------------------
# Buy-and-Hold baseline
# ------------------------------------------------------------------
def run_buy_and_hold(prices, initial_cash, window):
    start_price = prices[window]
    shares      = initial_cash / start_price
    port_vals   = [shares * p for p in prices[window:]]
    step_rets   = list(np.diff(prices[window:]) / (prices[window:-1] + 1e-9))
    return port_vals, step_rets


# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dqn_v1.yaml")
    parser.add_argument("--model",  default="models/policy_v1.pt")
    args = parser.parse_args()
    cfg  = load_cfg(args.config)

    prices = get_prices(cfg["ticker"], cfg["eval_start"], cfg["eval_end"])
    env    = StockTradingEnv(prices, initial_cash=cfg["initial_cash"],
                              window=cfg["window"])

    # Load agent (greedy — set epsilon to 0)
    agent         = DQNAgent(env.observation_space.shape[0],
                             env.action_space.n, cfg)
    agent.load(args.model)
    agent.epsilon = 0.0   # pure greedy during eval

    # Run both
    dqn_vals,  dqn_rets,  dqn_rews  = run_dqn(env, agent)
    bnh_vals,  bnh_rets              = run_buy_and_hold(
        prices, cfg["initial_cash"], cfg["window"])

    # Align lengths
    n = min(len(dqn_vals), len(bnh_vals))
    dqn_vals, bnh_vals = dqn_vals[:n], bnh_vals[:n]
    dqn_rets, bnh_rets = dqn_rets[:n-1], bnh_rets[:n-1]

    # ------------------------------------------------------------------
    # Metrics table
    # ------------------------------------------------------------------
    metrics = {
        "Final Portfolio ($)":  (round(dqn_vals[-1], 2),  round(bnh_vals[-1], 2)),
        "Total Return (%)":     (round((dqn_vals[-1]/cfg["initial_cash"]-1)*100, 2),
                                 round((bnh_vals[-1]/cfg["initial_cash"]-1)*100, 2)),
        "Sharpe Ratio":         (round(sharpe(dqn_rets), 4), round(sharpe(bnh_rets), 4)),
        "Max Drawdown (%)":     (round(max_drawdown(dqn_vals)*100, 2),
                                 round(max_drawdown(bnh_vals)*100, 2)),
        "Avg Step Return":      (round(np.mean(dqn_rets)*100, 4),
                                 round(np.mean(bnh_rets)*100, 4)),
    }

    print(f"\n{'='*55}")
    print(f"  Evaluation on {cfg['ticker']}  [{cfg['eval_start']} → {cfg['eval_end']}]")
    print(f"{'='*55}")
    print(f"  {'Metric':<25} {'DQN-Policy':>12} {'Buy&Hold':>12}")
    print(f"  {'-'*50}")
    for k, (dqn_v, bnh_v) in metrics.items():
        print(f"  {k:<25} {str(dqn_v):>12} {str(bnh_v):>12}")
    print(f"{'='*55}\n")

    # ------------------------------------------------------------------
    # Plot 1 — Portfolio value over time
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"DQN vs Buy-and-Hold  |  {cfg['ticker']}  |  "
                 f"{cfg['eval_start']} → {cfg['eval_end']}", fontsize=13)

    axes[0].plot(dqn_vals, label="DQN Policy",    color="royalblue", linewidth=1.5)
    axes[0].plot(bnh_vals, label="Buy & Hold",    color="tomato",    linewidth=1.5, linestyle="--")
    axes[0].axhline(cfg["initial_cash"], color="gray", linestyle=":", linewidth=1)
    axes[0].set_title("Portfolio Value Over Time")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2 — Cumulative reward over steps (DQN)
    cum_rew = np.cumsum(dqn_rews)
    axes[1].plot(cum_rew, color="seagreen", linewidth=1.5)
    axes[1].set_title("Cumulative Reward (DQN Policy)")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Cumulative Reward")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiments/eval_comparison.png", dpi=150)
    plt.show()
    print("  Plot saved → experiments/eval_comparison.png")


if __name__ == "__main__":
    main()
