"""
train.py — Train DQN agent on Stock Trading Environment
Usage: python train.py --config configs/dqn_v1.yaml
"""

import argparse
import csv
import os
import uuid
import yaml
import numpy as np
import yfinance as yf
from datetime import datetime
from sim.stock_env import StockTradingEnv
from agent.dqn_agent import DQNAgent


# ------------------------------------------------------------------
# Load config
# ------------------------------------------------------------------
def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Download / load price data
# ------------------------------------------------------------------
def get_prices(ticker: str, start: str, end: str) -> np.ndarray:
    print(f"  Fetching {ticker} from {start} to {end} ...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    prices = df["Close"].values.flatten().astype(np.float32)
    print(f"  {len(prices)} trading days loaded.")
    return prices


# ------------------------------------------------------------------
# Sharpe ratio helper
# ------------------------------------------------------------------
def sharpe(returns: list, rf: float = 0.0) -> float:
    r = np.array(returns)
    if r.std() < 1e-9:
        return 0.0
    return float((r.mean() - rf) / r.std() * np.sqrt(252))


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
def train(cfg: dict):
    run_id   = str(uuid.uuid4())[:8]
    exp_dir  = "experiments"
    mdl_dir  = "models"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(mdl_dir, exist_ok=True)

    # ---- Data ----
    prices = get_prices(cfg["ticker"], cfg["train_start"], cfg["train_end"])

    # ---- Env + Agent ----
    env   = StockTradingEnv(prices, initial_cash=cfg["initial_cash"],
                             window=cfg["window"])
    agent = DQNAgent(state_dim=env.observation_space.shape[0],
                     action_dim=env.action_space.n,
                     cfg=cfg)

    # ---- CSV logger ----
    csv_path = os.path.join(exp_dir, f"results_{run_id}.csv")
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=[
        "run_id", "episode", "total_reward", "avg_reward",
        "final_portfolio", "sharpe_ratio", "avg_wait_proxy",
        "epsilon", "lr", "loss"
    ])
    writer.writeheader()

    print(f"\n{'='*55}")
    print(f"  Run ID : {run_id}")
    print(f"  Ticker : {cfg['ticker']}  |  Episodes: {cfg['episodes']}")
    print(f"{'='*55}\n")

    best_sharpe  = -np.inf
    ep_rewards   = []

    for ep in range(1, cfg["episodes"] + 1):
        obs, _     = env.reset()
        done       = False
        tot_reward = 0.0
        step_returns = []
        losses     = []
        init_val   = env.portfolio_value

        while not done:
            action          = agent.select_action(obs)
            next_obs, rew, done, _, info = env.step(action)
            agent.store(obs, action, rew, next_obs, float(done))
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            obs         = next_obs
            tot_reward += rew
            step_returns.append(info["step_return"])

        final_val  = env.portfolio_value
        ep_sharpe  = sharpe(step_returns)
        avg_reward = tot_reward / env.current_step
        avg_loss   = float(np.mean(losses)) if losses else 0.0
        ep_rewards.append(tot_reward)

        row = {
            "run_id":           run_id,
            "episode":          ep,
            "total_reward":     round(tot_reward, 4),
            "avg_reward":       round(avg_reward, 6),
            "final_portfolio":  round(final_val, 2),
            "sharpe_ratio":     round(ep_sharpe, 4),
            "avg_wait_proxy":   round(1.0 / (ep_sharpe + 1e-9), 4),  # inverse sharpe proxy
            "epsilon":          round(agent.epsilon, 4),
            "lr":               cfg["lr"],
            "loss":             round(avg_loss, 6),
        }
        writer.writerow(row)
        csv_file.flush()

        if ep % 10 == 0:
            print(f"  Ep {ep:4d} | Reward: {tot_reward:8.4f} | "
                  f"Portfolio: ${final_val:9.2f} | "
                  f"Sharpe: {ep_sharpe:6.3f} | ε: {agent.epsilon:.3f}")

        # Save best policy
        if ep_sharpe > best_sharpe:
            best_sharpe = ep_sharpe
            agent.save(os.path.join(mdl_dir, "policy_v1.pt"))

    # Save final (most-explored) policy
    agent.save(os.path.join(mdl_dir, "policy_v2_explored.pt"))

    csv_file.close()
    print(f"\n  Training complete!")
    print(f"  Best Sharpe  : {best_sharpe:.4f}")
    print(f"  Results CSV  : {csv_path}")
    print(f"  Models saved : models/policy_v1.pt  |  policy_v2_explored.pt")


# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dqn_v1.yaml")
    args   = parser.parse_args()
    cfg    = load_cfg(args.config)
    train(cfg)
