"""
train.py — Train DQN agent with MLflow experiment tracking
Usage: python train.py --config configs/dqn_v1.yaml
"""

import argparse
import os
import yaml
import numpy as np
import mlflow
import mlflow.pytorch
import yfinance as yf
from sim.stock_env import StockTradingEnv
from agent.dqn_agent import DQNAgent


# ------------------------------------------------------------------
def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_prices(ticker, start, end):
    print(f"  Fetching {ticker} [{start} → {end}] ...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    prices = df["Close"].values.flatten().astype(float)
    print(f"  {len(prices)} trading days loaded.")
    return prices


def sharpe(returns, rf=0.0):
    r = np.array(returns)
    if r.std() < 1e-9:
        return 0.0
    return float((r.mean() - rf) / r.std() * np.sqrt(252))


def max_drawdown(portfolio_vals):
    vals = np.array(portfolio_vals)
    peak = np.maximum.accumulate(vals)
    return float(((vals - peak) / (peak + 1e-9)).min())


# ------------------------------------------------------------------
def train(cfg: dict, config_path: str):
    os.makedirs("models", exist_ok=True)

    prices = get_prices(cfg["ticker"], cfg["train_start"], cfg["train_end"])
    env    = StockTradingEnv(prices,
                              initial_cash=cfg["initial_cash"],
                              window=cfg["window"])
    agent  = DQNAgent(state_dim=env.observation_space.shape[0],
                      action_dim=env.action_space.n,
                      cfg=cfg)

    # ── MLflow setup ──────────────────────────────────────────────
    mlflow.set_experiment("DQN-Stock-Trading")

    with mlflow.start_run(run_name=f"dqn-{cfg['ticker']}-v{cfg.get('version','1')}") as run:
        print(f"\n  MLflow Run ID : {run.info.run_id}")

        # Log all config params
        mlflow.log_params({
            "ticker":           cfg["ticker"],
            "train_start":      cfg["train_start"],
            "train_end":        cfg["train_end"],
            "initial_cash":     cfg["initial_cash"],
            "gamma":            cfg["gamma"],
            "lr":               cfg["lr"],
            "batch_size":       cfg["batch_size"],
            "buffer_size":      cfg["buffer_size"],
            "epsilon_start":    cfg["epsilon_start"],
            "epsilon_min":      cfg["epsilon_min"],
            "epsilon_decay":    cfg["epsilon_decay"],
            "episodes":         cfg["episodes"],
            "target_update_freq": cfg["target_update_freq"],
        })

        # Log the config file itself as artifact
        mlflow.log_artifact(config_path, artifact_path="configs")

        best_sharpe   = -np.inf
        best_portfolio = 0.0

        print(f"\n{'='*58}")
        print(f"  Ticker: {cfg['ticker']}  |  Episodes: {cfg['episodes']}")
        print(f"{'='*58}\n")

        for ep in range(1, cfg["episodes"] + 1):
            obs, _       = env.reset()
            done         = False
            tot_reward   = 0.0
            step_returns = []
            port_vals    = []
            losses       = []

            while not done:
                action = agent.select_action(obs)
                obs, rew, done, _, info = env.step(action)
                agent.store(obs, action, rew, obs, float(done))
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                tot_reward += rew
                step_returns.append(info["step_return"])
                port_vals.append(info["portfolio_value"])

            final_val  = env.portfolio_value
            ep_sharpe  = sharpe(step_returns)
            ep_mdd     = max_drawdown(port_vals)
            avg_loss   = float(np.mean(losses)) if losses else 0.0
            total_ret  = (final_val - cfg["initial_cash"]) / cfg["initial_cash"] * 100

            # ── Log metrics to MLflow every episode ───────────────
            mlflow.log_metrics({
                "total_reward":     tot_reward,
                "avg_reward":       tot_reward / max(env.current_step, 1),
                "final_portfolio":  final_val,
                "total_return_pct": total_ret,
                "sharpe_ratio":     ep_sharpe,
                "max_drawdown":     ep_mdd,
                "epsilon":          agent.epsilon,
                "loss":             avg_loss,
            }, step=ep)

            # Save best model
            if ep_sharpe > best_sharpe:
                best_sharpe    = ep_sharpe
                best_portfolio = final_val
                agent.save("models/policy_v1.pt")
                # Log model to MLflow
                mlflow.pytorch.log_model(agent.policy_net,
                                          artifact_path="policy_v1")

            if ep % 10 == 0:
                print(f"  Ep {ep:4d} | Reward: {tot_reward:8.3f} | "
                      f"Portfolio: ${final_val:9.2f} | "
                      f"Sharpe: {ep_sharpe:6.3f} | ε: {agent.epsilon:.3f}")

        # Save final explored policy
        agent.save("models/policy_v2_explored.pt")
        mlflow.pytorch.log_model(agent.policy_net,
                                  artifact_path="policy_v2_explored")

        # Log final summary metrics
        mlflow.log_metrics({
            "best_sharpe_ratio":    best_sharpe,
            "best_portfolio_value": best_portfolio,
        }, step=cfg["episodes"])

        # Log both model files as artifacts
        mlflow.log_artifact("models/policy_v1.pt",          "saved_models")
        mlflow.log_artifact("models/policy_v2_explored.pt", "saved_models")

        print(f"\n  ✅ Training complete!")
        print(f"  Best Sharpe     : {best_sharpe:.4f}")
        print(f"  Best Portfolio  : ${best_portfolio:.2f}")
        print(f"  MLflow Run ID   : {run.info.run_id}")
        print(f"\n  👉 View results : mlflow ui")


# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dqn_v1.yaml")
    args = parser.parse_args()
    cfg  = load_cfg(args.config)
    train(cfg, args.config)