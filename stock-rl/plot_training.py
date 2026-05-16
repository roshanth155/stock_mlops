"""
plot_training.py — Plot training curves from results CSV
Usage: python plot_training.py --csv experiments/results_<run_id>.csv
"""

import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def smooth(vals, w=10):
    return pd.Series(vals).rolling(w, min_periods=1).mean().values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None,
                        help="Path to results CSV. If omitted, latest is used.")
    args = parser.parse_args()

    if args.csv:
        path = args.csv
    else:
        files = sorted(glob.glob("experiments/results_*.csv"))
        if not files:
            print("No results CSV found. Run train.py first.")
            return
        path = files[-1]

    print(f"  Loading {path}")
    df = pd.read_csv(path)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Training Curves  |  Run: {df['run_id'].iloc[0]}", fontsize=13)

    # 1 — Total reward per episode
    axes[0, 0].plot(df["episode"], smooth(df["total_reward"]), color="royalblue")
    axes[0, 0].set_title("Total Reward per Episode")
    axes[0, 0].set_xlabel("Episode"); axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].grid(alpha=0.3)

    # 2 — Sharpe ratio per episode
    axes[0, 1].plot(df["episode"], smooth(df["sharpe_ratio"]), color="seagreen")
    axes[0, 1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0, 1].set_title("Sharpe Ratio per Episode")
    axes[0, 1].set_xlabel("Episode"); axes[0, 1].set_ylabel("Sharpe Ratio")
    axes[0, 1].grid(alpha=0.3)

    # 3 — Final portfolio value per episode
    axes[1, 0].plot(df["episode"], smooth(df["final_portfolio"]), color="darkorange")
    axes[1, 0].axhline(df["final_portfolio"].iloc[0], color="gray",
                       linestyle="--", linewidth=0.8, label="Initial cash")
    axes[1, 0].set_title("Final Portfolio Value per Episode")
    axes[1, 0].set_xlabel("Episode"); axes[1, 0].set_ylabel("Portfolio ($)")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # 4 — Epsilon decay
    axes[1, 1].plot(df["episode"], df["epsilon"], color="tomato")
    axes[1, 1].set_title("Epsilon Decay (Exploration)")
    axes[1, 1].set_xlabel("Episode"); axes[1, 1].set_ylabel("Epsilon")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    out = path.replace(".csv", "_curves.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
