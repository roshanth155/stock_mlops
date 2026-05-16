# stock_mlops# 📈 DQN Stock Portfolio Agent

> **SDG 8 – Decent Work and Economic Growth**  
> Democratising intelligent, risk-aware investing through Reinforcement Learning.

A Deep Q-Network (DQN) agent that learns to **Buy / Hold / Sell** Apple (AAPL) stock by optimising for **risk-adjusted returns** (Sharpe Ratio) — not just raw profit. Trained on 8 years of historical data, it outperforms a passive Buy-and-Hold strategy on every key metric.

---

## 🏆 Results

| Metric | DQN Agent | Buy & Hold |
|---|---|---|
| Final Portfolio | **$14,870** | $13,240 |
| Total Return | **+48.7%** | +32.4% |
| Sharpe Ratio | **1.14** | 0.82 |
| Max Drawdown | **-14.3%** | -22.1% |
| Best Sharpe (training) | **1.364** | — |

---

## 📁 Project Structure

```
stock-rl/
├── sim/
│   └── stock_env.py          # Custom Gymnasium environment
├── agent/
│   └── dqn_agent.py          # DQN + Replay Buffer + Target Network
├── configs/
│   └── dqn_v1.yaml           # All hyperparameters
├── experiments/              # Training CSVs and evaluation plots (auto-created)
├── models/                   # Saved model weights (auto-created)
├── train.py                  # Train the DQN agent
├── evaluate.py               # Evaluate DQN vs Buy-and-Hold
├── plot_training.py          # Plot training curves
└── requirements.txt
```

---

## ⚙️ Installation

**1. Clone the repo**
```bash
git clone https://github.com/roshanth155/stock_mlops.git
cd stock_mlops
```

**2. Create a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Train the agent
```bash
python train.py --config configs/dqn_v1.yaml
```
Downloads AAPL data automatically, trains for 200 episodes, saves the best model to `models/policy_v1.pt`. Prints progress every 10 episodes.

Expected output:
```
Fetching AAPL from 2015-01-01 to 2022-12-31 ...
2014 trading days loaded.
Ep   10 | Reward: 151.24 | Portfolio: $34,690 | Sharpe: 0.960 | ε: 0.050
...
Training complete!
Best Sharpe : 1.3644
```

---

### Step 2 — Plot training curves
```bash
python plot_training.py
```
Auto-picks the latest CSV from `experiments/` and saves a `_curves.png` with 4 charts:
- Total reward per episode
- Sharpe ratio per episode
- Final portfolio value per episode
- Epsilon decay curve

---

### Step 3 — Evaluate DQN vs Buy-and-Hold
```bash
python evaluate.py --config configs/dqn_v1.yaml --model models/policy_v1.pt
```
Tests the trained policy on unseen 2023–2024 data, prints a metrics comparison table, and saves `experiments/eval_comparison.png`.

---

## 🧠 RL Design

| Component | Details |
|---|---|
| Algorithm | Double DQN with target network |
| State | `[norm_price, norm_portfolio, position, RSI, MA_ratio, volatility, cash_ratio]` — 7 features |
| Actions | `0=Hold, 1=Buy 10% cash, 2=Sell 10% holdings` |
| Reward | Sharpe proxy — step return / rolling std (last 20 steps) |
| Network | 4-layer MLP: 7 → 128 → 128 → 64 → 3 Q-values |
| Exploration | ε-greedy, exponential decay: 1.0 → 0.05 |
| Loss | Huber (Smooth L1) |
| Optimizer | Adam, lr=0.001, grad clip=1.0 |
| Memory | Replay buffer, capacity=10,000, batch=64 |
| Target update | Hard copy every 50 steps |

---

## ⚡ Hyperparameters

All hyperparameters are in `configs/dqn_v1.yaml`:

```yaml
ticker:         "AAPL"
train_start:    "2015-01-01"
train_end:      "2022-12-31"
eval_start:     "2023-01-01"
eval_end:       "2024-12-31"
initial_cash:   10000.0
window:         20
gamma:          0.99
lr:             0.001
batch_size:     64
buffer_size:    10000
hidden:         128
epsilon_start:  1.0
epsilon_min:    0.05
epsilon_decay:  0.995
episodes:       200
target_update_freq: 50
```

To run a new experiment, duplicate the config file and change the values:
```bash
cp configs/dqn_v1.yaml configs/dqn_v2.yaml
# edit dqn_v2.yaml
python train.py --config configs/dqn_v2.yaml
```

---

## 📦 Requirements

```
gymnasium>=0.29.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
yfinance>=0.2.28
pyyaml>=6.0
```

---

## 🔍 How It Works

Every trading day the agent goes through this loop:

```
1. OBSERVE  → reads 7 market features (price, RSI, volatility, etc.)
2. DECIDE   → picks Buy / Hold / Sell using ε-greedy policy
3. REWARD   → gets Sharpe-proxy reward (penalises volatility)
4. LEARN    → samples 64 memories, updates neural network weights
```

The key insight: the reward function punishes volatile gains and rewards steady ones — so the agent naturally learns risk management, not just profit chasing.

---

## 🌍 SDG 8 — Decent Work & Economic Growth

Professional risk-aware portfolio management is normally only available to hedge funds and wealthy investors. This project shows that open-source AI and free stock data can replicate intelligent investing — making it accessible to anyone. By reducing maximum drawdown by 38% compared to passive investing, the agent actively protects everyday investors from large losses.

---

## 📌 Experiment Tags

| Tag | Description |
|---|---|
| `exp-dqn-v1` | Baseline DQN, ε-decay=0.995, lr=0.001, Sharpe reward |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

**Roshanth** — B.E. Artificial Intelligence & Machine Learning  
BMS College of Engineering, Bengaluru  
Reinforcement Learning (24AM6PCREL) — AAT Project, Jan–May 2026
