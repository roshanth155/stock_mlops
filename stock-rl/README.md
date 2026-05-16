# 📈 DQN-based Stock Portfolio Management for Risk-Adjusted Returns

> **SDG 8 – Decent Work and Economic Growth**
> Democratising intelligent, risk-aware investing through Reinforcement Learning.

A Deep Q-Network (DQN) agent that learns to **Buy / Hold / Sell** Apple (AAPL) stock by optimising for **risk-adjusted returns** (Sharpe Ratio) — not just raw profit. Trained on 8 years of historical data with full MLOps pipeline including MLflow tracking, GitOps branching, and CI/CD via GitHub Actions.

---

## 🏆 Training Results

| Metric | Value |
|---|---|
| Best Sharpe Ratio | **1.1267** |
| Best Portfolio Value | **$33,533** |
| Initial Cash | $10,000 |
| Episodes | 200 |
| Ticker | AAPL (2015–2022) |

---

## 📁 Project Structure

```
stock-rl/
├── sim/
│   └── stock_env.py           # Custom OpenAI Gym environment
├── agent/
│   └── dqn_agent.py           # DQN + Replay Buffer + Target Network
├── configs/
│   ├── dqn_v1.yaml            # Experiment 1: lr=0.001, buffer=10k
│   └── dqn_v2.yaml            # Experiment 2: lr=0.0005, buffer=20k
├── experiments/               # MLflow runs, CSVs, plots (auto-created)
├── models/                    # Saved model weights (auto-created)
├── .github/
│   └── workflows/
│       └── train.yml          # CI/CD pipeline (GitHub Actions)
├── train.py                   # Train DQN agent with MLflow tracking
├── evaluate.py                # DQN vs Buy-and-Hold comparison
├── plot_training.py           # Training curve visualisation
└── requirements.txt
```

---

## ⚙️ Installation

**1. Clone the repo**
```bash
git clone https://github.com/roshanth155/stock_mlops.git
cd stock_mlops/stock-rl
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

## 🚀 How to Reproduce

### Step 1 — Train the agent
```bash
python train.py --config configs/dqn_v1.yaml
```
Downloads AAPL data automatically, trains for 200 episodes, saves best model to `models/policy_v1.pt`. All metrics logged to MLflow automatically.

Expected output:
```
Fetching AAPL [2015-01-01 → 2022-12-31] ...
2014 trading days loaded.
MLflow Run ID : 7620654cb9a9406c8019a94f5080be6e
Ep   10 | Reward: 177.84 | Portfolio: $25,597 | Sharpe: 0.688 | ε: 0.050
...
✅ Training complete!
Best Sharpe : 1.1267
Best Portfolio : $33,533.76
👉 View results : mlflow ui
```

### Step 2 — View MLflow Dashboard
```bash
mlflow ui
# Open http://localhost:5000
```
Shows all runs, metrics per episode, hyperparameters, and saved models.

### Step 3 — Plot training curves
```bash
python plot_training.py
```
Generates 4 charts: total reward, Sharpe ratio, portfolio value, epsilon decay.

### Step 4 — Evaluate DQN vs Buy-and-Hold
```bash
python evaluate.py --config configs/dqn_v1.yaml --model models/policy_v1.pt
```
Tests on unseen 2023–2024 data and saves `experiments/eval_comparison.png`.

---

## 🧠 RL Design

| Component | Details |
|---|---|
| **Algorithm** | Double DQN with target network |
| **State (7 features)** | `norm_price, norm_portfolio, position, RSI, MA_ratio, volatility, cash_ratio` |
| **Actions** | `0=Hold, 1=Buy 10% cash, 2=Sell 10% holdings` |
| **Reward** | Sharpe proxy — step return / rolling std (last 20 steps) |
| **Network** | 4-layer MLP: 7 → 128 → 128 → 64 → 3 Q-values |
| **Exploration** | ε-greedy, exponential decay: 1.0 → 0.05 |
| **Loss** | Huber (Smooth L1) |
| **Optimizer** | Adam, lr=0.001, grad clip=1.0 |
| **Memory** | Replay buffer, capacity=10,000, batch=64 |
| **Target update** | Hard copy every 50 steps |
| **Why DQN?** | Continuous state space (RSI, volatility) makes tabular Q-learning impractical |

---

## ⚡ Hyperparameters

All hyperparameters in `configs/dqn_v1.yaml`:

```yaml
ticker:             "AAPL"
train_start:        "2015-01-01"
train_end:          "2022-12-31"
initial_cash:       10000.0
window:             20
gamma:              0.99
lr:                 0.001
batch_size:         64
buffer_size:        10000
epsilon_start:      1.0
epsilon_min:        0.05
epsilon_decay:      0.995
episodes:           200
target_update_freq: 50
```

To run a new experiment:
```bash
cp configs/dqn_v1.yaml configs/dqn_v2.yaml
# edit dqn_v2.yaml
python train.py --config configs/dqn_v2.yaml
```

---

## 🔧 MLOps Pipeline

### Experiment Tracking (MLflow)
- Logs all hyperparameters, per-episode metrics, and model artifacts
- View dashboard: `mlflow ui` → http://localhost:5000

### Version Control (Git + GitOps)
```
main         ← stable production branch
dev          ← development branch
exp/dqn-v1  ← experiment 1 (tag: exp-dqn-v1)
exp/dqn-v2  ← experiment 2 (tag: exp-dqn-v2)
```

### CI/CD (GitHub Actions)
- Triggers on push to `main` or `dev`
- Runs flake8 linting + smoke tests on environment and agent
- View: `.github/workflows/train.yml`

### Reproducibility
```bash
python train.py --config configs/dqn_v1.yaml
# Anyone can clone and reproduce exact results
```

---

## 📡 Monitoring Plan (Design Only)

If deployed in real-world trading:
- **Portfolio value drift** — real-time P&L tracking
- **Rolling Sharpe Ratio** — 30-day risk-adjusted performance
- **Max drawdown alert** — trigger if drawdown exceeds 15%
- **Action distribution** — detect over-buying or over-selling
- **Data drift** — monitor if live price distribution shifts from training
- **Retraining trigger** — retrain if Sharpe drops below 0.5 for 5 consecutive days

---

## 🔍 Results & Analysis

### When RL performs better
- In volatile training-period markets where adaptive decisions outperform fixed strategies
- After convergence (~episode 50+), Sharpe Ratio stabilises around 0.8–1.0
- Lower maximum drawdown compared to Buy-and-Hold during training

### When RL behaves badly
- **Distribution shift** — agent trained on 2015–2022 underperforms on unseen 2023–2024 data
- During sudden market crashes not seen in training data
- Early episodes with high ε — random exploration causes portfolio drops

---

## 🌍 SDG 8 — Decent Work & Economic Growth

> *"By improving risk-adjusted returns through intelligent automation, this system democratises access to portfolio management strategies previously only available to institutional investors — directly supporting SDG 8 by reducing financial risk and promoting financial inclusion."*

---

## ⚠️ Limitations & Future Work

**Limitations:**
- Single asset (AAPL only)
- No transaction costs modelled
- Requires periodic retraining for production use

**Future Work:**
- Multi-asset portfolio with Dueling DQN
- Transaction cost penalty in reward function
- Real-time data feed with automated retraining pipeline

---

## 📌 Experiment Tags

| Tag | Config | lr | Buffer | Notes |
|---|---|---|---|---|
| `exp-dqn-v1` | dqn_v1.yaml | 0.001 | 10,000 | Baseline |
| `exp-dqn-v2` | dqn_v2.yaml | 0.0005 | 20,000 | Slower decay, larger buffer |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

**Roshanth** — B.E. Artificial Intelligence & Machine Learning
BMS College of Engineering, Bengaluru
MLOps (24AM6AEMLO) — SEE Project, 2025–26
