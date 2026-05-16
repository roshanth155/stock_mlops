import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# ------------------------------------------------------------------
# Neural Network
# ------------------------------------------------------------------
class DQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------
# Replay Buffer
# ------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# ------------------------------------------------------------------
# DQN Agent
# ------------------------------------------------------------------
class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: dict):
        self.action_dim   = action_dim
        self.gamma        = cfg.get("gamma", 0.99)
        self.lr           = cfg.get("lr", 1e-3)
        self.batch_size   = cfg.get("batch_size", 64)
        self.epsilon      = cfg.get("epsilon_start", 1.0)
        self.epsilon_min  = cfg.get("epsilon_min", 0.05)
        self.epsilon_decay= cfg.get("epsilon_decay", 0.995)
        self.target_update= cfg.get("target_update_freq", 50)
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.buffer    = ReplayBuffer(cfg.get("buffer_size", 10_000))
        self.loss_fn   = nn.SmoothL1Loss()   # Huber loss — more stable
        self._steps    = 0

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.policy_net(s).argmax(dim=1).item())

    # ------------------------------------------------------------------
    def store(self, *args):
        self.buffer.push(*args)

    # ------------------------------------------------------------------
    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r  = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d  = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        # Current Q values
        q_vals = self.policy_net(s).gather(1, a)

        # Target Q values (Double DQN style)
        with torch.no_grad():
            best_actions = self.policy_net(ns).argmax(dim=1, keepdim=True)
            q_next       = self.target_net(ns).gather(1, best_actions)
            q_target     = r + self.gamma * q_next * (1 - d)

        loss = self.loss_fn(q_vals, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self._steps += 1
        if self._steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({
            "policy_state": self.policy_net.state_dict(),
            "epsilon":      self.epsilon,
        }, path)
        print(f"  [✓] Model saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_state"])
        self.target_net.load_state_dict(ckpt["policy_state"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_min)
        print(f"  [✓] Model loaded ← {path}")
