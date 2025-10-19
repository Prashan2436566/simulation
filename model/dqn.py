# dqn.py
# Lightweight DQN for your adaptive agent (vector observations).
# Exposes: DQNConfig, DQN with act/push/learn/save APIs.

from dataclasses import dataclass
from typing import Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class DQNConfig:
    state_dim: int
    n_actions: int
    lr: float = 1e-3
    gamma: float = 0.95
    batch_size: int = 64
    buffer_size: int = 100_000
    start_learning: int = 1_000        # warmup transitions before learning
    target_update_interval: int = 1_000
    train_interval: int = 1            # learn every N env steps
    grad_clip: float = 1.0
    double_q: bool = True              # Double DQN
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "dqn_adaptive.pt" # checkpoint path


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity,), dtype=np.int64)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.not_done = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, s, a, r, s2, done):
        i = self.ptr
        self.state[i] = s
        self.action[i] = a
        self.reward[i] = r
        self.next_state[i] = s2
        self.not_done[i] = 0.0 if done else 1.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.state[idx]),
            torch.tensor(self.action[idx]),
            torch.tensor(self.reward[idx]),
            torch.tensor(self.next_state[idx]),
            torch.tensor(self.not_done[idx]),
        )


class DQN:
    """
    Minimal DQN with:
      - MLP Q-network + target net
      - Replay buffer
      - Double DQN targets (optional)
      - Periodic hard target update
    API:
      act(state_vec: np.ndarray, epsilon: float) -> int
      push(s, a, r, s2, done) -> None
      learn(global_step: Optional[int]) -> Optional[float]
      save() -> None
    """
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.q = QNetwork(cfg.state_dim, cfg.n_actions).to(self.device)
        self.target = QNetwork(cfg.state_dim, cfg.n_actions).to(self.device)
        self.target.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.buffer_size, cfg.state_dim)
        self.global_step = 0

        # Try to resume
        if os.path.exists(cfg.save_path):
            try:
                payload = torch.load(cfg.save_path, map_location=self.device)
                self.q.load_state_dict(payload["q"])
                self.target.load_state_dict(payload["target"])
                self.opt.load_state_dict(payload["opt"])
                if "step" in payload:
                    self.global_step = int(payload["step"])
                print(f"[DQN] Loaded weights from {cfg.save_path}")
            except Exception as e:
                print(f"[DQN] Failed to load {cfg.save_path}: {e}")

    def save(self):
        torch.save({
            "q": self.q.state_dict(),
            "target": self.target.state_dict(),
            "opt": self.opt.state_dict(),
            "step": self.global_step,
        }, self.cfg.save_path)

    @torch.no_grad()
    def act(self, state_vec: np.ndarray, epsilon: float) -> int:
        # state_vec shape: (state_dim,)
        if np.random.rand() < epsilon:
            return np.random.randint(self.cfg.n_actions)
        x = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        qv = self.q(x)  # (1, n_actions)
        return int(qv.argmax(dim=1).item())

    def push(self, s, a, r, s2, done):
        self.replay.push(s, a, r, s2, done)

    def _loss(self, batch) -> torch.Tensor:
        s, a, r, s2, not_done = [t.to(self.device) for t in batch]

        # Q(s,a)
        q_sa = self.q(s).gather(1, a.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_q:
                # action selection under online net
                a_max = self.q(s2).argmax(dim=1, keepdim=True)
                # action evaluation under target net
                q_next = self.target(s2).gather(1, a_max).squeeze(1)
            else:
                q_next = self.target(s2).max(dim=1).values

            target = r + not_done * self.cfg.gamma * q_next

        return nn.functional.smooth_l1_loss(q_sa, target)

    def learn(self, global_step: Optional[int] = None) -> Optional[float]:
        # Update internal step
        self.global_step = int(global_step if global_step is not None else self.global_step + 1)

        if self.replay.size < self.cfg.start_learning:
            return None
        if (self.global_step % self.cfg.train_interval) != 0:
            return None

        batch = self.replay.sample(self.cfg.batch_size)
        loss = self._loss(batch)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()

        if (self.global_step % self.cfg.target_update_interval) == 0:
            self.target.load_state_dict(self.q.state_dict())

        return float(loss.item())
