# sb3_shared_policy.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


OBS_DIM = 10
N_ACTIONS = 7     # idle, move_N, move_S, move_E, move_W, mine, repair


class _DummyObsEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = True
        truncated = False
        info: dict[str, Any] = {}
        return obs, reward, terminated, truncated, info


@dataclass
class SB3SharedPolicy:

    model_path: str
    vecnorm_path: Optional[str] = None
    device: str = "auto"
    deterministic: bool = True

    def __post_init__(self) -> None:
        base_env = DummyVecEnv([lambda: _DummyObsEnv()])

        if self.vecnorm_path is not None:
            self.vecnorm: VecNormalize = VecNormalize.load(
                self.vecnorm_path, base_env
            )
            self.vecnorm.training = False
            self.vecnorm.norm_reward = False
            env_for_model = self.vecnorm
        else:
            self.vecnorm = None
            env_for_model = base_env

        self.model: DQN = DQN.load(
            self.model_path,
            env=env_for_model,
            device=self.device,
        )

    def act(self, obs: np.ndarray) -> int:
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)

        if self.vecnorm is not None:
            obs_arr = self.vecnorm.normalize_obs(obs_arr)

        action, _ = self.model.predict(
            obs_arr,
            deterministic=self.deterministic,
        )

        if isinstance(action, np.ndarray):
            return int(action[0])
        return int(action)
