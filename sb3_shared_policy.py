import os
import numpy as np
from typing import Optional

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mesa_sb3_env import MesaSB3Env


class SB3SharedPolicy:

    def __init__(
        self,
        model_path: str = "models/dqn_sb3_final.zip",
        vecnorm_path: Optional[str] = "models/vecnorm.pkl",
        width: int = 15,
        height: int = 15,
        num_agents: int = 15,
        max_steps: int = 400,
        device: str = "cpu",
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SB3 model not found at: {model_path}")

        self.model = DQN.load(model_path, device=device)

        self.vecnorm = None
        if vecnorm_path is not None and os.path.exists(vecnorm_path):
            dummy_env = DummyVecEnv(
                [
                    lambda: MesaSB3Env(
                        width=width,
                        height=height,
                        num_agents=num_agents,
                        max_steps=max_steps,
                        seed=0,
                    )
                ]
            )
            self.vecnorm = VecNormalize.load(vecnorm_path, dummy_env)
            self.vecnorm.training = False
            self.vecnorm.norm_reward = False

    def _maybe_normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.vecnorm is None:
            return obs
        obs_batch = obs.reshape(1, -1)
        norm = self.vecnorm.normalize_obs(obs_batch)
        return norm[0]

    def act(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        obs = self._maybe_normalize_obs(obs)
        action, _ = self.model.predict(obs.reshape(1, -1), deterministic=True)
        return int(action)
