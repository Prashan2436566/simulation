# mesa_sb3_env.py
# SB3/Gymnasium wrapper that controls one 'adaptive' agent in your Mesa world.

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from model.model import IdeologyModel
from model.agent import IdeologyAgent

# If your project uses a different action index for "mine", change this:
MINING_ACTION = 5

# Observation size your agent emits (via _state_from_obs)
OBS_DIM = 10


class MesaSB3Env(gym.Env):
    """
    SB3-compatible single-agent wrapper around IdeologyModel.

    - Action space: Discrete(7)
    - Observation space: R^{10} (float32) from agent._state_from_obs()
    - Reward shaping:
        +1.0 if alive else -3.0 on death
        +0.2 * (post_energy - pre_energy)
        +0.2 if action == MINING_ACTION
      then clipped to [-3, 3]
    - Episode ends on agent death (terminated=True) or when steps reach max_steps (truncated=True).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        width: int = 30,
        height: int = 30,
        num_agents: int = 15,
        renewables_regenerate: bool = True,
        ideology: str = "adaptive",
        max_steps: int = 1500,
        seed: Optional[int] = None,
    ):
        super().__init__()

        # Keep only args that IdeologyModel.__init__ actually supports.
        # DO NOT include 'seed' or 'max_steps' here.
        self._model_args = dict(
            width=width,
            height=height,
            num_agents=num_agents,
            renewables_regenerate=renewables_regenerate,
            ideology=ideology,
        )

        # Environment-only controls
        self.max_steps = int(max_steps)
        self._seed: Optional[int] = None
        self.seed(seed)

        # Spaces
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        # Runtime state
        self.model: Optional[IdeologyModel] = None
        self.agent_id: Optional[Any] = None
        self.steps: int = 0

    # -----------------------------
    # Gymnasium core API
    # -----------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed(seed)

        # (Re)create world without passing unsupported args
        self.model = IdeologyModel(**self._model_args)
        self.agent_id = self._find_controlled_agent_id()
        self.steps = 0

        obs = self._observe()
        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.model is not None, "Call reset() before step()."

        # Inject action for this tick (project-specific; adjust if needed)
        if hasattr(self.model, "external_action_idx"):
            self.model.external_action_idx = int(action)
        else:
            # Fallback: attach action to the agent if that's how your model reads it
            agent = self._get_agent()
            if agent is not None:
                setattr(agent, "external_action_idx", int(action))

        # Pre-step energy for shaping
        agent = self._get_agent()
        pre_energy = float(agent.energy) if agent is not None else 0.0

        # Advance world one tick
        if hasattr(self.model, "step"):
            self.model.step()
        else:
            raise AttributeError("IdeologyModel is missing a .step() method.")

        self.steps += 1

        # Post-step state
        agent = self._get_agent()
        alive = agent is not None
        post_energy = float(agent.energy) if agent is not None else 0.0

        terminated = not alive                      # died
        truncated = self.steps >= self.max_steps    # episode length cap

        # Reward shaping (clip for TD stability)
        reward = (1.0 if alive else -3.0)
        reward += 0.2 * (post_energy - pre_energy)
        if int(action) == MINING_ACTION:
            reward += 0.2
        reward = float(np.clip(reward, -3.0, 3.0))

        obs = self._observe() if alive else np.zeros(OBS_DIM, dtype=np.float32)
        info: Dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    # -----------------------------
    # Helpers
    # -----------------------------
    def _find_controlled_agent_id(self):
        """
        Prefer an agent with ideology == 'adaptive'; otherwise any IdeologyAgent.
        """
        if self.model is None or not hasattr(self.model, "schedule"):
            return None
        # Try to locate via model.schedule.agents
        try:
            agents = list(self.model.schedule.agents)
        except Exception:
            agents = []

        # Prefer adaptive
        for a in agents:
            if isinstance(a, IdeologyAgent) and getattr(a, "ideology", "") == "adaptive":
                return getattr(a, "unique_id", None)
        # Fallback: first IdeologyAgent
        for a in agents:
            if isinstance(a, IdeologyAgent):
                return getattr(a, "unique_id", None)
        return None

    def _get_agent(self) -> Optional[IdeologyAgent]:
        if self.model is None or not hasattr(self.model, "schedule"):
            return None
        try:
            for a in self.model.schedule.agents:
                if getattr(a, "unique_id", None) == self.agent_id:
                    return a
        except Exception:
            pass
        return None

    def _observe(self) -> np.ndarray:
        agent = self._get_agent()
        if agent is None:
            return np.zeros(OBS_DIM, dtype=np.float32)
        if hasattr(agent, "_state_from_obs"):
            obs = agent._state_from_obs()
        else:
            # Last resort if your agent uses a different accessor
            obs = getattr(agent, "get_observation", lambda: np.zeros(OBS_DIM))()
        return np.asarray(obs, dtype=np.float32).reshape(OBS_DIM,)

    # -----------------------------
    # Seeding / Render / Close
    # -----------------------------
    def seed(self, seed: Optional[int] = None):
        self._seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return [seed]

    def render(self):
        # training-only wrapper: no UI
        pass

    def close(self):
        self.model = None
