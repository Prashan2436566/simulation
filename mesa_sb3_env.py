# mesa_sb3_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from model.model import IdeologyModel
from model.agent import IdeologyAgent

class MesaSB3Env(gym.Env):
    """
    Single-agent wrapper: SB3 controls exactly one 'adaptive' agent.
    Each step() injects the chosen action, then advances the whole Mesa world by 1 tick.
    """
    metadata = {"render.modes": []}

    def __init__(self,
                 width=30, height=30, num_agents=15,
                 renewables_regenerate=True,
                 ideology="adaptive",
                 max_steps=1000,
                 **model_kwargs):
        super().__init__()
        self._model_args = dict(width=width, height=height, num_agents=num_agents,
                                renewables_regenerate=renewables_regenerate,
                                ideology=ideology, **model_kwargs)
        self.model = IdeologyModel(**self._model_args)
        self.max_steps = int(max_steps)
        self.steps = 0
        self.agent_id = self._find_controlled_agent_id()

        # 7 discrete actions: ["idle","move_N","move_S","move_E","move_W","mine","repair"]
        self.action_space = spaces.Discrete(7)

        # Observation is your 10-dim compact tuple from _state_from_obs()
        obs0 = self._observe()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(len(obs0),), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.model = IdeologyModel(**self._model_args)
        self.agent_id = self._find_controlled_agent_id()
        self.steps = 0
        return np.asarray(self._observe(), dtype=np.float32), {}

    def step(self, action):
        # Provide the chosen action to the controlled agent for this tick
        self.model.external_action_idx = int(action)

        agent = self._get_agent()
        pre_e = agent.energy if agent is not None else 0.0

        # Advance world by one scheduler tick (RandomActivation) and bookkeeping
        # (Your model uses Mesa's scheduler; model.step() triggers one tick.)
        self.model.step()  # one global tick (agents + patches, datacollection, etc.)

        self.steps += 1
        agent = self._get_agent()  # may be None if agent died this step
        obs = np.asarray(self._observe(), dtype=np.float32)

        # Termination & truncation
        terminated = agent is None
        truncated = self.steps >= self.max_steps

        # Survival-first reward shaped by energy delta (+ small mine bias if desired)
        reward = 0.0
        if agent is not None:
            post_e = agent.energy
            reward += 1.0 if post_e > 0 else -3.0
            reward += 0.2 * (post_e - pre_e)
            # (optional bonus if action==mine)
            if int(action) == 5:  # index of "mine"
                reward += 0.2
        else:
            reward = -3.0

        return obs, float(reward), bool(terminated), bool(truncated), {}

    # ---------- helpers ----------
    def _find_controlled_agent_id(self):
        for a in self.model.schedule.agents:
            if isinstance(a, IdeologyAgent) and getattr(a, "ideology", "") == "adaptive":
                return a.unique_id
        # fallback to any energy-bearing agent if no adaptive present
        for a in self.model.schedule.agents:
            if hasattr(a, "energy"):
                return a.unique_id
        return None

    def _get_agent(self):
        for a in self.model.schedule.agents:
            if getattr(a, "unique_id", None) == self.agent_id:
                return a
        return None

    def _observe(self):
        a = self._get_agent()
        if a is None:
            return np.zeros(10, dtype=np.float32)
        return np.asarray(a._state_from_obs(), dtype=np.float32)
