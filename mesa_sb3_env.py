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

# Action mapping used by your adaptive DQN policy in agent.py:
# ["idle","move_N","move_S","move_E","move_W","mine","repair"]
MINING_ACTION = 5

# Observation size your agent emits (via _state_from_obs)
OBS_DIM = 10

# ---- Potential-based shaping knobs ----
SHAPING_K_DIST = 0.1     # weight for distance potential ΔΦ
IDLE_PENALTY   = 0.05    # tiny penalty for 'idle' (action 0)
DIST_CAP       = 15      # cap distance so ΔΦ isn't extreme


# -------- Helper: distance to nearest usable energy patch --------
def _dist_to_usable(model: IdeologyModel, pos) -> int:
    """
    Manhattan distance to the nearest 'usable' patch:
    - renewable or nonrenewable
    - amount > 0
    - for renewables: not degraded/under_maintenance
    Returns a capped distance (DIST_CAP) if nothing suitable is found.
    """
    best = 10**9
    try:
        for locs, rtype in ((model.renewable_locations, "renewable"),
                            (model.nonrenewable_locations, "nonrenewable")):
            for p in list(locs):
                cell = model.grid.get_cell_list_contents([p])
                patch = next(
                    (o for o in cell
                     if getattr(o, "resource_type", None) == rtype and getattr(o, "amount", 0) > 0),
                    None
                )
                if not patch:
                    continue
                if rtype == "renewable" and (
                    getattr(patch, "degraded", False)
                    or getattr(patch, "is_degraded", False)
                    or getattr(patch, "under_maintenance", False)
                ):
                    continue
                d = abs(pos[0] - p[0]) + abs(pos[1] - p[1])
                if d < best:
                    best = d
    except Exception:
        return DIST_CAP
    return best if best < 10**9 else DIST_CAP


class MesaSB3Env(gym.Env):
    """
    SB3-compatible single-agent wrapper around IdeologyModel.

    - Action space: Discrete(7)
    - Observation space: R^{10} (float32) from agent._state_from_obs()
    - Reward shaping:
        base:
          +1.0 if alive else -3.0
          +0.2 * (post_energy - pre_energy)
          +0.2 if action == MINING_ACTION
        added (potential-based):
          + SHAPING_K_DIST * (pre_dist - post_dist)
        tiny anti-idle:
          - IDLE_PENALTY if action == 0 ("idle")
      then clipped to [-3, 3]
    - Episode ends on agent death (terminated=True) or when steps reach max_steps (truncated=True).

    Additionally:
      • Tracks per-episode mining totals in `info` at episode end:
          - ep_mined_renewable, ep_mined_nonrenewable, ep_mined_total
      • Emits per-step telemetry in `info` every step:
          - step, step_mined_renewable, step_mined_nonrenewable
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

        # Episode mining accumulators
        self.ep_mined_ren: float = 0.0
        self.ep_mined_non: float = 0.0

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

        # Reset episode mining totals
        self.ep_mined_ren = 0.0
        self.ep_mined_non = 0.0
        # Zero per-step counters on the model if present (avoid double counting)
        if hasattr(self.model, "_mined_renewable_this_step"):
            self.model._mined_renewable_this_step = 0.0
        if hasattr(self.model, "_mined_nonrenewable_this_step"):
            self.model._mined_nonrenewable_this_step = 0.0

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
            agent = self._get_agent()
            if agent is not None:
                setattr(agent, "external_action_idx", int(action))

        # Pre-step: get controlled agent and compute pre metrics
        agent = self._get_agent()
        pre_energy = float(agent.energy) if agent is not None else 0.0
        pre_dist = _dist_to_usable(self.model, getattr(agent, "pos", (0, 0))) if agent is not None else DIST_CAP

        # Advance world one tick
        if hasattr(self.model, "step"):
            self.model.step()
        else:
            raise AttributeError("IdeologyModel is missing a .step() method.")

        self.steps += 1

        # Accumulate per-step mined amounts from the model
        ren_step = float(getattr(self.model, "_mined_renewable_this_step", 0.0))
        non_step = float(getattr(self.model, "_mined_nonrenewable_this_step", 0.0))
        self.ep_mined_ren += ren_step
        self.ep_mined_non += non_step
        # Reset step counters so we don't double count next tick
        if hasattr(self.model, "_mined_renewable_this_step"):
            self.model._mined_renewable_this_step = 0.0
        if hasattr(self.model, "_mined_nonrenewable_this_step"):
            self.model._mined_nonrenewable_this_step = 0.0

        # Post-step state
        agent = self._get_agent()
        alive = agent is not None
        post_energy = float(agent.energy) if agent is not None else 0.0
        post_dist = _dist_to_usable(self.model, getattr(agent, "pos", (0, 0))) if alive else DIST_CAP

        terminated = not alive                      # died
        truncated = self.steps >= self.max_steps    # episode length cap

        # -------- Base reward (existing) --------
        reward = (1.0 if alive else -3.0)
        reward += 0.2 * (post_energy - pre_energy)
        if int(action) == MINING_ACTION:
            reward += 0.2

        # -------- Added potential-based shaping --------
        # Positive if the agent moved closer to a usable patch.
        reward += SHAPING_K_DIST * (pre_dist - post_dist)

        # -------- Tiny anti-idle penalty (assuming action 0 == "idle") --------
        if int(action) == 0:
            reward -= IDLE_PENALTY

        # Clip for TD stability
        reward = float(np.clip(reward, -3.0, 3.0))

        obs = self._observe() if alive else np.zeros(OBS_DIM, dtype=np.float32)

        # Build info dict with per-step telemetry
        info: Dict[str, Any] = {
            "step": self.steps,
            "step_mined_renewable": ren_step,
            "step_mined_nonrenewable": non_step,
        }

        # Include episode mining totals in info on episode end
        if terminated or truncated:
            info.update({
                "ep_mined_renewable": self.ep_mined_ren,
                "ep_mined_nonrenewable": self.ep_mined_non,
                "ep_mined_total": self.ep_mined_ren + self.ep_mined_non,
            })

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
