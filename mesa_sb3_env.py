from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from model.model import IdeologyModel
from model.agent import IdeologyAgent


MINING_ACTION = 5

OBS_DIM = 10


SHAPING_K_DIST = 0.1   
IDLE_PENALTY   = 0.05  
DIST_CAP       = 15     
MINING_BONUS_K = 0.05    


ENABLE_RESPAWN = True
MAX_RESPAWNS   = 3

DEBUG_TUNING: Dict[str, Any] = {

    "degrade_period": 20,         
    "degrade_chance": 0.1,        

    "renewable_cooldown_steps": 2,  
    "renewable_overuse_trigger": 10,
    "renewable_fatigue_decay": 2,


}

# -------- Helpers --------
def _dist_to_usable(model: IdeologyModel, pos) -> int:

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


def _count_usable(model: IdeologyModel) -> Tuple[int, int]:
    ren = non = 0
    try:
        for p in list(getattr(model, "renewable_locations", [])):
            cell = model.grid.get_cell_list_contents([p])
            patch = next((o for o in cell if getattr(o, "resource_type", "") == "renewable"), None)
            if patch and getattr(patch, "amount", 0) > 0 and not getattr(patch, "degraded", False):
                ren += 1
        for p in list(getattr(model, "nonrenewable_locations", [])):
            cell = model.grid.get_cell_list_contents([p])
            patch = next((o for o in cell if getattr(o, "resource_type", "") == "nonrenewable"), None)
            if patch and getattr(patch, "amount", 0) > 0:
                non += 1
    except Exception:
        pass
    return ren, non


class MesaSB3Env(gym.Env):

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

        self._model_args = dict(
            width=width,
            height=height,
            num_agents=num_agents,
            renewables_regenerate=renewables_regenerate,
            ideology=ideology,
        )

        self.max_steps = int(max_steps)
        self._seed: Optional[int] = None
        self.seed(seed)

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self.model: Optional[IdeologyModel] = None
        self.agent_id: Optional[Any] = None
        self.steps: int = 0

        self.ep_mined_ren: float = 0.0
        self.ep_mined_non: float = 0.0


    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed(seed)


        self.model = IdeologyModel(**self._model_args)


        for k, v in DEBUG_TUNING.items():
            if hasattr(self.model, k):
                try:
                    setattr(self.model, k, v)
                except Exception:
                    pass

        if ENABLE_RESPAWN and hasattr(self.model, "max_adaptive_respawns"):
            try:
                self.model.max_adaptive_respawns = int(MAX_RESPAWNS)
            except Exception:
                pass

        self.agent_id = self._find_controlled_agent_id()
        self.steps = 0

        self.ep_mined_ren = 0.0
        self.ep_mined_non = 0.0
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


        if hasattr(self.model, "external_action_idx"):
            self.model.external_action_idx = int(action)
        else:
            agent = self._get_agent()
            if agent is not None:
                setattr(agent, "external_action_idx", int(action))


        agent = self._get_agent()
        pre_energy = float(agent.energy) if agent is not None else 0.0
        pre_dist = _dist_to_usable(self.model, getattr(agent, "pos", (0, 0))) if agent is not None else DIST_CAP

        if hasattr(self.model, "step"):
            self.model.step()
        else:
            raise AttributeError("IdeologyModel is missing a .step() method.")

        self.steps += 1


        ren_step = float(getattr(self.model, "_mined_renewable_this_step", 0.0))
        non_step = float(getattr(self.model, "_mined_nonrenewable_this_step", 0.0))
        self.ep_mined_ren += ren_step
        self.ep_mined_non += non_step

        if hasattr(self.model, "_mined_renewable_this_step"):
            self.model._mined_renewable_this_step = 0.0
        if hasattr(self.model, "_mined_nonrenewable_this_step"):
            self.model._mined_nonrenewable_this_step = 0.0

        
        agent = self._get_agent()
        alive = agent is not None
        post_energy = float(agent.energy) if agent is not None else 0.0
        post_dist = _dist_to_usable(self.model, getattr(agent, "pos", (0, 0))) if alive else DIST_CAP


        if not alive and ENABLE_RESPAWN and hasattr(self.model, "max_adaptive_respawns") and self.model.max_adaptive_respawns > 0:
            new_id = self._find_controlled_agent_id()
            if new_id is not None and new_id != self.agent_id:
                self.agent_id = new_id
                agent = self._get_agent()
                if agent is not None:
                    alive = True
                    post_energy = float(agent.energy)
                    post_dist = _dist_to_usable(self.model, getattr(agent, "pos", (0, 0)))

        terminated = not alive                   
        truncated = self.steps >= self.max_steps    

        reward = (1.0 if alive else -3.0)
        reward += 0.2 * (post_energy - pre_energy)
        if int(action) == MINING_ACTION:
            reward += 0.2

        reward += SHAPING_K_DIST * (pre_dist - post_dist)

        if int(action) == 0:
            reward -= IDLE_PENALTY

        reward += MINING_BONUS_K * (ren_step + non_step)

        # Clip 
        reward = float(np.clip(reward, -3.0, 3.0))

        obs = self._observe() if alive else np.zeros(OBS_DIM, dtype=np.float32)

        on_type, on_amount, on_degraded = None, 0.0, False
        if alive:
            cell = self.model.grid.get_cell_list_contents([getattr(agent, "pos", (0, 0))])
            patch = next((o for o in cell if getattr(o, "resource_type", None) in ("renewable", "nonrenewable")), None)
            if patch is not None:
                on_type = getattr(patch, "resource_type", None)
                on_amount = float(getattr(patch, "amount", 0.0))
                on_degraded = bool(getattr(patch, "degraded", False))
        usable_ren_cnt, usable_non_cnt = _count_usable(self.model)

        info: Dict[str, Any] = {
            "step": self.steps,
            "step_mined_renewable": ren_step,
            "step_mined_nonrenewable": non_step,
            "nearest_usable_dist": post_dist,
            "on_patch_type": on_type,
            "on_patch_amount": on_amount,
            "on_patch_degraded": on_degraded,
            "usable_counts": (usable_ren_cnt, usable_non_cnt),
            "agent_energy": float(post_energy),
        }

        if terminated or truncated:
            info.update({
                "ep_mined_renewable": self.ep_mined_ren,
                "ep_mined_nonrenewable": self.ep_mined_non,
                "ep_mined_total": self.ep_mined_ren + self.ep_mined_non,
                "done_reason": "death" if terminated else "time",
            })

        return obs, reward, terminated, truncated, info

    def _find_controlled_agent_id(self):
        if self.model is None or not hasattr(self.model, "schedule"):
            return None
        try:
            agents = list(self.model.schedule.agents)
        except Exception:
            agents = []

        for a in agents:
            if isinstance(a, IdeologyAgent) and getattr(a, "ideology", "") == "adaptive":
                return getattr(a, "unique_id", None)
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


    def seed(self, seed: Optional[int] = None):
        self._seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return [seed]

    def render(self):
        pass

    def close(self):
        self.model = None
