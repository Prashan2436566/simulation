from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from model.model import IdeologyModel
from model.agent import IdeologyAgent

MINING_ACTION = 5
OBS_DIM = 10
SHAPING_K_DIST = 0.1
IDLE_PENALTY = 0.05
DIST_CAP = 15
MINING_BONUS_K = 0.05
ENABLE_RESPAWN = True
...

DEBUG_TUNING = {
    "degrade_period": 20,
    "degrade_chance": 0.1,
    "renewable_cooldown_steps": 2,
    "renewable_overuse_trigger": 10,
    "renewable_fatigue_decay": 2,
}

def _dist_to_usable(model, pos):
    best = 10**9
    try:
        for locs, rtype in (
            (model.renewable_locations, "renewable"),
            (model.nonrenewable_locations, "nonrenewable"),
        ):
            for p in locs:
                try:
                    patch = next(
                        o
                        for o in model.grid.get_cell_list_contents([p])
                        if hasattr(o, "resource_type") and o.resource_type == rtype
                    )
                except StopIteration:
                    continue
                if getattr(patch, "amount", 0) <= 0:
                    continue
                if getattr(patch, "degraded", False) or getattr(
                    patch, "under_maintenance", False
                ):
                    continue
                d = abs(p[0] - pos[0]) + abs(p[1] - pos[1])
                if d < best:
                    best = d
    except Exception:
        return DIST_CAP
    return min(best, DIST_CAP)


def _count_usable(model):
    ren_cnt = 0
    non_cnt = 0
    try:
        for locs, rtype in (
            (model.renewable_locations, "renewable"),
            (model.nonrenewable_locations, "nonrenewable"),
        ):
            for p in locs:
                try:
                    patch = next(
                        o
                        for o in model.grid.get_cell_list_contents([p])
                        if hasattr(o, "resource_type") and o.resource_type == rtype
                    )
                except StopIteration:
                    continue
                if getattr(patch, "amount", 0) <= 0:
                    continue
                if getattr(patch, "degraded", False) or getattr(
                    patch, "under_maintenance", False
                ):
                    continue
                if rtype == "renewable":
                    ren_cnt += 1
                else:
                    non_cnt += 1
    except Exception:
        return 0, 0
    return ren_cnt, non_cnt


MAX_RESPAWNS = 0


class MesaSB3EnvCentralised(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        width=30,
        height=30,
        num_agents=15,
        renewables_regenerate=True,
        ideology="adaptive",
        max_steps=1500,
        seed=None,
    ):
        super().__init__()
        self._model_args = dict(
            width=width,
            height=height,
            num_agents=num_agents,
            renewables_regenerate=renewables_regenerate,
            ideology=ideology,
        )
        self.max_steps = max_steps
        self._seed = None
        self.seed(seed)
        self.n_controlled = num_agents

        # SB3 DQN only supports Discrete, so a single primitive action is chosen
        # and broadcast to all controlled agents.
        self.action_space = spaces.Discrete(7)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_controlled * OBS_DIM,),
            dtype=np.float32,
        )
        self.model = None
        self.controlled_ids: List[Optional[int]] = []
        self.steps = 0
        self.ep_mined_ren = 0.0
        self.ep_mined_non = 0.0

    def _get_controlled_agents(self):
        out = []
        for uid in self.controlled_ids:
            if uid is None:
                out.append(None)
            else:
                agent = next(
                    (a for a in self.model.schedule.agents if a.unique_id == uid),
                    None,
                )
                out.append(agent)
        return out

    def _observe_agent(self, agent: Optional[IdeologyAgent]) -> np.ndarray:
        if agent is None or agent.energy <= 0:
            return np.zeros(OBS_DIM, dtype=np.float32)
        try:
            pos = agent.pos
            cell = self.model.grid.get_cell_list_contents([pos])
            patch = next(
                (
                    o
                    for o in cell
                    if hasattr(o, "resource_type")
                    and o.resource_type in ("renewable", "nonrenewable")
                ),
                None,
            )
        except Exception:
            patch = None

        energy = float(agent.energy)
        tile_type = 0.0
        amount = 0.0
        scar = 0.0
        degraded_flag = 0.0
        hub_flag = 0.0

        if patch is not None:
            amount = float(getattr(patch, "amount", 0.0))
            scar = float(getattr(patch, "scar_level", 0.0))
            if getattr(patch, "resource_type", "") == "renewable":
                tile_type = 1.0
            elif getattr(patch, "resource_type", "") == "nonrenewable":
                tile_type = 2.0
            if getattr(patch, "degraded", False) or getattr(
                patch, "under_maintenance", False
            ):
                degraded_flag = 1.0

        try:
            for o in cell:
                if hasattr(o, "is_hub") and getattr(o, "is_hub"):
                    hub_flag = 1.0
                    break
        except Exception:
            pass

        dist_ren = float(_dist_to_usable(self.model, agent.pos))
        dist_non = float(_dist_to_usable(self.model, agent.pos))
        net_yield = 0.0

        try:
            ren_left = len(self.model.renewable_locations)
        except Exception:
            ren_left = 0
        try:
            non_left = len(self.model.nonrenewable_locations)
        except Exception:
            non_left = 0

        scarcity = float(non_left)

        return np.array(
            [
                energy,
                tile_type,
                amount,
                scar,
                degraded_flag,
                hub_flag,
                dist_ren,
                dist_non,
                net_yield,
                scarcity,
            ],
            dtype=np.float32,
        )

    def _observe_all(self) -> np.ndarray:
        agents = self._get_controlled_agents()
        obs_list = [self._observe_agent(a) for a in agents]
        return np.concatenate(obs_list, axis=0)

    def reset(self, *, seed=None, options=None):
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
                self.model.max_adaptive_respawns = MAX_RESPAWNS
            except Exception:
                pass
        self.controlled_ids = self._find_controlled_agent_ids()
        if len(self.controlled_ids) < self.n_controlled:
            self.controlled_ids += [None] * (
                self.n_controlled - len(self.controlled_ids)
            )
        self.steps = 0
        self.ep_mined_ren = 0.0
        self.ep_mined_non = 0.0
        setattr(self.model, "external_action_map", {})
        obs = self._observe_all()
        return obs, {}

    def step(self, action):
        # single scalar action from SB3, broadcast to all controlled agents
        a_scalar = int(np.asarray(action, dtype=np.int64).item())
        action_map: Dict[int, int] = {}
        for uid in self.controlled_ids:
            if uid is not None:
                action_map[uid] = a_scalar
        setattr(self.model, "external_action_map", action_map)

        agents = self._get_controlled_agents()
        alive_mask = [a is not None for a in agents]
        n_alive = sum(alive_mask)
        if n_alive > 0:
            pre_energy = np.mean(
                [float(a.energy) for a in agents if a is not None]
            )
            pre_dist_mean = np.mean(
                [_dist_to_usable(self.model, a.pos) for a in agents if a is not None]
            )
        else:
            pre_energy = 0.0
            pre_dist_mean = DIST_CAP

        self.model.step()
        self.steps += 1

        ren_step = float(getattr(self.model, "_mined_renewable_this_step", 0.0))
        non_step = float(getattr(self.model, "_mined_nonrenewable_this_step", 0.0))
        self.ep_mined_ren += ren_step
        self.ep_mined_non += non_step
        if hasattr(self.model, "_mined_renewable_this_step"):
            self.model._mined_renewable_this_step = 0.0
        if hasattr(self.model, "_mined_nonrenewable_this_step"):
            self.model._mined_nonrenewable_this_step = 0.0

        agents = self._get_controlled_agents()
        alive_mask = [a is not None for a in agents]
        n_alive = sum(alive_mask)
        if n_alive > 0:
            post_energy = np.mean(
                [float(a.energy) for a in agents if a is not None]
            )
            post_dist_mean = np.mean(
                [_dist_to_usable(self.model, a.pos) for a in agents if a is not None]
            )
        else:
            post_energy = 0.0
            post_dist_mean = DIST_CAP

        terminated = n_alive == 0
        truncated = self.steps >= self.max_steps

        reward = 1.0 if n_alive > 0 else -3.0
        reward += 0.2 * (post_energy - pre_energy)
        reward += 0.2 * (1.0 if a_scalar == MINING_ACTION else 0.0)
        reward += SHAPING_K_DIST * (pre_dist_mean - post_dist_mean)
        reward -= IDLE_PENALTY * (1.0 if a_scalar == 0 else 0.0)
        reward += MINING_BONUS_K * (ren_step + non_step)
        reward = float(np.clip(reward, -3.0, 3.0))

        obs = (
            self._observe_all()
            if n_alive > 0
            else np.zeros(self.n_controlled * OBS_DIM, dtype=np.float32)
        )
        usable_ren_cnt, usable_non_cnt = _count_usable(self.model)

        info = {
            "step": self.steps,
            "step_mined_renewable": ren_step,
            "step_mined_nonrenewable": non_step,
            "nearest_usable_dist_mean": post_dist_mean,
            "usable_counts": (usable_ren_cnt, usable_non_cnt),
            "alive_agents": n_alive,
        }

        if terminated or truncated:
            info.update(
                {
                    "ep_mined_renewable": self.ep_mined_ren,
                    "ep_mined_nonrenewable": self.ep_mined_non,
                    "ep_mined_total": self.ep_mined_ren + self.ep_mined_non,
                    "done_reason": "death" if terminated else "time",
                }
            )

        return obs, reward, terminated, truncated, info

    def _find_controlled_agent_ids(self):
        agents = list(self.model.schedule.agents)
        adaptive_ids = []
        for a in agents:
            if isinstance(a, IdeologyAgent) and getattr(a, "ideology", "") in (
                "adaptive",
                "adaptive_direct",
                "adaptive_dqn",
            ):
                adaptive_ids.append(a.unique_id)
        adaptive_ids = sorted(adaptive_ids)
        return adaptive_ids[: self.n_controlled]

    def seed(self, seed=None):
        self._seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return [seed]

    def render(self):
        pass

    def close(self):
        self.model = None
