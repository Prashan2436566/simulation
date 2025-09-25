from mesa import Agent
import math
import random


# =========================
# Resource / Structure
# =========================
class ResourcePatch(Agent):
    """Grid patch holding either renewable or nonrenewable resources."""
    def __init__(
        self,
        unique_id: int,
        model,
        resource_type: str,
        max_capacity: int = 5,
        regen_rate: int = 1,
    ) -> None:
        super().__init__(unique_id, model)
        self.resource_type = resource_type  # "renewable" or "nonrenewable"
        self.amount = max_capacity
        self.max_capacity = max_capacity
        self.base_regen_rate = regen_rate if resource_type == "renewable" else 0
        self.regen_rate = self.base_regen_rate

        # Overuse dynamics
        self.cooldown_remaining = 0
        self.fatigue = 0

        # Environment
        self.scar_level: float = 0.0

        # Degradation (offline until repaired)
        self.is_degraded: bool = False
        # compatibility flags for your current CanvasGrid portrayal
        self.degraded: bool = False
        self.under_maintenance: bool = False  # not actively used here, but kept for UI

    # ----- degrade helpers -----
    def mark_degraded(self):
        self.is_degraded = True
        self.degraded = True

    def clear_degraded(self):
        self.is_degraded = False
        self.degraded = False
        self.under_maintenance = False

    def step(self) -> None:
        # decay scars
        if self.scar_level > 0:
            self.scar_level = max(0.0, self.scar_level - self.model.scar_decay)

        # collapse if scar too high (renewables only)
        if self.resource_type == "renewable" and self.scar_level >= self.model.scar_collapse_threshold:
            try:
                self.model.grid.remove_agent(self)
            except Exception:
                pass
            try:
                self.model.schedule.remove(self)
            except Exception:
                pass
            try:
                if self.pos in self.model.renewable_locations:
                    self.model.renewable_locations.remove(self.pos)
            except Exception:
                pass
            return

        # renewable regen logic
        if self.resource_type == "renewable":
            if self.fatigue > 0:
                self.fatigue = max(0, self.fatigue - self.model.renewable_fatigue_decay)

            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
                self.regen_rate = 0
            else:
                penalty = max(0.0, 1.0 - self.model.scar_regen_alpha * self.scar_level)
                self.regen_rate = self.base_regen_rate * penalty

        # apply regen unless degraded (degraded = offline)
        if self.regen_rate > 0 and not self.is_degraded:
            self.amount = min(self.max_capacity, self.amount + self.regen_rate)

    def harvest(self, amount: int) -> int:
        # if degraded renewable: yield zero
        if self.resource_type == "renewable" and self.is_degraded:
            return 0

        collected = min(self.amount, amount)
        self.amount -= collected
        self.model.total_mined_energy += collected

        if collected > 0:
            if self.resource_type == "renewable":
                self.model._mined_renewable_this_step += collected
            else:
                self.model._mined_nonrenewable_this_step += collected

        # Renewable fatigue/cooldown
        if self.resource_type == "renewable" and collected > 0:
            self.fatigue += collected
            if self.amount <= 0 or self.fatigue >= self.model.renewable_overuse_trigger:
                self.cooldown_remaining = max(self.cooldown_remaining, self.model.renewable_cooldown_steps)

        # Nonrenewable spill -> scar nearby renewables
        if self.resource_type == "nonrenewable" and collected > 0:
            radius = self.model.scar_radius
            bump = self.model.scar_increase_per_unit * collected
            neighs = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True, radius=radius)
            for p in neighs:
                for obj in self.model.grid.get_cell_list_contents([p]):
                    if isinstance(obj, ResourcePatch) and obj.resource_type == "renewable":
                        obj.scar_level = min(self.model.scar_max, obj.scar_level + bump)

        return collected


class EnergyHub(Agent):
    """Built on a renewable patch by at least 2 socialist agents. Improves mining efficiency."""
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos
        self.built = True

    def step(self):
        pass  # passive structure


# =========================
# Agents
# =========================
class IdeologyAgent(Agent):
    def __init__(self, unique_id: int, model, ideology: str) -> None:
        super().__init__(unique_id, model)
        self.ideology = ideology
        self.energy = 10.0
        self.total_collected_energy = 0.0
        self.mining = False
        self.mining_counter = 0
        self.mining_target: ResourcePatch | None = None
        self.model.total_agents_created += 1
        self.renewable_setup_paid: set[int] = set()

        # Socialist-ish tuning
        self.share_radius = 1
        self.share_fraction = 0.60
        self.min_keep = 8.0
        self.help_threshold = 7.0
        self.renewable_bias = 3
        self.energy_cap = 12.0
        self.emergency_floor = 4.0
        self.coop_build_time = 2
        self.build_counter = 0
        self.intent_build = False

        # --- RL (Adaptive) setup ---
        if ideology == "adaptive":
            # survival + faster feedback
            self.energy = 20.0            # stronger start
            self.emergency_floor = 6.0    # earlier emergency
            self.adaptive_mine_ticks = 3  # shorter bursts => quicker rewards

            # RL knobs
            self.rl_alpha = 0.2
            self.rl_gamma = 0.95
            self.rl_epsilon = 0.9
            self.rl_epsilon_min = 0.05
            self.rl_epsilon_decay = 0.995

            # tabular Q: dict[state] -> dict[action] -> value
            self.q_table: dict[tuple, dict[str, float]] = {}

            # available discrete actions
            self.RL_ACTIONS = [
                "idle", "move_N", "move_S", "move_E", "move_W",
                "mine", "repair"
            ]
            self._last_state = None
            self._last_action = None

            # --- respawn-with-memory bookkeeping ---
            self.episodes = 0
            self.max_adaptive_respawns = getattr(self.model, "max_adaptive_respawns", float("inf"))
            self.q_init = getattr(self.model, "q_init", 15.0)

            # (optional) terminal-only Monte-Carlo credit at death; off by default
            self.use_mc_terminal_updates = False
            self.episode_traj = []  # list of (state, action) tuples for MC updates

            # Neutralize built-in renewable bias for adaptive (unless user sets it)
            self.renewable_bias = getattr(self.model, "adaptive_renewable_bias", 0)

            # For SMDP/option credit during mining
            self._mine_k = 0
            self._mine_accum_r = 0.0

        elif ideology == "adaptive_direct":
            # survival-focused, but with a simplified action space
            self.energy = 20.0
            self.emergency_floor = 6.0
            self.adaptive_mine_ticks = 3

            # RL knobs (reuse same defaults)
            self.rl_alpha = 0.2
            self.rl_gamma = 0.95
            self.rl_epsilon = 0.3
            self.rl_epsilon_min = 0.05
            self.rl_epsilon_decay = 0.995

            # tabular Q + limited action set
            self.q_table: dict[tuple, dict[str, float]] = {}
            self.RL_ACTIONS = [
                "goto_nearest_non",
                "goto_nearest_ren",
                "mine",
                "repair",
            ]
            self._last_state = None
            self._last_action = None

            # respawn-with-memory options (mirror adaptive)
            self.episodes = 0
            self.max_adaptive_respawns = getattr(self.model, "max_adaptive_respawns", float("inf"))
            self.q_init = getattr(self.model, "q_init", 15.0)

            self.use_mc_terminal_updates = False
            self.episode_traj = []

            # no built-in renewable bias unless you set it in the model
            self.renewable_bias = getattr(self.model, "adaptive_renewable_bias", 0)


    # ---------- COMMON PER-TICK ----------
    def step(self) -> None:
        if self.ideology == "socialist":
            self.socialist_step()
        elif self.ideology == "capitalist":
            self.capitalist_step()
        elif self.ideology == "green_capitalist":
            self.capitalist_green_step()
        elif self.ideology == "green_socialist":
            self.socialist_green_step()
        elif self.ideology == "adaptive":
            self.adaptive_step()
        elif self.ideology == "adaptive_direct":
            self.adaptive_direct_step()
        else:
            self.capitalist_step()

        # baseline upkeep
        upkeep = 0.3 if self.ideology == "adaptive" else 0.5
        self.energy -= upkeep
        if self.energy <= 0:
            if self.ideology == "adaptive":
                if getattr(self, "use_mc_terminal_updates", False):
                    self._mc_update_on_death()
                if self.episodes < getattr(self, "max_adaptive_respawns", float("inf")):
                    self.episodes += 1
                    self._respawn_with_memory()
                    return
                self._remove_self()
                return

            self._remove_self()
            return

    def _remove_self(self):
        try:
            self.model.grid.remove_agent(self)
        except Exception:
            pass
        try:
            self.model.schedule.remove(self)
        except Exception:
            pass

    def _find_spawn_cell(self, max_tries: int = 1000):
        """Find a random empty cell; fallback to center if none found."""
        W, H = self.model.width, self.model.height
        for _ in range(max_tries):
            pos = (random.randrange(W), random.randrange(H))
            try:
                if self.model.grid.is_cell_empty(pos):
                    return pos
            except Exception:
                if len(self.model.grid.get_cell_list_contents([pos])) == 0:
                    return pos
        return (W // 2, H // 2)

    def _respawn_with_memory(self):
        inherited_q   = self.q_table
        inherited_eps = self.rl_epsilon
        inherited_eps_count = self.episodes
        inherited_cap = self.max_adaptive_respawns
        inherited_mc  = self.use_mc_terminal_updates

        self._remove_self()

        succ = IdeologyAgent(self.model.next_id(), self.model, ideology="adaptive")
        succ.q_table = inherited_q
        succ.rl_epsilon = max(succ.rl_epsilon_min, inherited_eps * 0.98)
        succ.episodes = inherited_eps_count
        succ.max_adaptive_respawns = inherited_cap
        succ.use_mc_terminal_updates = inherited_mc

        spawn_pos = self._find_spawn_cell()
        self.model.grid.place_agent(succ, spawn_pos)
        self.model.schedule.add(succ)

    def _mc_update_on_death(self):
        """
        Monte Carlo credit assignment with terminal-only reward for survival.
        If an episode lasted T steps, each (s_t, a_t) gets return G_t = γ^(T-1-t).
        Only used if self.use_mc_terminal_updates is True.
        """
        traj = getattr(self, "episode_traj", [])
        if not traj:
            return
        T = len(traj)
        for t, (s, a) in enumerate(traj):
            G = (self.rl_gamma ** (T - 1 - t)) * 1.0
            row = self._qrow(s)
            row[a] += self.rl_alpha * (G - row[a])
        self.episode_traj.clear()

    # ---------- MAINTENANCE HELPERS ----------
    def _nearest_degraded_patch(self):
        """Return (pos, dist, patch) for closest degraded renewable."""
        best = (None, 10**9, None)
        for pos in list(self.model.renewable_locations):
            cell = self.model.grid.get_cell_list_contents([pos])
            patch = next((o for o in cell if isinstance(o, ResourcePatch) and o.resource_type == "renewable"), None)
            if not patch or not (getattr(patch, "is_degraded", False) or getattr(patch, "degraded", False)):
                continue
            d = self.manhattan_distance(self.pos, pos)
            if d < best[1]:
                best = (pos, d, patch)
        return best

    def _should_repair(self, patch) -> bool:
        """Ideology-specific willingness to spend energy to repair."""
        cost = getattr(self.model, "repair_energy_cost", 10.0)
        if self.energy < cost:
            return False

        avg_e = self.model.average_energy()
        if self.ideology == "capitalist":
            nearby_non = any(self.manhattan_distance(self.pos, pos) <= 5 for pos in self.model.nonrenewable_locations)
            return (not nearby_non) and (self.energy >= cost + 5)

        if self.ideology == "green_capitalist":
            nearby_non = any(self.manhattan_distance(self.pos, pos) <= 5 for pos in self.model.nonrenewable_locations)
            local_scar = getattr(patch, "scar_level", 0.0)
            return (not nearby_non or local_scar >= 0.5) and (self.energy >= cost + 5)

        if self.ideology == "socialist":
            floor = getattr(self.model, "pool_floor", 10.0)
            return (avg_e <= 6.0) or (self.energy >= max(floor, cost + 2))

        if self.ideology == "green_socialist":
            floor = getattr(self.model, "pool_floor", 10.0)
            return self.energy >= max(floor, cost)

        if self.ideology == "adaptive":
            # conservative unless quite healthy
            return self.energy >= (cost + 8.0)

        return False

    def _maybe_repair_instant(self) -> bool:
        """
        If on a degraded renewable and ideology says yes, spend energy and repair instantly.
        Returns True if we repaired (or moved to repair in this tick).
        """
        # If we're standing on a renewable, try to repair immediately
        cell = self.model.grid.get_cell_list_contents([self.pos])
        patch_here = next((o for o in cell if isinstance(o, ResourcePatch) and o.resource_type == "renewable"), None)
        if patch_here and (getattr(patch_here, "is_degraded", False) or getattr(patch_here, "degraded", False)) and self._should_repair(patch_here):
            cost = getattr(self.model, "repair_energy_cost", 10.0)
            if self.energy >= cost:
                self.energy -= cost
                # clear flags
                if hasattr(patch_here, "clear_degraded"):
                    patch_here.clear_degraded()
                else:
                    patch_here.is_degraded = False
                    patch_here.degraded = False
                    patch_here.under_maintenance = False
                # small heal
                patch_here.amount = max(patch_here.amount, int(0.5 * patch_here.max_capacity))
                return True

        # If not on one, optionally move toward closest degraded if we intend to repair
        target_pos, dist, patch = self._nearest_degraded_patch()
        if patch is None:
            return False
        if not self._should_repair(patch):
            return False

        if self.pos != target_pos:
            self.move_towards(target_pos, speed=1)
            return True  # took our action this tick (movement toward repair)
        return False

    # ---------- RL helpers (Adaptive) ----------
    def _expected_net_here(self, p) -> float:
        """Expected one-burst net on current tile, including setup if unpaid."""
        if p is None:
            return -999.0
        setup_due  = (p.resource_type == "renewable") and (p.unique_id not in self.renewable_setup_paid)
        setup_cost = self.model.cost_renewable_setup if setup_due else 0.0
        if p.resource_type == "renewable":
            desired, op_cost = self.model.yield_per_mine_renewable, self.model.cost_extract_renewable
        else:
            desired, op_cost = self.model.yield_per_mine_nonrenewable, self.model.cost_extract_nonrenewable
        take = min(getattr(p, "amount", 0), desired)
        return take - op_cost - setup_cost

    def _state_from_obs(self) -> tuple:
        """
        Compact, discrete state for tabular Q-learning.
        Now also encodes: expected net yield on this tile, and global nonrenewable scarcity.
        """
        # ---- energy bin
        e = self.energy
        if e <= 3: e_bin = 0
        elif e <= 7: e_bin = 1
        elif e <= 12: e_bin = 2
        else: e_bin = 3

        # ---- what's under my feet?
        cell = self.model.grid.get_cell_list_contents([self.pos])
        patch = next((o for o in cell if isinstance(o, ResourcePatch)), None)
        hub_here = any(getattr(o, "built", False) for o in cell if isinstance(o, EnergyHub))

        if patch is None:
            tile_type = 0      # empty
            amt_bin = 0
            scar_bin = 0
            degraded = 0
        else:
            if patch.resource_type == "renewable":
                if (getattr(patch, "under_maintenance", False)
                    or getattr(patch, "degraded", False)
                    or getattr(patch, "is_degraded", False)):
                    tile_type = 2  # renewable but unusable
                else:
                    tile_type = 1  # healthy renewable
            else:
                tile_type = 3      # nonrenewable

            # amount bin
            a = getattr(patch, "amount", 0)
            if a <= 0: amt_bin = 0
            elif a <= 2: amt_bin = 1
            elif a <= 4: amt_bin = 2
            else: amt_bin = 3

            # scar (coarse)
            scar = getattr(patch, "scar_level", 0.0)
            scar_bin = int(min(3, math.floor(scar)))

            degraded = 1 if (
                getattr(patch, "degraded", False)
                or getattr(patch, "under_maintenance", False)
                or getattr(patch, "is_degraded", False)
            ) else 0

        # ---- coarse distances to nearest renewable / nonrenewable
        def dist_bin_to(rtype: str):
            pos, d, p = self._nearest_patch(rtype)
            if p is None:
                return 3  # none/very far
            if d == 0: return 0
            if d <= 3: return 1
            if d <= 8: return 2
            return 3

        d_ren = dist_bin_to("renewable")
        d_non = dist_bin_to("nonrenewable")

        # ---- expected net if we mine HERE now (includes setup if unpaid)
        exp_net = self._expected_net_here(patch)
        if exp_net <= -2:   net_bin = 0
        elif exp_net <= 0:  net_bin = 1
        elif exp_net <= 2:  net_bin = 2
        elif exp_net <= 4:  net_bin = 3
        else:               net_bin = 4

        # ---- global scarcity of nonrenewables
        non_count = len(self.model.nonrenewable_locations)
        if non_count == 0:        non_bin = 0
        elif non_count <= 30:     non_bin = 1
        elif non_count <= 70:     non_bin = 2
        else:                     non_bin = 3

        # return the extended state
        return (
            e_bin, tile_type, amt_bin, scar_bin, degraded, int(hub_here),
            d_ren, d_non, net_bin, non_bin
        )

    def _qrow(self, state: tuple) -> dict:
        if state not in self.q_table:
            q0 = getattr(self, "q_init", 15.0)
            row = {a: q0 for a in self.RL_ACTIONS}
            if "idle" in row:
                row["idle"] = -1.0
            self.q_table[state] = row
        return self.q_table[state]

    def _choose_action(self, state: tuple) -> str:
        # epsilon-greedy
        if random.random() < self.rl_epsilon:
            return random.choice(self.RL_ACTIONS)
        row = self._qrow(state)
        # argmax with deterministic tie-break
        return max(self.RL_ACTIONS, key=lambda a: row[a])

    def _best_q(self, state: tuple) -> float:
        row = self._qrow(state)
        return max(row.values())

    def _update_q(self, s: tuple, a: str, r: float, s2: tuple) -> None:
        row = self._qrow(s)
        td_target = r + self.rl_gamma * self._best_q(s2)
        row[a] += self.rl_alpha * (td_target - row[a])

    # --- Survival-first reward for the ADAPTIVE agent ---
    def _calc_reward(
        self,
        action: str,
        pre_e: float,
        post_e: float,
        patch=None,
        event_bonus: float = 0.0,
        renewable_bonus: float = 0.0,
        nonrenewable_bonus: float = 0.0,
        idle_penalty: float = 0.0,
        bad_action_penalty: float = 0.0,
        alive_bonus: float = 0.0,
        max_abs_reward: float = 20.0,
    ) -> float:
        """
        Survival-first reward:
          +1 if, after this action, the agent would still be alive AFTER next upkeep,
           0 otherwise (optionally add a small death penalty).
        """
        UPCOMING_UPKEEP = 0.3
        USE_DEATH_PENALTY = True
        DEATH_PENALTY = -3.0

        will_survive = (self.energy - UPCOMING_UPKEEP) > 0.0
        if will_survive:
            return 1.0
        else:
            return 0.0 + (DEATH_PENALTY if USE_DEATH_PENALTY else 0.0)

    # ---------- CAPITALIST ----------
    def capitalist_step(self) -> None:
        # maintenance first
        if self._maybe_repair_instant():
            return

        # Continue mining if already engaged
        if self.mining:
            self.mining_counter -= 1
            if self.mining_counter <= 0:
                cell_contents = self.model.grid.get_cell_list_contents([self.pos])
                for obj in cell_contents:
                    if isinstance(obj, ResourcePatch) and obj == self.mining_target:
                        if obj.resource_type == "renewable" and obj.unique_id not in self.renewable_setup_paid:
                            self.energy -= self.model.cost_renewable_setup
                            self.renewable_setup_paid.add(obj.unique_id)

                        if obj.resource_type == "renewable":
                            desired = self.model.yield_per_mine_renewable
                            op_cost = self.model.cost_extract_renewable
                        else:
                            desired = self.model.yield_per_mine_nonrenewable
                            op_cost = self.model.cost_extract_nonrenewable

                        gained = obj.harvest(desired)
                        net_gain = gained - op_cost
                        self.energy += net_gain
                        if net_gain > 0:
                            self.total_collected_energy += net_gain

                        if obj.amount <= 0 and obj.resource_type == "nonrenewable":
                            self.model.grid.remove_agent(obj)
                            self.model.schedule.remove(obj)
                            try:
                                self.model.nonrenewable_locations.remove(self.pos)
                            except ValueError:
                                pass

                self.mining = False
                self.mining_target = None
            return

        # Targeting logic
        closest_nonrenewable = None
        min_dist_nonrenewable = None
        for pos in self.model.nonrenewable_locations:
            if any(isinstance(a, IdeologyAgent) for a in self.model.grid.get_cell_list_contents([pos])):
                continue
            cell_objs = self.model.grid.get_cell_list_contents([pos])
            patch = next((o for o in cell_objs if isinstance(o, ResourcePatch)
                          and o.resource_type == "nonrenewable" and o.amount > 0), None)
            if patch is None:
                continue
            dist = self.manhattan_distance(self.pos, pos)
            if min_dist_nonrenewable is None or dist < min_dist_nonrenewable:
                min_dist_nonrenewable = dist
                closest_nonrenewable = pos

        best_patch_pos = None
        if closest_nonrenewable is not None and (min_dist_nonrenewable <= 10):
            best_patch_pos = closest_nonrenewable
        else:
            closest_renewable = None
            min_dist_renewable = None
            for pos in self.model.renewable_locations:
                if any(isinstance(a, IdeologyAgent) for a in self.model.grid.get_cell_list_contents([pos])):
                    continue
                cell_objs = self.model.grid.get_cell_list_contents([pos])
                patch = next((o for o in cell_objs if isinstance(o, ResourcePatch)
                              and o.resource_type == "renewable" and o.amount > 0), None)
                if patch is None:
                    continue
                dist = self.manhattan_distance(self.pos, pos)
                if min_dist_renewable is None or dist < min_dist_renewable:
                    min_dist_renewable = dist
                    closest_renewable = pos
            best_patch_pos = closest_renewable if closest_renewable else closest_nonrenewable

        if best_patch_pos:
            speed = 2 if self.energy > 15 else 1
            self.move_towards(best_patch_pos, speed=speed)
            if self.pos == best_patch_pos:
                # try repair if this is a degraded renewable
                if self._maybe_repair_instant():
                    return
                self.mining = True
                self.mining_counter = 3
                resources_here = [o for o in self.model.grid.get_cell_list_contents([self.pos]) if isinstance(o, ResourcePatch)]
                self.mining_target = resources_here[0] if resources_here else None

    # ---------- SOCIALIST ----------
    def socialist_step(self) -> None:
        # maintenance first
        if self._maybe_repair_instant():
            return

        if self.mining:
            self.mining_counter -= 1
            if self.mining_counter <= 0:
                cell = self.model.grid.get_cell_list_contents([self.pos])
                patch = next((o for o in cell if isinstance(o, ResourcePatch)), None)
                hub_here = any(isinstance(o, EnergyHub) and o.built for o in cell)

                if patch:
                    if patch.resource_type == "renewable":
                        if patch.unique_id not in self.renewable_setup_paid:
                            setup_cost = self.model.cost_renewable_setup
                            if self.energy >= setup_cost:
                                self.energy -= setup_cost
                                self.renewable_setup_paid.add(patch.unique_id)
                            else:
                                self.mining = False
                                self.mining_target = None
                                return

                        desired = self.model.yield_per_mine_renewable + (1 if hub_here else 0)
                        op_cost = max(0, self.model.cost_extract_renewable - (1 if hub_here else 0))
                    else:
                        desired = self.model.yield_per_mine_nonrenewable
                        op_cost = self.model.cost_extract_nonrenewable

                    gained = patch.harvest(desired)
                    net_gain = gained - op_cost
                    self.energy += net_gain
                    if net_gain > 0:
                        self.total_collected_energy += net_gain

                    if patch.amount <= 0 and patch.resource_type == "nonrenewable":
                        self.model.grid.remove_agent(patch)
                        self.model.schedule.remove(patch)
                        if patch.pos in self.model.nonrenewable_locations:
                            try:
                                self.model.nonrenewable_locations.remove(patch.pos)
                            except ValueError:
                                pass

                    # Safe tithe + wealth cap using pool_floor as guard
                    floor = getattr(self.model, "pool_floor", 10.0)
                    if net_gain > 0 and self.energy > floor:
                        tithe = self.model.tithe_rate * net_gain
                        tithe = min(tithe, max(0.0, self.energy - floor))
                        if tithe > 0:
                            self.energy -= tithe
                            self.model.community_pool += tithe

                    safe_cap = max(self.energy_cap, floor)
                    if self.energy > safe_cap:
                        skim = max(0.0, self.energy - safe_cap)
                        if skim > 0:
                            self.energy -= skim
                            self.model.community_pool += skim

                self.mining = False
                self.mining_target = None
                self.redistribute_to_neighbors()
            return

        # Target selection
        if self.energy < self.emergency_floor:
            pos_r, dist_r, patch_r = self._nearest_patch("renewable")
            pos_n, dist_n, patch_n = self._nearest_patch("nonrenewable")
            if patch_r is None and patch_n is None:
                self.idle_wander(); return
            target = pos_r if (patch_n is None or (patch_r is not None and dist_r <= dist_n)) else pos_n
        else:
            avg_e = self.model.average_energy()
            preferred = "renewable" if avg_e > 5.0 else "nonrenewable"
            pos_p, dist_p, patch_p = self._nearest_patch(preferred)
            if patch_p is None:
                other = "nonrenewable" if preferred == "renewable" else "renewable"
                pos_p, dist_p, patch_p = self._nearest_patch(other)
                if patch_p is None:
                    self.idle_wander(); return
            target = pos_p

        if target:
            self.move_towards(target, speed=1)
            if self.pos == target:
                # repair if needed, otherwise hub/mine
                if self._maybe_repair_instant():
                    return
                self._maybe_build_hub_or_mine()
                return

        if self.energy > (self.min_keep + 6):
            self.redistribute_to_neighbors()

    # ---------- GREEN CAPITALIST ----------
    def capitalist_green_step(self):
        if self._maybe_repair_instant():
            return

        if self.mining:
            self.mining_counter -= 1
            if self.mining_counter <= 0:
                cell = self.model.grid.get_cell_list_contents([self.pos])
                patch = next((o for o in cell if isinstance(o, ResourcePatch)), None)
                if patch:
                    if patch.resource_type == "renewable":
                        if patch.unique_id not in self.renewable_setup_paid and self.energy >= self.model.cost_renewable_setup:
                            self.energy -= self.model.cost_renewable_setup
                            self.renewable_setup_paid.add(patch.unique_id)
                        desired = self.model.yield_per_mine_renewable
                        op_cost = self.model.cost_extract_renewable
                    else:
                        desired = self.model.yield_per_mine_nonrenewable
                        op_cost = self.model.cost_extract_nonrenewable

                    gained = patch.harvest(desired)
                    net = gained - op_cost

                    if gained > 0:
                        if patch.resource_type == "nonrenewable":
                            tax = self.model.carbon_tax_per_unit * gained
                            net -= tax
                            self.model.community_pool += max(0.0, tax)
                        else:
                            subsidy = self.model.renewable_subsidy_per_unit * gained
                            net += subsidy

                    self.energy += net
                    if net > 0:
                        self.total_collected_energy += net

                    if patch.amount <= 0 and patch.resource_type == "nonrenewable":
                        self.model.grid.remove_agent(patch)
                        self.model.schedule.remove(patch)
                        if patch.pos in self.model.nonrenewable_locations:
                            try:
                                self.model.nonrenewable_locations.remove(patch.pos)
                            except ValueError:
                                pass

                self.mining = False
                self.mining_target = None
            return

        best_pos, best_score = None, -1e18
        # scan renewables
        for pos in list(self.model.renewable_locations):
            cell = self.model.grid.get_cell_list_contents([pos])
            patch = next((o for o in cell if isinstance(o, ResourcePatch) and o.amount > 0 and o.resource_type == "renewable"), None)
            if not patch:
                continue
            if any(isinstance(a, IdeologyAgent) for a in cell):
                continue
            s = self._green_profit_score(pos, patch)
            if s > best_score:
                best_pos, best_score = pos, s

        # scan nonrenewables
        for pos in list(self.model.nonrenewable_locations):
            cell = self.model.grid.get_cell_list_contents([pos])
            patch = next((o for o in cell if isinstance(o, ResourcePatch) and o.amount > 0 and o.resource_type == "nonrenewable"), None)
            if not patch:
                continue
            if any(isinstance(a, IdeologyAgent) for a in cell):
                continue
            s = self._green_profit_score(pos, patch)
            if s > best_score:
                best_pos, best_score = pos, s

        if best_pos:
            self.move_towards(best_pos, speed=2 if self.energy > 15 else 1)
            if self.pos == best_pos:
                if self._maybe_repair_instant():
                    return
                self.mining = True
                self.mining_counter = 3
                self.mining_target = next((o for o in self.model.grid.get_cell_list_contents([self.pos]) if isinstance(o, ResourcePatch)), None)
        else:
            self.idle_wander()

    # ---------- GREEN SOCIALIST ----------
    def socialist_green_step(self):
        if self._maybe_repair_instant():
            return

        if self.mining:
            self.mining_counter -= 1
            if self.mining_counter <= 0:
                cell = self.model.grid.get_cell_list_contents([self.pos])
                patch = next((o for o in cell if isinstance(o, ResourcePatch)), None)
                hub_here = any(isinstance(o, EnergyHub) and o.built for o in cell)

                if patch:
                    if patch.resource_type == "renewable":
                        if patch.unique_id not in self.renewable_setup_paid:
                            setup_cost = self.model.cost_renewable_setup
                            if self.energy >= setup_cost:
                                self.energy -= setup_cost
                                self.renewable_setup_paid.add(patch.unique_id)
                            else:
                                self.mining = False
                                self.mining_target = None
                                return
                        desired = self.model.yield_per_mine_renewable + (1 if hub_here else 0)
                        op_cost = max(0, self.model.cost_extract_renewable - (1 if hub_here else 0))
                        subsidy = self.model.renewable_subsidy_per_unit * desired
                    else:
                        desired = self.model.yield_per_mine_nonrenewable
                        op_cost = self.model.cost_extract_nonrenewable
                        subsidy = - self.model.carbon_tax_per_unit * desired

                    gained = patch.harvest(desired)
                    net_gain = gained - op_cost + subsidy
                    self.energy += net_gain
                    if net_gain > 0:
                        self.total_collected_energy += net_gain

                    if patch.amount <= 0 and patch.resource_type == "nonrenewable":
                        self.model.grid.remove_agent(patch)
                        self.model.schedule.remove(patch)
                        if patch.pos in self.model.nonrenewable_locations:
                            try:
                                self.model.nonrenewable_locations.remove(patch.pos)
                            except ValueError:
                                pass

                    # Redistribution + cap guarded by pool_floor
                    floor = getattr(self.model, "pool_floor", 10.0)
                    if net_gain > 0 and self.energy > floor:
                        tithe = self.model.tithe_rate * net_gain
                        tithe = min(tithe, max(0.0, self.energy - floor))
                        if tithe > 0:
                            self.energy -= tithe
                            self.model.community_pool += tithe
                    safe_cap = max(self.energy_cap, floor)
                    if self.energy > safe_cap:
                        skim = max(0.0, self.energy - safe_cap)
                        if skim > 0:
                            self.energy -= skim
                            self.model.community_pool += skim

                self.mining = False
                self.mining_target = None
                self.redistribute_to_neighbors()
            return

        avg_energy = self.model.average_energy()
        best_pos, best_score = None, -1e18
        for res_type in ["renewable", "nonrenewable"]:
            if avg_energy > 5 and res_type == "nonrenewable":
                continue
            if avg_energy <= 5 and res_type == "renewable":
                continue
            locations = self.model.renewable_locations if res_type == "renewable" else self.model.nonrenewable_locations
            for pos in list(locations):
                cell = self.model.grid.get_cell_list_contents([pos])
                patch = next((o for o in cell if isinstance(o, ResourcePatch) and o.amount > 0 and o.resource_type == res_type), None)
                if not patch:
                    continue
                score = self._green_profit_score(pos, patch)
                if score > best_score:
                    best_pos, best_score = pos, score

        if best_pos:
            self.move_towards(best_pos, speed=1)
            if self.pos == best_pos:
                if self._maybe_repair_instant():
                    return
                self._maybe_build_hub_or_mine()
        else:
            self.idle_wander()

        if self.energy > (self.min_keep + 6):
            self.redistribute_to_neighbors()

    # ---------- Adaptive (Tabular Q-learning; time-survival reward) ----------
    def adaptive_step(self) -> None:
        """
        Reward = 1 for each tick survived (i.e., if energy after NEXT upkeep > 0),
        0 on the tick before death (plus small death penalty).
        SMDP update for multi-tick mining bursts.
        """
        UPCOMING_UPKEEP = 0.3
        USE_DEATH_PENALTY = True
        DEATH_PENALTY = -3.0

        def survival_reward():
            will_survive = (self.energy - UPCOMING_UPKEEP) > 0
            return 1.0 if will_survive else (0.0 + (DEATH_PENALTY if USE_DEATH_PENALTY else 0.0))

        # ----------------------------
        # 1) Resolve ongoing mining (accumulate discounted rewards per tick)
        # ----------------------------
        if self.mining:
            # accumulate survival reward for this mining tick
            self._mine_k += 1
            self._mine_accum_r += (self.rl_gamma ** (self._mine_k - 1)) * survival_reward()

            # finish burst?
            self.mining_counter -= 1
            if self.mining_counter <= 0:
                cell = self.model.grid.get_cell_list_contents([self.pos])
                patch = next((o for o in cell if isinstance(o, ResourcePatch)), None)

                if patch:
                    # pay renewable setup once if needed
                    if patch.resource_type == "renewable" and patch.unique_id not in self.renewable_setup_paid:
                        if self.energy >= self.model.cost_renewable_setup:
                            self.energy -= self.model.cost_renewable_setup
                            self.renewable_setup_paid.add(patch.unique_id)
                        else:
                            # couldn't pay; abort mining (still do SMDP update below)
                            pass

                    # do the actual harvest (if setup OK or nonrenewable)
                    if (patch.resource_type != "renewable") or (patch.unique_id in self.renewable_setup_paid):
                        if patch.resource_type == "renewable":
                            desired = self.model.yield_per_mine_renewable
                            op_cost = self.model.cost_extract_renewable
                        else:
                            desired = self.model.yield_per_mine_nonrenewable
                            op_cost = self.model.cost_extract_nonrenewable

                        gained = patch.harvest(desired)
                        net = gained - op_cost
                        self.energy += net
                        if net > 0:
                            self.total_collected_energy += net

                        # remove depleted nonrenewable
                        if patch.amount <= 0 and patch.resource_type == "nonrenewable":
                            try:
                                self.model.grid.remove_agent(patch)
                                self.model.schedule.remove(patch)
                            except Exception:
                                pass
                            try:
                                if patch.pos in self.model.nonrenewable_locations:
                                    self.model.nonrenewable_locations.remove(patch.pos)
                            except ValueError:
                                pass

                # SMDP Q-update for the staged (state, "mine")
                if self._last_state is not None and self._last_action is not None:
                    s2 = self._state_from_obs()
                    k  = max(1, self._mine_k)
                    r_k = self._mine_accum_r
                    row = self._qrow(self._last_state)
                    td_target = r_k + (self.rl_gamma ** k) * self._best_q(s2)
                    row[self._last_action] += self.rl_alpha * (td_target - row[self._last_action])
                    self._last_state, self._last_action = None, None

                self.mining = False
                self.mining_target = None
                self._mine_k = 0
                self._mine_accum_r = 0.0
            return

        # -----------------------------------
        # 2) Emergency: go to nearest positive-net patch (R or NR)
        # -----------------------------------
        if self.energy <= max(self.emergency_floor, 6.0):
            candidates = []
            for rtype in ("nonrenewable", "renewable"):
                pos, _, p = self._nearest_patch(rtype)
                if p and getattr(p, "amount", 0) > 0 and not (
                    getattr(p, "degraded", False) or getattr(p, "under_maintenance", False) or getattr(p, "is_degraded", False)
                ):
                    en = self._expected_net_here(p)
                    candidates.append((pos, p, en))
            if candidates:
                positives = [c for c in candidates if c[2] > 0]
                pool = positives if positives else candidates
                pos, patch, _ = min(pool, key=lambda c: self.manhattan_distance(self.pos, c[0]))
                if self.pos != pos:
                    self.move_towards(pos, speed=2)
                    return
                # start short mining burst; stage SMDP update on completion
                self.mining = True
                self.mining_counter = getattr(self, "adaptive_mine_ticks", 2)
                self.mining_target = patch
                self._last_state = self._state_from_obs()
                self._last_action = "mine"
                self._mine_k = 0
                self._mine_accum_r = 0.0
                return

        # ----------------------------
        # 3) Normal RL step
        # ----------------------------
        s = self._state_from_obs()

        low_energy = (self.energy <= 5.0)
        action_space = ["move_N", "move_S", "move_E", "move_W", "mine"] if low_energy else self.RL_ACTIONS

        if random.random() < self.rl_epsilon:
            a = random.choice(action_space)
        else:
            row = self._qrow(s)
            vals = [row[x] for x in action_space]
            maxv, minv = max(vals), min(vals)
            if (maxv - minv) < 1e-6:
                a = self._heuristic_best_action(allow_idle=("idle" in action_space), low_energy=low_energy)
                if a not in action_space:
                    a = "mine" if "mine" in action_space else random.choice([m for m in action_space if m.startswith("move_")])
            else:
                best = [x for x in action_space if row[x] == maxv]
                a = random.choice(best)

        # If we’re already on a usable patch, mine only if net>0 (or emergency buffer)
        if self._usable_here():
            cell = self.model.grid.get_cell_list_contents([self.pos])
            patch_here = next((o for o in cell if isinstance(o, ResourcePatch)), None)
            if self._expected_net_here(patch_here) > 0 or self.energy <= self.emergency_floor + 1:
                a = "mine"

        # Execute action
        if a.startswith("move_"):
            dx, dy = 0, 0
            if a == "move_N": dy = 1
            elif a == "move_S": dy = -1
            elif a == "move_E": dx = 1
            elif a == "move_W": dx = -1
            target = (self.pos[0] + dx, self.pos[1] + dy)
            self.move_towards(target, speed=1)

        elif a == "mine":
            cell = self.model.grid.get_cell_list_contents([self.pos])
            patch = next((o for o in cell if isinstance(o, ResourcePatch)), None)
            if patch and not getattr(patch, "degraded", False) and not getattr(patch, "under_maintenance", False) \
               and not getattr(patch, "is_degraded", False) and patch.amount > 0:

                # handle renewable setup up front
                if patch.resource_type == "renewable" and patch.unique_id not in self.renewable_setup_paid:
                    setup = self.model.cost_renewable_setup
                    if self.energy < setup:
                        pass  # can't start yet; fall through to update below
                    else:
                        self.energy -= setup
                        self.renewable_setup_paid.add(patch.unique_id)

                # start burst if allowed
                if (patch.resource_type != "renewable") or (patch.unique_id in self.renewable_setup_paid):
                    self.mining = True
                    self.mining_counter = getattr(self, "adaptive_mine_ticks", 2)
                    self.mining_target = patch
                    self._last_state, self._last_action = s, a
                    self._mine_k = 0
                    self._mine_accum_r = 0.0
                    return
            # else: invalid mine attempt; we’ll just update below

        elif a == "repair":
            if self.energy >= 18.0:
                cell = self.model.grid.get_cell_list_contents([self.pos])
                patch = next((o for o in cell if isinstance(o, ResourcePatch) and o.resource_type == "renewable"), None)
                if patch and (getattr(patch, "degraded", False) or getattr(patch, "under_maintenance", False) or getattr(patch, "is_degraded", False)):
                    self.energy -= 10.0
                    if hasattr(patch, "clear_degraded"):
                        patch.clear_degraded()
                    else:
                        patch.is_degraded = False
                        patch.degraded = False
                        patch.under_maintenance = False

        # 1-step Q-learning update with survival reward
        s2 = self._state_from_obs()
        r = survival_reward()
        self._update_q(s, a, r, s2)

        # ε decay
        if self.rl_epsilon > self.rl_epsilon_min:
            self.rl_epsilon = max(self.rl_epsilon_min, self.rl_epsilon * self.rl_epsilon_decay)
    def adaptive_direct_step(self) -> None:
        """
        Simpler RL: actions are high-level choices:
        - goto_nearest_non: move one step toward the closest usable nonrenewable
        - goto_nearest_ren: move one step toward the closest usable renewable
        - mine: begin/continue a short mining burst if possible
        - repair: repair a degraded renewable if on it (when healthy enough)

        Reward = 1 per tick survived (energy after next upkeep > 0), else 0 (with small death penalty).
        """
        UPCOMING_UPKEEP = 0.3
        USE_DEATH_PENALTY = True
        DEATH_PENALTY = -3.0

        def survival_reward():
            return 1.0 if (self.energy - UPCOMING_UPKEEP) > 0 else (0.0 + (-3.0 if USE_DEATH_PENALTY else 0.0))

        # 1) Resolve ongoing mining bursts (same as adaptive)
        if self.mining:
            self.mining_counter -= 1
            if self.mining_counter <= 0:
                cell = self.model.grid.get_cell_list_contents([self.pos])
                patch = next((o for o in cell if isinstance(o, ResourcePatch)), None)

                if patch:
                    # one-time renewable setup
                    if patch.resource_type == "renewable" and patch.unique_id not in self.renewable_setup_paid:
                        if self.energy >= self.model.cost_renewable_setup:
                            self.energy -= self.model.cost_renewable_setup
                            self.renewable_setup_paid.add(patch.unique_id)
                        else:
                            # abort burst; still do the Q update below
                            pass

                    # do harvest
                    if patch.resource_type == "renewable":
                        desired = self.model.yield_per_mine_renewable
                        op_cost = self.model.cost_extract_renewable
                    else:
                        desired = self.model.yield_per_mine_nonrenewable
                        op_cost = self.model.cost_extract_nonrenewable

                    gained = patch.harvest(desired)
                    net = gained - op_cost
                    self.energy += net
                    if net > 0:
                        self.total_collected_energy += net

                    # remove empty nonrenewable
                    if patch.amount <= 0 and patch.resource_type == "nonrenewable":
                        try:
                            self.model.grid.remove_agent(patch)
                            self.model.schedule.remove(patch)
                        except Exception:
                            pass
                        try:
                            if patch.pos in self.model.nonrenewable_locations:
                                self.model.nonrenewable_locations.remove(patch.pos)
                        except ValueError:
                            pass

                # Q update for previously staged (s,"mine")
                if self._last_state is not None and self._last_action is not None:
                    s2 = self._state_from_obs()
                    r = survival_reward()
                    self._update_q(self._last_state, self._last_action, r, s2)
                    self._last_state, self._last_action = None, None

                self.mining = False
                self.mining_target = None
            return

        # 2) Emergency: beeline toward usable energy if very low
        if self.energy <= max(self.emergency_floor, 6.0):
            # prefer nonrenewable first for immediate yield
            pos_n, _, pn = self._nearest_patch("nonrenewable")
            tgt_pos, tgt_patch = (pos_n, pn) if (pn and pn.amount > 0) else (None, None)
            if tgt_patch is None:
                pos_r, _, pr = self._nearest_patch("renewable")
                if pr and not (getattr(pr, "degraded", False) or getattr(pr, "under_maintenance", False) or getattr(pr, "is_degraded", False)) and pr.amount > 0:
                    tgt_pos, tgt_patch = pos_r, pr

            if tgt_patch is not None and tgt_pos is not None:
                if self.pos != tgt_pos:
                    self.move_towards(tgt_pos, speed=2)
                    return
                # start a short burst
                self.mining = True
                self.mining_counter = getattr(self, "adaptive_mine_ticks", 2)
                self.mining_target = tgt_patch
                self._last_state = self._state_from_obs()
                self._last_action = "mine"
                return

        # 3) RL choose among the 4 actions
        s = self._state_from_obs()

        # ε-greedy on limited action set
        if random.random() < self.rl_epsilon:
            a = random.choice(self.RL_ACTIONS)
        else:
            row = self._qrow(s)
            bestv = max(row[x] for x in self.RL_ACTIONS)
            best = [x for x in self.RL_ACTIONS if row[x] == bestv]
            a = random.choice(best)

        # If we can actually mine here, override to "mine"
        if self._usable_here():
            a = "mine"

        # Execute the chosen action
        if a == "goto_nearest_non":
            pos, _, patch = self._nearest_patch("nonrenewable")
            if patch is not None and pos is not None and pos != self.pos:
                self.move_towards(pos, speed=1)

        elif a == "goto_nearest_ren":
            pos, _, patch = self._nearest_patch("renewable")
            if patch is not None and pos is not None and pos != self.pos:
                self.move_towards(pos, speed=1)

        elif a == "mine":
            cell = self.model.grid.get_cell_list_contents([self.pos])
            patch = next((o for o in cell if isinstance(o, ResourcePatch)), None)
            if patch and not (getattr(patch, "degraded", False) or getattr(patch, "under_maintenance", False) or getattr(patch, "is_degraded", False)) and patch.amount > 0:
                if patch.resource_type == "renewable" and patch.unique_id not in self.renewable_setup_paid:
                    setup = self.model.cost_renewable_setup
                    if self.energy >= setup:
                        self.energy -= setup
                        self.renewable_setup_paid.add(patch.unique_id)
                # start the burst only if setup is paid or not needed
                if (patch.resource_type != "renewable") or (patch.unique_id in self.renewable_setup_paid):
                    self.mining = True
                    self.mining_counter = getattr(self, "adaptive_mine_ticks", 2)
                    self.mining_target = patch
                    self._last_state, self._last_action = s, a
                    return

        elif a == "repair":
            if self.energy >= 18.0:
                cell = self.model.grid.get_cell_list_contents([self.pos])
                patch = next((o for o in cell if isinstance(o, ResourcePatch) and o.resource_type == "renewable"), None)
                if patch and (getattr(patch, "degraded", False) or getattr(patch, "under_maintenance", False) or getattr(patch, "is_degraded", False)):
                    self.energy -= 10.0
                    if hasattr(patch, "clear_degraded"):
                        patch.clear_degraded()
                    else:
                        patch.is_degraded = False
                        patch.degraded = False
                        patch.under_maintenance = False

        # Q update (survival-only reward)
        s2 = self._state_from_obs()
        r = survival_reward()
        self._update_q(s, a, r, s2)

        # ε decay
        if self.rl_epsilon > self.rl_epsilon_min:
            self.rl_epsilon = max(self.rl_epsilon_min, self.rl_epsilon * self.rl_epsilon_decay)


    
    def _usable_here(self) -> bool:
        """True if the tile underfoot can be mined right now."""
        cell = self.model.grid.get_cell_list_contents([self.pos])
        patch = next((o for o in cell if isinstance(o, ResourcePatch)), None)
        if not patch:
            return False
        if getattr(patch, "amount", 0) <= 0:
            return False
        if (getattr(patch, "degraded", False)
            or getattr(patch, "under_maintenance", False)
            or getattr(patch, "is_degraded", False)):
            return False
        # renewable requires setup paid or affordable now
        if patch.resource_type == "renewable" and patch.unique_id not in self.renewable_setup_paid:
            if self.energy < self.model.cost_renewable_setup:
                return False
        return True

    def _nearest_usable_patch_any(self):
        """Nearest patch we can ACTUALLY exploit now (filters: amount>0, not degraded, setup affordable, not crowded)."""
        best = (None, 10**9, None)
        for rtype, locs in (("renewable", self.model.renewable_locations),
                            ("nonrenewable", self.model.nonrenewable_locations)):
            for pos in list(locs):
                cell = self.model.grid.get_cell_list_contents([pos])

                # skip if another agent is already standing there
                if any(isinstance(a, IdeologyAgent) and a is not self for a in cell):
                    continue

                patch = next((o for o in cell if isinstance(o, ResourcePatch) and o.resource_type == rtype), None)
                if not patch:
                    continue
                if getattr(patch, "amount", 0) <= 0:
                    continue

                # renewable must be usable now
                if rtype == "renewable":
                    if (getattr(patch, "degraded", False)
                        or getattr(patch, "under_maintenance", False)
                        or getattr(patch, "is_degraded", False)):
                        continue
                    # require setup affordable OR already paid
                    if patch.unique_id not in self.renewable_setup_paid and self.energy < self.model.cost_renewable_setup:
                        continue

                d = self.manhattan_distance(self.pos, pos)
                if rtype == "renewable":
                    d = max(0, d - self.renewable_bias)  # adaptive can set this to 0
                if d < best[1]:
                    best = (pos, d, patch)
        return best

    def _heuristic_best_action(self, allow_idle: bool, low_energy: bool) -> str:
        """Heuristic fallback: mine if standing on a usable patch, else step toward nearest usable patch."""
        cell = self.model.grid.get_cell_list_contents([self.pos])
        patch = next((o for o in cell if isinstance(o, ResourcePatch)), None)
        usable_here = (
            patch is not None
            and getattr(patch, "amount", 0) > 0
            and not (getattr(patch, "degraded", False) or getattr(patch, "under_maintenance", False) or getattr(patch, "is_degraded", False))
            and (patch.resource_type != "renewable"
                or (patch.unique_id in self.renewable_setup_paid or self.energy >= self.model.cost_renewable_setup))
        )
        if usable_here:
            # Only mine if it’s actually good (or you’re desperate)
            if self._expected_net_here(patch) > 0 or self.energy <= self.emergency_floor + 1:
                return "mine"

        target_pos, dist, _ = self._nearest_usable_patch_any()
        if target_pos is not None and dist > 0:
            cx, cy = self.pos
            tx, ty = target_pos
            if abs(tx - cx) >= abs(ty - cy):
                return "move_E" if tx > cx else "move_W" if tx < cx else ("move_N" if ty > cy else "move_S")
            else:
                return "move_N" if ty > cy else "move_S" if ty < cy else ("move_E" if tx > cx else "move_W")

        return "idle" if allow_idle and not low_energy else random.choice(["move_N", "move_S", "move_E", "move_W"])

    # ---------- helpers ----------
    def _maybe_build_hub_or_mine(self):
        cell = self.model.grid.get_cell_list_contents([self.pos])
        patch = next((o for o in cell if isinstance(o, ResourcePatch)), None)
        if not patch:
            return
        hub_here = any(isinstance(o, EnergyHub) and o.built for o in cell)

        if patch.resource_type == "renewable" and not hub_here:
            self.intent_build = True
            partners = [a for a in self.model.grid.get_cell_list_contents([self.pos])
                        if isinstance(a, IdeologyAgent) and a.ideology in ("socialist", "green_socialist")
                        and getattr(a, "intent_build", False)]
            if len(partners) >= 2 and self.energy >= 3:
                self.build_counter += 1
                self.energy -= 0.2
                if self.build_counter >= self.coop_build_time:
                    hub = EnergyHub(self.model.next_id(), self.model, self.pos)
                    self.model.grid.place_agent(hub, self.pos)
                    self.model.schedule.add(hub)
                    self.energy -= 2.0
                    self.build_counter = 0
                    self.intent_build = False
                    return
                else:
                    return
            else:
                self.build_counter = 0

        self.mining = True
        self.mining_counter = 3
        self.mining_target = patch

    def idle_wander(self):
        cx, cy = (self.model.width // 2, self.model.height // 2)
        self.move_towards((cx, cy), speed=1)

    def redistribute_to_neighbors(self):
        floor = getattr(self.model, "pool_floor", 10.0)
        if self.energy <= floor:
            return
        surplus = max(0.0, self.energy - floor)
        give_pool = self.share_fraction * surplus

        neighs = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=self.share_radius)
        needy = []
        for p in neighs:
            for a in self.model.grid.get_cell_list_contents([p]):
                if isinstance(a, IdeologyAgent) and a is not self and a.energy < self.help_threshold:
                    needy.append(a)
        if not needy or give_pool <= 0:
            return
        deficits = [self.help_threshold - a.energy for a in needy]
        total_def = sum(deficits)
        if total_def <= 0:
            return
        for a, d in zip(needy, deficits):
            if self.energy <= floor:
                break
            share_i = give_pool * (d / total_def)
            share_i = min(share_i, max(0.0, self.energy - floor))
            if share_i <= 0:
                continue
            a.energy += share_i
            self.energy -= share_i

    def move_towards(self, target_pos, speed: int = 1) -> None:
        x, y = self.pos
        tx, ty = target_pos
        for _ in range(speed):
            dx = tx - x
            dy = ty - y
            if dx == 0 and dy == 0:
                break
            # move along the dominant axis ONLY (Manhattan step)
            if abs(dx) >= abs(dy):
                nx, ny = x + (1 if dx > 0 else -1 if dx < 0 else 0), y
            else:
                nx, ny = x, y + (1 if dy > 0 else -1 if dy < 0 else 0)

            # wrap if torus, else stop at boundary
            if getattr(self.model.grid, "torus", False):
                nx %= self.model.width
                ny %= self.model.height
            elif self.model.grid.out_of_bounds((nx, ny)):
                break

            self.model.grid.move_agent(self, (nx, ny))
            x, y = nx, ny

    def manhattan_distance(self, p1, p2) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _nearest_patch(self, rtype: str):
        locs = self.model.renewable_locations if rtype == "renewable" else self.model.nonrenewable_locations
        best_pos, best_dist, best_patch = None, None, None
        for pos in list(locs):
            cell = self.model.grid.get_cell_list_contents([pos])
            patch = next((o for o in cell if isinstance(o, ResourcePatch) and o.resource_type == rtype and o.amount > 0), None)
            if not patch:
                continue
            # avoid crowding: skip if an agent stands here already
            if any(isinstance(a, IdeologyAgent) for a in cell):
                continue
            d = self.manhattan_distance(self.pos, pos)
            if rtype == "renewable":
                d = max(0, d - self.renewable_bias)
            if best_dist is None or d < best_dist:
                best_pos, best_dist, best_patch = pos, d, patch
        return best_pos, (best_dist if best_dist is not None else 10**9), best_patch

    def _green_profit_score(self, pos, patch) -> float:
        if patch.resource_type == "renewable":
            base_yield = self.model.yield_per_mine_renewable
            op_cost = self.model.cost_extract_renewable
            policy = self.model.renewable_subsidy_per_unit * base_yield
            scar = getattr(patch, "scar_level", 0.0)
            scar_pen = self.model.scar_avoid_alpha * scar
        else:
            base_yield = self.model.yield_per_mine_nonrenewable
            op_cost = self.model.cost_extract_nonrenewable
            policy = - self.model.carbon_tax_per_unit * base_yield
            scar_pen = 0.0
        est_net = (base_yield - op_cost) + policy - scar_pen
        d = self.manhattan_distance(self.pos, pos)
        return est_net / (d + 1)
