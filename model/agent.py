from mesa import Agent

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

        self.scar_level: float = 0.0

    def step(self) -> None:
        # --- decay scars every tick ---
        if self.scar_level > 0:
            self.scar_level = max(0.0, self.scar_level - self.model.scar_decay)

        # NEW: collapse renewable patch if scar is too high
        if self.resource_type == "renewable" and self.scar_level >= self.model.scar_collapse_threshold:
            # remove self from grid & schedule, and from model's renewable_locations cache
            try:
                self.model.grid.remove_agent(self)
            except Exception:
                pass
            try:
                self.model.schedule.remove(self)
            except Exception:
                pass
            # keep the cache in sync
            try:
                if hasattr(self.model, "renewable_locations") and self.pos in self.model.renewable_locations:
                    self.model.renewable_locations.remove(self.pos)
            except ValueError:
                pass
            return  # stop further processing; this patch is gone

        if self.resource_type == "renewable":
            if self.fatigue > 0:
                self.fatigue = max(0, self.fatigue - self.model.renewable_fatigue_decay)
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
                self.regen_rate = 0
            else:
                # Base regen, then apply scar penalty
                # penalty factor = max(0, 1 - alpha * scar_level)
                penalty = max(0.0, 1.0 - self.model.scar_regen_alpha * self.scar_level)
                self.regen_rate = self.base_regen_rate * penalty

        if self.regen_rate > 0:
            self.amount = min(self.max_capacity, self.amount + self.regen_rate)

    def harvest(self, amount: int) -> int:
        collected = min(self.amount, amount)
        self.amount -= collected
        self.model.total_mined_energy += collected

        if collected > 0:
            if self.resource_type == "renewable":
                self.model._mined_renewable_this_step += collected
            else:
                self.model._mined_nonrenewable_this_step += collected

        # Renewable fatigue / cooldown as before
        if self.resource_type == "renewable" and collected > 0:
            self.fatigue += collected
            if self.amount <= 0 or self.fatigue >= self.model.renewable_overuse_trigger:
                self.cooldown_remaining = max(self.cooldown_remaining, self.model.renewable_cooldown_steps)

        # --- NEW: nonrenewable spill -> scar nearby renewable tiles ---
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

        # --- Socialist tuning (safe defaults even for non-socialists) ---
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
        else:
            self.capitalist_step()
    
        # baseline upkeep
        self.energy -= 0.5
        if self.energy <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
            return

    # --- Capitalist behaviour (existing, kept) ---
    def capitalist_step(self) -> None:
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

                        # remove depleted nonrenewable patch from world
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

        # --- Targeting logic ---
        # 1. Look for closest nonrenewable patch with resources
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

        # 2. If no nonrenewable OR it's far, look for closest renewable
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

        # 3. Move towards target and mine when reached
        if best_patch_pos:
            speed = 2 if self.energy > 15 else 1
            self.move_towards(best_patch_pos, speed=speed)
            if self.pos == best_patch_pos:
                self.mining = True
                self.mining_counter = 3
                resources_here = [o for o in self.model.grid.get_cell_list_contents([self.pos]) if isinstance(o, ResourcePatch)]
                self.mining_target = resources_here[0] if resources_here else None

    # --- Socialist behaviour (new) ---
    def socialist_step(self) -> None:
        # Finish mining if already mining
        if self.mining:
            self.mining_counter -= 1
            if self.mining_counter <= 0:
                # resolve mining payoff on the tile we’re on
                cell = self.model.grid.get_cell_list_contents([self.pos])
                patch = next((o for o in cell if isinstance(o, ResourcePatch)), None)
                hub_here = any(isinstance(o, EnergyHub) and o.built for o in cell)

                if patch:
                    # setup + op costs + yield (hub reduces op_cost by 1 and adds +1 yield on renewables)
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

                    # Deplete nonrenewables fully remove
                    if patch.amount <= 0 and patch.resource_type == "nonrenewable":
                        self.model.grid.remove_agent(patch)
                        self.model.schedule.remove(patch)
                        if patch.pos in self.model.nonrenewable_locations:
                            try:
                                self.model.nonrenewable_locations.remove(patch.pos)
                            except ValueError:
                                pass

                    # ---- Tithe + wealth cap ----
                    if net_gain > 0:
                        tithe = self.model.tithe_rate * net_gain
                        tithe = max(0.0, min(tithe, self.energy))
                        self.energy -= tithe
                        self.model.community_pool += tithe

                    if self.energy > self.energy_cap:
                        skim = self.energy - self.energy_cap
                        self.energy -= skim
                        self.model.community_pool += skim

                self.mining = False
                self.mining_target = None

                # Share locally after mining
                self.redistribute_to_neighbors()
            return

        # ------- TARGET SELECTION -------
        # Emergency: go to nearest resource (speed > rules)
        if self.energy < self.emergency_floor:
            pos_r, dist_r, patch_r = self._nearest_patch("renewable")
            pos_n, dist_n, patch_n = self._nearest_patch("nonrenewable")
            if patch_r is None and patch_n is None:
                self.idle_wander(); return
            target = pos_r if (patch_n is None or (patch_r is not None and dist_r <= dist_n)) else pos_n
        else:
            # NEW: choose resource type based on average energy
            avg_e = self.model.average_energy()
            preferred = "renewable" if avg_e > 5.0 else "nonrenewable"

            pos_p, dist_p, patch_p = self._nearest_patch(preferred)
            if patch_p is None:
                # fallback to the other type if none available
                other = "nonrenewable" if preferred == "renewable" else "renewable"
                pos_p, dist_p, patch_p = self._nearest_patch(other)
                if patch_p is None:
                    self.idle_wander(); return
            target = pos_p

        # Move toward target
        if target:
            self.move_towards(target, speed=1)
            # cooperative build if we’re on a renewable patch with no hub
            if self.pos == target:
                self._maybe_build_hub_or_mine()
                return

        # Opportunistic sharing while travelling (very rich)
        if self.energy > (self.min_keep + 6):
            self.redistribute_to_neighbors()

    def capitalist_green_step(self):
        # Finish mining if already engaged
        if self.mining:
            self.mining_counter -= 1
            if self.mining_counter <= 0:
                cell = self.model.grid.get_cell_list_contents([self.pos])
                patch = next((o for o in cell if isinstance(o, ResourcePatch)), None)
                if patch:
                    # (No hubs for capitalists; same setup logic for renewables)
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

                    # Apply policy: tax nonrenewables, subsidize renewables
                    if gained > 0:
                        if patch.resource_type == "nonrenewable":
                            tax = self.model.carbon_tax_per_unit * gained
                            net -= tax
                            # send tax to community pool (or drop it if you prefer)
                            self.model.community_pool += max(0.0, tax)
                        else:
                            subsidy = self.model.renewable_subsidy_per_unit * gained
                            net += subsidy

                    self.energy += net
                    if net > 0:
                        self.total_collected_energy += net

                    # Remove depleted nonrenewables from world + cache
                    if patch.amount <= 0 and patch.resource_type == "nonrenewable":
                        self.model.grid.remove_agent(patch)
                        self.model.schedule.remove(patch)
                        if patch.pos in self.model.nonrenewable_locations:
                            try: self.model.nonrenewable_locations.remove(patch.pos)
                            except ValueError: pass

                self.mining = False
                self.mining_target = None
            return

        # Target selection: pick the patch with best green-adjusted score
        best_pos, best_score = None, -1e18

        # scan renewables
        for pos in list(self.model.renewable_locations):
            cell = self.model.grid.get_cell_list_contents([pos])
            patch = next((o for o in cell if isinstance(o, ResourcePatch) and o.amount > 0 and o.resource_type == "renewable"), None)
            if not patch: continue
            if any(isinstance(a, IdeologyAgent) for a in cell):  # avoid crowding
                continue
            s = self._green_profit_score(pos, patch)
            if s > best_score:
                best_pos, best_score = pos, s

        # scan nonrenewables
        for pos in list(self.model.nonrenewable_locations):
            cell = self.model.grid.get_cell_list_contents([pos])
            patch = next((o for o in cell if isinstance(o, ResourcePatch) and o.amount > 0 and o.resource_type == "nonrenewable"), None)
            if not patch: continue
            if any(isinstance(a, IdeologyAgent) for a in cell):
                continue
            s = self._green_profit_score(pos, patch)
            if s > best_score:
                best_pos, best_score = pos, s

        # Move & mine
        if best_pos:
            self.move_towards(best_pos, speed=2 if self.energy > 15 else 1)
            if self.pos == best_pos:
                self.mining = True
                self.mining_counter = 3
                self.mining_target = next((o for o in self.model.grid.get_cell_list_contents([self.pos]) if isinstance(o, ResourcePatch)), None)
        else:
            self.idle_wander()

    def socialist_green_step(self):
        # Finish mining
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

                    # Redistribution logic
                    if net_gain > 0:
                        tithe = self.model.tithe_rate * net_gain
                        tithe = max(0.0, min(tithe, self.energy))
                        self.energy -= tithe
                        self.model.community_pool += tithe

                    if self.energy > self.energy_cap:
                        skim = self.energy - self.energy_cap
                        self.energy -= skim
                        self.model.community_pool += skim

                self.mining = False
                self.mining_target = None
                self.redistribute_to_neighbors()
            return

        # Target selection: pick highest green-adjusted score but still follow socialist "safety" rule
        avg_energy = self.model.average_energy()
        best_pos, best_score = None, -1e18
        for res_type in ["renewable", "nonrenewable"]:
            # socialist logic: prefer renewable if avg energy high
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

        # Move & act
        if best_pos:
            self.move_towards(best_pos, speed=1)
            if self.pos == best_pos:
                self._maybe_build_hub_or_mine()
        else:
            self.idle_wander()

        # Opportunistic sharing while moving
        if self.energy > (self.min_keep + 6):
            self.redistribute_to_neighbors()

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

    # ---------- Socialist helpers ----------
    def _maybe_build_hub_or_mine(self):
        cell = self.model.grid.get_cell_list_contents([self.pos])
        patch = next((o for o in cell if isinstance(o, ResourcePatch)), None)
        if not patch:
            return
        hub_here = any(isinstance(o, EnergyHub) and o.built for o in cell)

        if patch.resource_type == "renewable" and not hub_here:
            # signal and look for partner socialist at same tile
            self.intent_build = True
            partners = [a for a in self.model.grid.get_cell_list_contents([self.pos])
                        if isinstance(a, IdeologyAgent) and a.ideology == "socialist" and getattr(a, "intent_build", False)]
            if len(partners) >= 2 and self.energy >= 3:
                self.build_counter += 1
                self.energy -= 0.2  # light build drain
                if self.build_counter >= self.coop_build_time:
                    hub = EnergyHub(self.model.next_id(), self.model, self.pos)
                    self.model.grid.place_agent(hub, self.pos)
                    self.model.schedule.add(hub)
                    self.energy -= 2.0  # joint build cost (each agent pays when it completes)
                    self.build_counter = 0
                    self.intent_build = False
                    return
                else:
                    return  # keep building next step
            else:
                self.build_counter = 0  # wait for partner; if none arrives, fall through

        # no building: start mining
        self.mining = True
        self.mining_counter = 3
        self.mining_target = patch

    def idle_wander(self):
        cx, cy = (self.model.width // 2, self.model.height // 2)
        self.move_towards((cx, cy), speed=1)

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
            # renewable bias: treat as if closer
            if rtype == "renewable":
                d = max(0, d - self.renewable_bias)
            if best_dist is None or d < best_dist:
                best_pos, best_dist, best_patch = pos, d, patch
        return best_pos, (best_dist if best_dist is not None else 10**9), best_patch

    def redistribute_to_neighbors(self):
        surplus = max(0.0, self.energy - self.min_keep)
        if surplus <= 0:
            return
        give_pool = self.share_fraction * surplus
        neighs = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=self.share_radius)
        needy = []
        for p in neighs:
            for a in self.model.grid.get_cell_list_contents([p]):
                if isinstance(a, IdeologyAgent) and a is not self and a.energy < self.help_threshold:
                    needy.append(a)
        if not needy:
            return
        deficits = [self.help_threshold - a.energy for a in needy]
        total_def = sum(deficits)
        if total_def <= 0:
            return
        for a, d in zip(needy, deficits):
            share_i = give_pool * (d / total_def)
            a.energy += share_i
            self.energy -= share_i
            if self.energy <= self.min_keep:
                break

    # --- movement helpers (kept from your file) ---
    def move_towards(self, target_pos, speed: int = 1) -> None:
        curr_x, curr_y = self.pos
        target_x, target_y = target_pos
        dx = target_x - curr_x
        dy = target_y - curr_y
        for _ in range(speed):
            step_x = curr_x + (1 if dx > 0 else -1 if dx < 0 else 0)
            step_y = curr_y + (1 if dy > 0 else -1 if dy < 0 else 0)
            new_pos = (step_x, step_y)
            if not self.model.grid.out_of_bounds(new_pos):
                self.model.grid.move_agent(self, new_pos)
                curr_x, curr_y = new_pos
                dx = target_x - curr_x
                dy = target_y - curr_y
            else:
                break

    def manhattan_distance(self, p1, p2) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def _green_profit_score(self, pos, patch) -> float:
        """
        Estimate profit after policy (tax/subsidy) and distance; penalize highly scarred renewables.
        Higher is better.
        """
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

