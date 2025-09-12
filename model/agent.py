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

        # Environment
        self.scar_level: float = 0.0

        # Degradation (offline until repaired)
        self.is_degraded: bool = False

    # ----- degrade helpers -----
    def mark_degraded(self):
        self.is_degraded = True

    def clear_degraded(self):
        self.is_degraded = False

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
            self.capitalist_step()
        else:
            self.capitalist_step()

        # baseline upkeep
        self.energy -= 0.5
        if self.energy <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
            return

    # ---------- MAINTENANCE HELPERS ----------
    def _nearest_degraded_patch(self):
        """Return (pos, dist, patch) for closest degraded renewable."""
        best = (None, 10**9, None)
        for pos in list(self.model.renewable_locations):
            cell = self.model.grid.get_cell_list_contents([pos])
            patch = next((o for o in cell if isinstance(o, ResourcePatch) and o.resource_type == "renewable"), None)
            if not patch or not getattr(patch, "is_degraded", False):
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
        # Capitalist: only if close and no nearby nonrenewables, and with buffer
        if self.ideology == "capitalist":
            nearby_non = any(self.manhattan_distance(self.pos, pos) <= 5 for pos in self.model.nonrenewable_locations)
            return (not nearby_non) and (self.energy >= cost + 5)

        # Green-capitalist: more willing (or if area is scarred)
        if self.ideology == "green_capitalist":
            nearby_non = any(self.manhattan_distance(self.pos, pos) <= 5 for pos in self.model.nonrenewable_locations)
            local_scar = getattr(patch, "scar_level", 0.0)
            return (not nearby_non or local_scar >= 0.5) and (self.energy >= cost + 5)

        # Socialist: maintain the commons if community is struggling or I'm comfy
        if self.ideology == "socialist":
            floor = getattr(self.model, "pool_floor", 10.0)
            return (avg_e <= 6.0) or (self.energy >= max(floor, cost + 2))

        # Green-socialist: most willing as long as above safety floor
        if self.ideology == "green_socialist":
            floor = getattr(self.model, "pool_floor", 10.0)
            return self.energy >= max(floor, cost)

        # default conservative
        return False

    def _maybe_repair_instant(self) -> bool:
        """
        If on a degraded renewable and ideology says yes, spend energy and repair instantly.
        Returns True if we repaired (or moved to repair in this tick).
        """
        # If we're standing on a renewable, try to repair immediately
        cell = self.model.grid.get_cell_list_contents([self.pos])
        patch_here = next((o for o in cell if isinstance(o, ResourcePatch) and o.resource_type == "renewable"), None)
        if patch_here and getattr(patch_here, "is_degraded", False) and self._should_repair(patch_here):
            cost = getattr(self.model, "repair_energy_cost", 10.0)
            if self.energy >= cost:
                self.energy -= cost
                patch_here.clear_degraded()
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

    def _nearest_patch(self, rtype: str):
        locs = self.model.renewable_locations if rtype == "renewable" else self.model.nonrenewable_locations
        best_pos, best_dist, best_patch = None, None, None
        for pos in list(locs):
            cell = self.model.grid.get_cell_list_contents([pos])
            patch = next((o for o in cell if isinstance(o, ResourcePatch) and o.resource_type == rtype and o.amount > 0), None)
            if not patch:
                continue
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
