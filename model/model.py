from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.visualization.modules import TextElement
from mesa.datacollection import DataCollector
import random

from .agent import IdeologyAgent, ResourcePatch, EnergyHub


class LegendElement(TextElement):
    def render(self, model):
        return """
        <div style='font-size:14px; line-height:1.25;'>
            <b>Legend</b><br>
            <span style='color:lightgreen;'>🟩 Renewable</span> 
            <span style='color:#ff0000;'>(D border = degraded)</span><br>
            <span style='color:gray;'>⬜ Non-renewable</span><br>
            <span style='color:blue;'>🔵 Capitalist</span>
            <span style='color:teal;'>🟢 Green Capitalist</span><br>
            <span style='color:orange;'>🟠 Socialist</span>
            <span style='color:brown;'>🟤 Green Socialist</span><br>
            <span style='color:orange;'>🟧 S</span> = Energy Hub (renewable)<br>
            <span style='color:red;'>🔴 Numbers</span> on green tiles = local scar (pollution)
        </div>
        """


class StatsElement(TextElement):
    def render(self, model):
        agents = [a for a in model.schedule.agents if hasattr(a, "energy")]
        n_agents = len(agents)
        avg_e = (sum(a.energy for a in agents) / n_agents) if n_agents else 0.0
        scars = 0.0
        for x in range(model.width):
            for y in range(model.height):
                for obj in model.grid.get_cell_list_contents((x, y)):
                    if isinstance(obj, ResourcePatch) and obj.resource_type == "renewable":
                        scars += getattr(obj, "scar_level", 0.0)

        return f"""
        <div style='font-size:14px; line-height:1.25;'>
            <b>Stats</b><br>
            Total Agents Created: {model.total_agents_created}<br>
            Living Agents: {n_agents}<br>
            Total Resources Mined (raw): {model.total_mined_energy:.2f}<br>
            Community Pool: {model.community_pool:.2f} (redistribute every {model.redistribute_every} steps)<br>
            Avg Energy: {avg_e:.2f}<br>
            Total Scar Level: {scars:.2f}
        </div>
        """


class IdeologyModel(Model):
    def __init__(
        self,
        width: int = 30,
        height: int = 30,
        num_agents: int = 15,
        renewables_regenerate: bool = True,
        ideology: str = "capitalist",
        policy_enabled: bool = False,
        # Economics
        cost_renewable_setup: float = 5.0,
        cost_extract_renewable: float = 1.0,
        cost_extract_nonrenewable: float = 2.0,
        yield_per_mine_renewable: int = 4,
        yield_per_mine_nonrenewable: int = 6,
        # Overuse dynamics
        renewable_cooldown_steps: int = 5,
        renewable_overuse_trigger: int = 6,
        renewable_fatigue_decay: int = 1,
        # Redistribution safety floor (used by socialist sharing + repair heuristics)
        pool_floor: float = 10.0,
        # Degradation controls (periodic chance to degrade one renewable)
        degrade_period: int = 10,
        degrade_chance: float = 0.5,
        # Instant repair cost (energy spent by a repairing agent)
        repair_energy_cost: float = 10.0,
    ) -> None:
        self.current_id = 0
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)
        self.running = True
        self.width = width
        self.height = height

        # Config
        self.renewables_regenerate = renewables_regenerate
        self.ideology = ideology
        self.policy_enabled = policy_enabled

        # Economics
        self.cost_renewable_setup = cost_renewable_setup
        self.cost_extract_renewable = cost_extract_renewable
        self.cost_extract_nonrenewable = cost_extract_nonrenewable
        self.yield_per_mine_renewable = yield_per_mine_renewable
        self.yield_per_mine_nonrenewable = yield_per_mine_nonrenewable

        # Renewable overuse dynamics
        self.renewable_cooldown_steps = renewable_cooldown_steps
        self.renewable_overuse_trigger = renewable_overuse_trigger
        self.renewable_fatigue_decay = renewable_fatigue_decay

        # Social policy knobs
        self.community_pool = 0.0
        self.tithe_rate = 0.30
        self.basic_income = 0.10
        self.redistribute_every = 5
        self.share_floor = 6.0
        self._since_last_redistribute = 0
        self.pool_floor = pool_floor  # agents won’t donate below this

        # Environmental scar mechanics
        self.scar_radius = 1
        self.scar_increase_per_unit = 0.15
        self.scar_decay = 0.02
        self.scar_regen_alpha = 0.5
        self.scar_max = 3.0
        self.scar_collapse_threshold = 1.0  # if scar >= this, renewable collapses

        # Green policy knobs
        self.carbon_tax_per_unit = 0.5
        self.renewable_subsidy_per_unit = 0.2
        self.scar_avoid_alpha = 0.4

        # Repair/degrade system
        self.degrade_period = degrade_period
        self.degrade_chance = degrade_chance
        self.repair_energy_cost = repair_energy_cost
        self.step_count = 0

        # Mining accounting (for charts)
        self._mined_renewable_this_step = 0
        self._mined_nonrenewable_this_step = 0
        self.mined_renewable_last_step = 0
        self.mined_nonrenewable_last_step = 0

        # Counters & caches
        self.total_agents_created = 0
        self.total_mined_energy = 0.0
        self.nonrenewable_locations: list[tuple[int, int]] = []
        self.renewable_locations: list[tuple[int, int]] = []

        # Scatter resources
        self._scatter_resources("renewable", 100)
        self._scatter_resources("nonrenewable", 100)

        #adaptive RL params
        self.max_adaptive_respawns = float("inf")
        #self.max_adaptive_respawns = 0
        self.q_init = 15.0  # optimistic initial Q for adaptive agents (≈ 1/(1-γ) with γ=0.95 → 20)

        # Totals that batch/visualization will read
        self.mined_renewable_total = 0.0
        self.mined_nonrenewable_total = 0.0

        # Optional DQN diagnostics (so they exist from step 0)
        self.dqn_replay_size = 0
        self.dqn_last_loss = float("nan")
        self.dqn_steps = 0
        # self.dqn_eps is created when DQN lazy-inits; that's fine


        # Spawn agents
        for _ in range(num_agents):
            agent = IdeologyAgent(self.next_id(), self, ideology)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)

        # Data collector for charts
        self.datacollector = DataCollector(
            model_reporters={
                "Ideology_adaptive": lambda m: sum(
                    1 for a in m.schedule.agents if getattr(a, "ideology", "") == "adaptive"
                ),
                "Ideology_adaptive_direct": lambda m: sum(
                    1 for a in m.schedule.agents if getattr(a, "ideology", "") == "adaptive_direct"
                ),
                "Ideology_communist": lambda m: sum(
                    1 for a in m.schedule.agents if getattr(a, "ideology", "") == "communist"
                ),
                "Ideology_green_communist": lambda m: sum(
                    1 for a in m.schedule.agents if getattr(a, "ideology", "") == "green_communist"
                ),



                "AvgEnergy": lambda m: m.average_energy(),
                "CommunityPool": lambda m: m.community_pool,
                "TotalScar": lambda m: sum(
                    getattr(obj, "scar_level", 0.0)
                    for x in range(m.width)
                    for y in range(m.height)
                    for obj in m.grid.get_cell_list_contents((x, y))
                    if isinstance(obj, ResourcePatch)
                ),
                "Renewables": lambda m: len(m.renewable_locations),
                "NonRenewables": lambda m: len(m.nonrenewable_locations),
                "InfrastructureSites": lambda m: 0,  # placeholder
                "AgentsAlive": lambda m: sum(1 for a in m.schedule.agents if hasattr(a, "energy")),
                "GiniEnergy": lambda m: m.gini_energy(),
                "AvgAdaptiveReward": lambda m: (
                    sum(getattr(a, "last_reward", 0.0) for a in m.schedule.agents
                        if getattr(a, "ideology", "") == "adaptive")
                    / max(1, sum(1 for a in m.schedule.agents if getattr(a, "ideology","")=="adaptive"))
                ),
                "MinedRenewable":    lambda m: m.mined_renewable_last_step,
                "MinedNonrenewable": lambda m: m.mined_nonrenewable_last_step,
                "AvgAdaptiveEpsilon": lambda m: (
                    sum(getattr(a, "rl_epsilon", 0.0) for a in m.schedule.agents
                        if getattr(a, "ideology", "") == "adaptive")
                    / max(1, sum(1 for a in m.schedule.agents if getattr(a, "ideology","")=="adaptive"))
                ),
                "AvgAdaptiveEpsilonGlobal": lambda m: getattr(m, "dqn_eps", float("nan")),


            }
        )
        self.datacollector.collect(self)
        self.shared_q_table = {}


    # ---------- world setup ----------
    def _scatter_resources(self, resource_type: str, num_patches: int) -> None:
        placed = 0
        while placed < num_patches:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if not any(isinstance(obj, ResourcePatch) for obj in self.grid.get_cell_list_contents((x, y))):
                regen = 1 if resource_type == "renewable" and self.renewables_regenerate else 0
                patch = ResourcePatch(self.next_id(), self, resource_type, max_capacity=5, regen_rate=regen)
                self.grid.place_agent(patch, (x, y))
                self.schedule.add(patch)
                if resource_type == "nonrenewable":
                    self.nonrenewable_locations.append((x, y))
                else:
                    self.renewable_locations.append((x, y))
                placed += 1

    # ---------- degrade control ----------
    def degrade_random_renewable(self):
        """Pick one healthy renewable and mark it degraded (if any)."""
        candidates = []
        for pos in list(self.renewable_locations):
            cell = self.grid.get_cell_list_contents([pos])
            patch = next((o for o in cell if isinstance(o, ResourcePatch) and o.resource_type == "renewable"), None)
            if patch and not getattr(patch, "is_degraded", False):
                candidates.append(patch)
        if candidates:
            self.random.choice(candidates).mark_degraded()

    # ---------- redistribution ----------
    def redistribute_pool(self):
        agents = [a for a in self.schedule.agents if hasattr(a, "energy")]
        needy = [a for a in agents if a.energy < self.share_floor]
        if not needy or self.community_pool <= 0:
            return
        deficits = [self.share_floor - a.energy for a in needy]
        total_deficit = sum(deficits)
        if total_deficit <= 0:
            return
        pool = self.community_pool
        for a, d in zip(needy, deficits):
            give = pool * (d / total_deficit)
            a.energy += give
            self.community_pool -= give

    # ---------- step ----------
    def step(self) -> None:
        # periodic degrade
        self.step_count += 1
        if self.degrade_period > 0:
            if self.step_count % self.degrade_period == 0 and self.random.random() < self.degrade_chance:
                self.degrade_random_renewable()

        # reset per-step mined counters
        self._mined_renewable_this_step = 0
        self._mined_nonrenewable_this_step = 0

        # basic income
        for a in list(self.schedule.agents):
            if hasattr(a, "energy"):
                a.energy += self.basic_income

        # periodic redistribution
        self._since_last_redistribute += 1
        if self._since_last_redistribute >= self.redistribute_every:
            self.redistribute_pool()
            self._since_last_redistribute = 0

        # advance
        self.schedule.step()

        # publish this step’s mined totals
        self.mined_renewable_last_step = self._mined_renewable_this_step
        self.mined_nonrenewable_last_step = self._mined_nonrenewable_this_step

        # collect for charts
        self.datacollector.collect(self)

    # ---------- utils ----------
    def next_id(self) -> int:
        self.current_id += 1
        return self.current_id

    def average_energy(self) -> float:
        vals = [a.energy for a in self.schedule.agents if hasattr(a, "energy")]
        return (sum(vals) / len(vals)) if vals else 0.0

    def gini_energy(self) -> float:
        vals = sorted([a.energy for a in self.schedule.agents if hasattr(a, "energy") and a.energy >= 0])
        n = len(vals)
        if n == 0:
            return 0.0
        total = sum(vals)
        if total == 0:
            return 0.0
        cum = 0.0
        for i, v in enumerate(vals, 1):
            cum += i * v
        return (2 * cum) / (n * total) - (n + 1) / n
