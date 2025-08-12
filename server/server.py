from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from model.model import IdeologyModel, LegendElement, StatsElement
from model.agent import IdeologyAgent, ResourcePatch, EnergyHub
from mesa.visualization.modules import ChartModule


def agent_portrayal(agent):
    if isinstance(agent, IdeologyAgent):
        colour_map = {
        "capitalist": "blue",
        "socialist": "orange",
        "green_capitalist": "teal",
        "green_socialist": "brown",
        "communist": "red",
        "adaptive": "purple",
        }

        color = colour_map.get(getattr(agent, "ideology", "capitalist"), "blue")
        energy_norm = min(max(agent.energy / 10, 0), 1) if hasattr(agent, "energy") else 0.5
        return {
            "Shape": "circle",
            "Color": color,
            "Filled": "true",
            "Layer": 1,
            "r": 0.6 + 0.2 * energy_norm,
        }

    if isinstance(agent, ResourcePatch):
        # Base color by type
        color = "grey" if agent.resource_type == "nonrenewable" else "lightgreen"
        # NEW: display scar level on renewables (simple text cue)
        label = ""
        if agent.resource_type == "renewable":
            s = getattr(agent, "scar_level", 0.0)
            label = f"{s:.1f}" if s >= 0.5 else ""
        return {
            "Shape": "rect",
            "Color": color,
            "Filled": "true",
            "Layer": 0,
            "w": 1,
            "h": 1,
            "text": label,
            "text_color": "black",
        }

    if isinstance(agent, EnergyHub):
        return {
            "Shape": "rect",
            "Color": "orange",
            "Filled": "false",
            "Layer": 2,
            "w": 0.9,
            "h": 0.9,
            "text": "S",
            "text_color": "orange",
        }

    return None

grid = CanvasGrid(agent_portrayal, 30, 30, 500, 500)
chart_econ = ChartModule(
    [
        {"Label": "AvgEnergy", "Color": "blue"},
        {"Label": "CommunityPool", "Color": "orange"},
        {"Label": "AgentsAlive", "Color": "green"},
    ],
    data_collector_name="datacollector",
)

chart_env = ChartModule(
    [
        {"Label": "TotalScar", "Color": "red"},
        {"Label": "Renewables", "Color": "darkgreen"},
        {"Label": "NonRenewables", "Color": "gray"},
        {"Label": "InfrastructureSites", "Color": "black"},
    ],
    data_collector_name="datacollector",
)

# Optional 3rd chart (per-ideology counts) — only if you uncommented the reporters in model.py
chart_ideo = ChartModule(
     [
         {"Label": "Ideology_capitalist", "Color": "blue"},
         {"Label": "Ideology_green_capitalist", "Color": "teal"},
         {"Label": "Ideology_socialist", "Color": "orange"},
         {"Label": "Ideology_green_socialist", "Color": "brown"},
     ],
    data_collector_name="datacollector",
)

chart_mining = ChartModule(
    [
        {"Label": "MinedRenewable", "Color": "darkgreen"},
        {"Label": "MinedNonrenewable", "Color": "gray"},
    ],
    data_collector_name="datacollector",
)
chart_gini = ChartModule(
    [
        {"Label": "GiniEnergy", "Color": "purple"},
    ],
    data_collector_name="datacollector",
)


model_params = {
    "width": 30,
    "height": 30,
    "num_agents": UserSettableParameter("slider", "Number of agents", 15, 1, 50, 1),
    "renewables_regenerate": UserSettableParameter("checkbox", "Renewables regenerate", True),
    "ideology": "capitalist",  # will be overridden from main.py
    # Economics
    "cost_renewable_setup": UserSettableParameter("slider", "Setup cost (renewable)", 5.0, 0.0, 20.0, 0.5),
    "cost_extract_renewable": UserSettableParameter("slider", "Op cost per mine (renewable)", 1.0, 0.0, 10.0, 0.5),
    "cost_extract_nonrenewable": UserSettableParameter("slider", "Op cost per mine (nonrenewable)", 2.0, 0.0, 10.0, 0.5),
    "yield_per_mine_renewable": UserSettableParameter("slider", "Yield per mine (renewable)", 4, 1, 10, 1),
    "yield_per_mine_nonrenewable": UserSettableParameter("slider", "Yield per mine (nonrenewable)", 6, 1, 10, 1),
    # Overuse dynamics
    "renewable_cooldown_steps": UserSettableParameter("slider", "Renewable cooldown steps", 5, 0, 20, 1),
    "renewable_overuse_trigger": UserSettableParameter("slider", "Overuse trigger (units)", 6, 1, 20, 1),
    "renewable_fatigue_decay": UserSettableParameter("slider", "Fatigue decay per step", 1, 0, 5, 1),
}

server = ModularServer(
    IdeologyModel,
    [grid, LegendElement(), StatsElement(), chart_econ, chart_env, chart_ideo, chart_mining, chart_gini],
    "Sociopolitical Ideologies Simulation",
    model_params,
)

