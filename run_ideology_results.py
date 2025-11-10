#python run_ideology_results.py --steps 1500 --seeds 10 --agents 15 --width 30 --height 30 --out results_ideologies_10runs

#!/usr/bin/env python3
"""
run_ideology_results.py

Run ideological simulations (capitalist, green_capitalist, socialist, green_socialist, communist, green_communist)
and export averaged data and figures for:
  - AgentsAlive (proxy for lifespan)
  - AvgEnergy
  - CommunityPool
  - GiniEnergy

It runs each ideology multiple times (using --seeds) and averages results.
Outputs CSVs, plots, and summary statistics.

Usage:
    python run_ideology_results.py --steps 1500 --seeds 10 --agents 15 --width 30 --height 30 --out results_ideologies
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

# --- Project imports ---
try:
    from model.model import IdeologyModel
except Exception as e:
    print("[ERROR] Could not import from 'model'. Make sure you run this from your project root.")
    print("        Tried: from model.model import IdeologyModel")
    raise

IDEOLOGIES = [
    "capitalist",
    "green_capitalist",
    "socialist",
    "green_socialist",
    "communist",
    "green_communist",
]

TS_METRICS = ["AgentsAlive", "AvgEnergy", "CommunityPool", "GiniEnergy"]


def run_one(ideology, steps, width, height, num_agents, seed):
    """Run a single simulation and return (df_ts, avg_lifespan)."""
    np.random.seed(seed)
    model = IdeologyModel(
        width=width,
        height=height,
        num_agents=num_agents,
        renewables_regenerate=True,
        ideology=ideology,
    )

    ages = {}
    for a in list(model.schedule.agents):
        if hasattr(a, "energy"):
            ages[a.unique_id] = 0

    for t in range(steps):
        model.step()
        for a in list(model.schedule.agents):
            if hasattr(a, "energy"):
                ages[a.unique_id] = ages.get(a.unique_id, 0) + 1

    df_ts = model.datacollector.get_model_vars_dataframe()

    for col in TS_METRICS:
        if col not in df_ts.columns:
            df_ts[col] = np.nan

    avg_lifespan = np.mean(list(ages.values())) if ages else 0.0
    return df_ts, avg_lifespan


def aggregate_over_seeds(ideology, steps, width, height, num_agents, seeds):
    """Run multiple seeds and average results."""
    runs, lifespans = [], []
    for s in range(seeds):
        df_ts, avg_ls = run_one(ideology, steps, width, height, num_agents, seed=1000 + s)
        runs.append(df_ts[TS_METRICS].reset_index(drop=True))
        lifespans.append(avg_ls)

    L = min(len(df) for df in runs)
    runs = [df.iloc[:L].copy() for df in runs]
    arr = np.stack([df.values for df in runs], axis=0)
    mean_arr = arr.mean(axis=0)
    std_arr = arr.std(axis=0)

    df_mean = pd.DataFrame(mean_arr, columns=TS_METRICS)
    df_std = pd.DataFrame(std_arr, columns=[f"{m}_std" for m in TS_METRICS])
    out = pd.concat([df_mean, df_std], axis=1)
    out.index.name = "timestep"

    return out, float(np.mean(lifespans)), float(np.std(lifespans))


def plot_one_timeseries(df_mean, ideology, outdir):
    """Plot one figure per metric."""
    for metric in TS_METRICS:
        plt.figure(figsize=(8, 4.5))
        y = df_mean[metric].to_numpy()
        x = np.arange(len(y))
        plt.plot(x, y, label=metric)
        plt.xlabel("Timestep")
        plt.ylabel(metric)
        plt.title(f"{ideology}: {metric} over time")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{ideology}_{metric}.png"), dpi=150)
        plt.close()


def plot_cross_ideology_bar(values_dict, title, ylabel, outpath):
    names = list(values_dict.keys())
    vals = [values_dict[k] for k in names]
    plt.figure(figsize=(9, 5))
    x = np.arange(len(names))
    plt.bar(x, vals)
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--agents", type=int, default=15)
    ap.add_argument("--width", type=int, default=30)
    ap.add_argument("--height", type=int, default=30)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.out or f"results_ideologies_{timestamp}"
    os.makedirs(outdir, exist_ok=True)

    # Save config
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(
            dict(
                steps=args.steps,
                seeds=args.seeds,
                agents=args.agents,
                width=args.width,
                height=args.height,
                ideologies=IDEOLOGIES,
            ),
            f,
            indent=2,
        )

    lifespan_means, lifespan_stds = {}, {}
    final_avg_energy, final_gini, final_pool = {}, {}, {}

    for ideol in IDEOLOGIES:
        print(f"[RUN] {ideol}")
        df_mean, ls_mean, ls_std = aggregate_over_seeds(
            ideol, args.steps, args.width, args.height, args.agents, args.seeds
        )

        csv_path = os.path.join(outdir, f"{ideol}_timeseries.csv")
        df_mean.to_csv(csv_path)
        plot_one_timeseries(df_mean, ideol, outdir)

        lifespan_means[ideol] = ls_mean
        lifespan_stds[ideol] = ls_std
        final_avg_energy[ideol] = float(df_mean["AvgEnergy"].iloc[-1])
        final_gini[ideol] = float(df_mean["GiniEnergy"].iloc[-1])
        final_pool[ideol] = float(df_mean["CommunityPool"].iloc[-1])

    # Cross-ideology plots
    plot_cross_ideology_bar(lifespan_means, "Average Agent Lifespan", "Avg Lifespan (steps)",
                            os.path.join(outdir, "lifespan_comparison.png"))
    plot_cross_ideology_bar(final_avg_energy, "Final AvgEnergy", "AvgEnergy (final timestep)",
                            os.path.join(outdir, "avg_energy_final_comparison.png"))
    plot_cross_ideology_bar(final_gini, "Final GiniEnergy", "GiniEnergy (final timestep)",
                            os.path.join(outdir, "gini_final_comparison.png"))
    plot_cross_ideology_bar(final_pool, "Final CommunityPool", "CommunityPool (final timestep)",
                            os.path.join(outdir, "pool_final_comparison.png"))

    summary_rows = []
    for ideol in IDEOLOGIES:
        summary_rows.append({
            "ideology": ideol,
            "avg_lifespan": lifespan_means[ideol],
            "avg_lifespan_std": lifespan_stds[ideol],
            "final_AvgEnergy": final_avg_energy[ideol],
            "final_GiniEnergy": final_gini[ideol],
            "final_CommunityPool": final_pool[ideol],
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("ideology")
    summary_df.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    print(f"[OK] Results saved to {outdir}")


if __name__ == "__main__":
    main()
