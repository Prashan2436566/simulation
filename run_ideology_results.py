#python run_ideology_results.py --steps 1500 --seeds 10 --agents 15 --width 30 --height 30 --out results_ideologies_10runs

"""
run_ideology_results.py

Run ideological simulations (capitalist, green_capitalist, socialist, green_socialist, communist, green_communist)
and export averaged data and figures for:
  - AgentsAlive (proxy for lifespan)
  - AvgEnergy
  - CommunityPool (excluded for capitalist & green_capitalist)
  - GiniEnergy
  - TotalScar
  - MinedRenewable
  - MinedNonrenewable

It runs each ideology multiple times (using --seeds) and averages results.
Outputs per-ideology CSVs (with conditional columns), plots, and a cross-ideology summary.

Usage:
    python run_ideology_results.py --steps 1500 --seeds 10 --agents 15 --width 30 --height 30 --out results_ideologies
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

# --- Project imports ---
try:
    from model.model import IdeologyModel
except Exception:
    print("[ERROR] Could not import from 'model'. Run this from project root.")
    print("        Expected: from model.model import IdeologyModel")
    raise

IDEOLOGIES = [
    "capitalist",
    "green_capitalist",
    "socialist",
    "green_socialist",
    "communist",
    "green_communist",
]

# Time-series metrics expected in the model DataCollector
TS_METRICS = [
    "AgentsAlive",
    "AvgEnergy",
    "CommunityPool",         # may not exist/used for capitalist variants
    "GiniEnergy",
    "TotalScar",
    "MinedRenewable",
    "MinedNonrenewable",
]

POOLLESS = {"capitalist", "green_capitalist"}  # ideologies without a community pool


def run_one(ideology: str, steps: int, width: int, height: int, num_agents: int, seed: int):
    """
    Run a single simulation and return:
      - df_ts: time-series DataFrame of model reporters
      - avg_lifespan: mean lifespan (steps alive) of agents present at start
    """
    np.random.seed(seed)
    model = IdeologyModel(
        width=width,
        height=height,
        num_agents=num_agents,
        renewables_regenerate=True,
        ideology=ideology,
    )

    # Track how long each starting agent stays alive (age in steps)
    ages = {}
    for a in list(model.schedule.agents):
        if hasattr(a, "energy"):
            ages[a.unique_id] = 0

    for _ in range(steps):
        model.step()
        for a in list(model.schedule.agents):
            if hasattr(a, "energy"):
                ages[a.unique_id] = ages.get(a.unique_id, 0) + 1

    # Collect reporters
    df_ts = model.datacollector.get_model_vars_dataframe()

    # Ensure all expected columns exist for alignment
    for col in TS_METRICS:
        if col not in df_ts.columns:
            df_ts[col] = np.nan

    avg_lifespan = np.mean(list(ages.values())) if ages else 0.0
    return df_ts, avg_lifespan


def aggregate_over_seeds(ideology: str, steps: int, width: int, height: int, num_agents: int, seeds: int):
    """
    Run multiple seeds and average the time-series per timestep.
    Returns:
        df_mean_std: DataFrame with columns [metrics..., metrics_std...]
        avg_lifespan_mean: float
        avg_lifespan_std: float
    """
    runs = []
    lifespans = []
    for s in range(seeds):
        df_ts, avg_ls = run_one(ideology, steps, width, height, num_agents, seed=1000 + s)
        runs.append(df_ts[TS_METRICS].reset_index(drop=True))
        lifespans.append(avg_ls)

    # Align by min length (should be uniform anyway)
    L = min(len(df) for df in runs)
    runs = [df.iloc[:L].copy() for df in runs]

    arr = np.stack([df.values for df in runs], axis=0)  # (seeds, T, metrics)
    mean_arr = arr.mean(axis=0)
    std_arr = arr.std(axis=0)

    df_mean = pd.DataFrame(mean_arr, columns=TS_METRICS)
    df_std = pd.DataFrame(std_arr, columns=[f"{m}_std" for m in TS_METRICS])
    out = pd.concat([df_mean, df_std], axis=1)
    out.index.name = "timestep"

    return out, float(np.mean(lifespans)), float(np.std(lifespans))


def plot_one_timeseries(df_mean: pd.DataFrame, ideology: str, outdir: str):
    """
    Save one figure per metric for the ideology.
    Skips CommunityPool for capitalist and green_capitalist.
    """
    skip_pool = ideology in POOLLESS
    for metric in TS_METRICS:
        if skip_pool and metric == "CommunityPool":
            continue
        plt.figure(figsize=(8, 4.5))
        y = df_mean[metric].to_numpy()
        x = np.arange(len(y))
        plt.plot(x, y)
        plt.xlabel("Timestep")
        plt.ylabel(metric)
        plt.title(f"{ideology}: {metric} over time")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{ideology}_{metric}.png"), dpi=150)
        plt.close()


def plot_cross_ideology_bar(values_dict: dict, title: str, ylabel: str, outpath: str):
    """Simple bar chart from a dict of {name: value}."""
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


def save_csv_conditionally(df_mean: pd.DataFrame, ideology: str, out_csv_path: str):
    """
    Save the per-ideology timeseries CSV.
    Drops CommunityPool columns entirely for capitalist & green_capitalist.
    Keeps std columns aligned with dropped metrics as well.
    """
    if ideology in POOLLESS:
        cols = [c for c in df_mean.columns if not (c == "CommunityPool" or c == "CommunityPool_std")]
        df_mean.loc[:, cols].to_csv(out_csv_path)
    else:
        df_mean.to_csv(out_csv_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500, help="Timesteps per run")
    ap.add_argument("--seeds", type=int, default=3, help="Independent runs to average")
    ap.add_argument("--agents", type=int, default=15, help="Agents per run")
    ap.add_argument("--width", type=int, default=30)
    ap.add_argument("--height", type=int, default=30)
    ap.add_argument("--out", type=str, default=None, help="Output directory")
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.out or f"results_ideologies_{timestamp}"
    os.makedirs(outdir, exist_ok=True)

    # Save run config
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(
            dict(
                steps=args.steps,
                seeds=args.seeds,
                agents=args.agents,
                width=args.width,
                height=args.height,
                ideologies=IDEOLOGIES,
                poolless=list(POOLLESS),
                metrics=TS_METRICS,
            ),
            f,
            indent=2,
        )

    # Aggregates for cross-ideology comparisons
    lifespan_means, lifespan_stds = {}, {}
    final_avg_energy, final_gini, final_pool = {}, {}, {}
    final_total_scar, final_mined_ren, final_mined_nonren = {}, {}, {}

    # Per-ideology runs
    for ideol in IDEOLOGIES:
        print(f"[RUN] {ideol}")
        df_mean, ls_mean, ls_std = aggregate_over_seeds(
            ideology=ideol,
            steps=args.steps,
            width=args.width,
            height=args.height,
            num_agents=args.agents,
            seeds=args.seeds,
        )

        # Save per-ideology CSV (conditionally drop pool columns)
        csv_path = os.path.join(outdir, f"{ideol}_timeseries.csv")
        save_csv_conditionally(df_mean, ideol, csv_path)

        # Per-ideology plots (skip pool where applicable)
        plot_one_timeseries(df_mean, ideol, outdir)

        # Collect final-step headline values
        lifespan_means[ideol] = ls_mean
        lifespan_stds[ideol] = ls_std

        final_avg_energy[ideol] = float(df_mean["AvgEnergy"].iloc[-1])
        final_gini[ideol] = float(df_mean["GiniEnergy"].iloc[-1])

        # Community pool: not applicable to capitalist variants
        if ideol in POOLLESS:
            final_pool[ideol] = np.nan
        else:
            final_pool[ideol] = float(df_mean["CommunityPool"].iloc[-1])

        # Newly added sustainability and extraction metrics
        final_total_scar[ideol] = float(df_mean["TotalScar"].iloc[-1])
        final_mined_ren[ideol] = float(df_mean["MinedRenewable"].iloc[-1])
        final_mined_nonren[ideol] = float(df_mean["MinedNonrenewable"].iloc[-1])

    # Cross-ideology plots
    plot_cross_ideology_bar(
        lifespan_means,
        "Average Agent Lifespan",
        "Avg Lifespan (steps)",
        os.path.join(outdir, "lifespan_comparison.png"),
    )
    plot_cross_ideology_bar(
        final_avg_energy,
        "Final AvgEnergy",
        "AvgEnergy (final timestep)",
        os.path.join(outdir, "avg_energy_final_comparison.png"),
    )
    plot_cross_ideology_bar(
        final_gini,
        "Final GiniEnergy",
        "GiniEnergy (final timestep)",
        os.path.join(outdir, "gini_final_comparison.png"),
    )

    # Community pool comparison (remove capitalist entries)
    filtered_pool = {k: v for k, v in final_pool.items() if k not in POOLLESS}
    plot_cross_ideology_bar(
        filtered_pool,
        "Final CommunityPool",
        "CommunityPool (final timestep)",
        os.path.join(outdir, "pool_final_comparison.png"),
    )

    # New cross-ideology comparisons
    plot_cross_ideology_bar(
        final_total_scar,
        "Final TotalScar",
        "TotalScar (final timestep)",
        os.path.join(outdir, "scar_final_comparison.png"),
    )
    plot_cross_ideology_bar(
        final_mined_ren,
        "Final MinedRenewable",
        "MinedRenewable (final timestep)",
        os.path.join(outdir, "mined_renewable_final_comparison.png"),
    )
    plot_cross_ideology_bar(
        final_mined_nonren,
        "Final MinedNonrenewable",
        "MinedNonrenewable (final timestep)",
        os.path.join(outdir, "mined_nonrenewable_final_comparison.png"),
    )

    # Summary table (now includes scar & mined metrics)
    summary_rows = []
    for ideol in IDEOLOGIES:
        summary_rows.append(
            {
                "ideology": ideol,
                "avg_lifespan": lifespan_means[ideol],
                "avg_lifespan_std": lifespan_stds[ideol],
                "final_AvgEnergy": final_avg_energy[ideol],
                "final_GiniEnergy": final_gini[ideol],
                "final_CommunityPool": final_pool[ideol],             # NaN for POOLLESS ideologies
                "final_TotalScar": final_total_scar[ideol],
                "final_MinedRenewable": final_mined_ren[ideol],
                "final_MinedNonrenewable": final_mined_nonren[ideol],
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("ideology")
    summary_df.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    print(f"[OK] Results saved to {outdir}")


if __name__ == "__main__":
    main()