#!/usr/bin/env python3
"""
Evaluate a trained SB3 DQN on the Mesa environment and export time-series
metrics similar to run_ideology_results.py, specifically:

  - AgentsAlive
  - AvgEnergy
  - GiniEnergy
  - TotalScar
  - MinedRenewable
  - MinedNonrenewable

Outputs:
  • Per-episode CSVs (step-indexed timeseries)
  • Mean±Std CSV across episodes (aligned by step, padding with last value)
  • Line plots per metric (mean with ±std band)

Usage:
  python eval_dqn_timeseries.py --model models/dqn_sb3_final.zip --vecnorm models/vecnorm.pkl --episodes 10 --width 30 --height 30 --agents 15 --max_steps 400 --out dqn_results_runlike
"""

import os, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mesa_sb3_env import MesaSB3Env


METRICS = [
    "AgentsAlive",
    "AvgEnergy",
    "GiniEnergy",
    "TotalScar",
    "MinedRenewable",
    "MinedNonrenewable",
]

def make_env(width=30, height=30, num_agents=15, max_steps=1500):
    return MesaSB3Env(width, height, num_agents, True, "adaptive", max_steps)

def _get_latest_metrics_from_model(model_obj):
    """
    Read the latest row from the model.datacollector for the metrics we care about.
    Returns (dict metric->value, step_index) where step_index is 0-based.
    """
    df = model_obj.datacollector.get_model_vars_dataframe()
    if len(df) == 0:
        # Before first step, return NaNs
        return {k: np.nan for k in METRICS}, 0
    row = df.iloc[-1]
    out = {}
    for k in METRICS:
        out[k] = float(row[k]) if k in df.columns else np.nan
    # DataCollector uses the row index as timestep (starts at 0)
    step_idx = int(df.index[-1]) if df.index.dtype.kind in ("i", "u") else len(df) - 1
    return out, step_idx

def _get_base_model_from_vecenv(venv):
    """
    Pull the underlying Mesa model from a VecNormalize(DummyVecEnv(...)).
    Uses get_attr pass-through to access sub-env attributes.
    """
    try:
        models = venv.get_attr("model")
        return models[0]
    except Exception:
        # Fallback: drill down (VecNormalize -> DummyVecEnv -> envs[0])
        try:
            base_env = venv.venv.envs[0]
            return getattr(base_env, "model", None)
        except Exception:
            return None

def run_one_episode(model, venv, episode_id, outdir):
    """
    Run one episode and return a per-step DataFrame with the chosen metrics.
    Also saves a CSV for this episode.
    """
    obs = venv.reset()
    rows = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)

        base_model = _get_base_model_from_vecenv(venv)
        if base_model is None:
            raise RuntimeError("Could not access underlying Mesa model from VecEnv.")

        metrics, step_idx = _get_latest_metrics_from_model(base_model)
        row = {"timestep": step_idx}
        row.update(metrics)
        rows.append(row)

        if bool(dones[0]):
            break

    df = pd.DataFrame(rows).sort_values("timestep").reset_index(drop=True)
    ep_csv = os.path.join(outdir, f"episode_{episode_id:02d}.csv")
    df.to_csv(ep_csv, index=False)
    print(f"[OK] Episode {episode_id} saved -> {ep_csv}")
    return df

def align_and_aggregate(dfs, max_steps):
    """
    Align each episode's timeseries to length max_steps by forward-filling
    the last observed value. Then compute mean and std per step.
    """
    def pad_forward(series, T):
        if len(series) == 0:
            return np.full(T, np.nan)
        if len(series) >= T:
            return series.iloc[:T].to_numpy()
        last = series.iloc[-1]
        pad = np.full(T - len(series), last)
        return np.concatenate([series.to_numpy(), pad])

    # Build matrices [episodes x T] for each metric
    mats_mean = {}
    mats_std = {}
    xs = np.arange(max_steps)

    for m in METRICS:
        mat = []
        for df in dfs:
            # Ensure continuous timesteps starting at 0; if missing, forward-fill
            s = df[m]
            mat.append(pad_forward(s, max_steps))
        mat = np.vstack(mat)  # [E, T]
        mats_mean[m] = np.nanmean(mat, axis=0)
        mats_std[m] = np.nanstd(mat, axis=0)

    mean_df = pd.DataFrame({"timestep": xs})
    std_df  = pd.DataFrame({"timestep": xs})
    for m in METRICS:
        mean_df[m] = mats_mean[m]
        std_df[m + "_std"] = mats_std[m]
    agg = pd.concat([mean_df, std_df.drop(columns=["timestep"])], axis=1)
    return agg

def plot_metric(mean_std_df, metric, outdir, title_prefix="DQN"):
    x = mean_std_df["timestep"].to_numpy()
    y = mean_std_df[metric].to_numpy()
    s = mean_std_df[metric + "_std"].to_numpy()
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, label=f"{metric} (mean)")
    plt.fill_between(x, y - s, y + s, alpha=0.2, label="±1 std")
    plt.xlabel("Timestep")
    plt.ylabel(metric)
    plt.title(f"{title_prefix}: {metric} over time (mean ± std)")
    plt.legend()
    plt.tight_layout()
    outpath = os.path.join(outdir, f"{metric}_timeseries.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"[OK] Plot saved -> {outpath}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dqn_sb3_final.zip")
    ap.add_argument("--vecnorm", default="models/vecnorm.pkl")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--width", type=int, default=30)
    ap.add_argument("--height", type=int, default=30)
    ap.add_argument("--agents", type=int, default=15)
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--out", type=str, default="dqn_results_runlike")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Save config
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Build eval env w/ VecNormalize stats (as in other eval scripts)
    venv = DummyVecEnv([lambda: make_env(args.width, args.height, args.agents, args.max_steps)])
    venv = VecNormalize.load(args.vecnorm, venv)
    venv.training = False
    venv.norm_reward = False

    model = DQN.load(args.model, env=venv, device="auto")

    # Run episodes, collect per-step DataFrames
    ep_dfs = []
    for ep in range(1, args.episodes + 1):
        df = run_one_episode(model, venv, ep, args.out)
        ep_dfs.append(df)

    # Aggregate as mean ± std across episodes (aligned by step)
    agg = align_and_aggregate(ep_dfs, args.max_steps)
    agg_csv = os.path.join(args.out, "timeseries_mean_std.csv")
    agg.to_csv(agg_csv, index=False)
    print(f"[OK] Mean±Std CSV -> {agg_csv}")

    # Plots per metric (line with ±std band), same idea as ideology script. :contentReference[oaicite:2]{index=2}
    for m in METRICS:
        plot_metric(agg, m, args.out, title_prefix="DQN")

    print(f"[DONE] Results saved in: {args.out}")

if __name__ == "__main__":
    main()
