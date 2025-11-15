import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model.model import IdeologyModel
from sb3_shared_policy import SB3SharedPolicy


def run_single_episode(
    seed: int,
    policy: SB3SharedPolicy,
    width: int,
    height: int,
    num_agents: int,
    max_steps: int,
) -> pd.DataFrame:
    model = IdeologyModel(
        width=width,
        height=height,
        num_agents=num_agents,
        renewables_regenerate=True,
        ideology="adaptive",
        shared_policy=policy,
    )

    model.random.seed(seed)

    for _ in range(max_steps):
        model.step()

    df = model.datacollector.get_model_vars_dataframe()
    df = df.copy()
    df["Episode"] = seed
    df["Step"] = np.arange(len(df))
    return df


def plot_metric(
    steps: np.ndarray,
    mean_series: pd.Series,
    std_series: pd.Series,
    n_episodes: int,
    title: str,
    ylabel: str,
    out_path: str,
):
    ci_half_width = 1.96 * std_series.to_numpy() / np.sqrt(max(n_episodes, 1))
    ci_lower = mean_series.to_numpy() - ci_half_width
    ci_upper = mean_series.to_numpy() + ci_half_width

    plt.figure(figsize=(12, 6))

    plt.plot(steps, mean_series, label=f"{ylabel} (mean)")

    plt.fill_between(
        steps,
        mean_series - std_series,
        mean_series + std_series,
        alpha=0.3,
        label="±1 std",
    )

    plt.fill_between(
        steps,
        ci_lower,
        ci_upper,
        alpha=0.15,
        label="95% CI",
    )

    plt.xlabel("Timestep")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    policy = SB3SharedPolicy(
        model_path=args.model_path,
        vecnorm_path=args.vecnorm_path,
        device=args.device,
        deterministic=True,
    )

    all_dfs = []
    seeds = list(range(args.start_seed, args.start_seed + args.n_episodes))
    for seed in seeds:
        print(f"[INFO] Running episode with seed={seed}")
        df_ep = run_single_episode(
            seed=seed,
            policy=policy,
            width=args.width,
            height=args.height,
            num_agents=args.num_agents,
            max_steps=args.max_steps,
        )
        all_dfs.append(df_ep)

    big_df = pd.concat(all_dfs, ignore_index=True)

    csv_path = os.path.join(args.out_dir, "shared_policy_timeseries.csv")
    big_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved time-series CSV to {csv_path}")

    grouped = big_df.groupby("Step")

    metrics = [
        ("AgentsAlive", "AgentsAlive", "AgentsAlive_timeseries.png"),
        ("AvgEnergy", "AvgEnergy", "AvgEnergy_timeseries.png"),
        ("GiniEnergy", "GiniEnergy", "GiniEnergy_timeseries.png"),
        ("MinedNonrenewable", "MinedNonrenewable", "MinedNonrenewable_timeseries.png"),
        ("MinedRenewable", "MinedRenewable", "MinedRenewable_timeseries.png"),
        ("TotalScar", "TotalScar", "TotalScar_timeseries.png"),
    ]

    steps = grouped.size().index.to_numpy()
    n_episodes = args.n_episodes

    for col, ylabel, filename in metrics:
        if col not in big_df.columns:
            print(f"[WARN] Column '{col}' not found in DataFrame; skipping plot.")
            continue

        mean_series = grouped[col].mean()
        std_series = grouped[col].std().fillna(0.0)

        title = f"DQN: {ylabel} over time (mean ± std)"
        out_path = os.path.join(args.out_dir, filename)
        plot_metric(steps, mean_series, std_series, n_episodes, title, ylabel, out_path)

    print(f"[INFO] Saved plots into {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="models/dqn_sb3_final.zip")
    parser.add_argument("--vecnorm_path", type=str, default="models/vecnorm.pkl")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--height", type=int, default=30)
    parser.add_argument("--num_agents", type=int, default=15)
    parser.add_argument("--max_steps", type=int, default=200)

    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--start_seed", type=int, default=0)

    parser.add_argument("--out_dir", type=str, default="exp_shared_policy_DQN")


    args = parser.parse_args()
    main(args)
