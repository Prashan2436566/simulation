# batch_runner.py
import argparse, random
import numpy as np
import pandas as pd

# Import your model exactly like the UI does
from model.model import IdeologyModel

def run_once(steps: int, ideology: str, seed: int | None, overrides: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    m = IdeologyModel(
        width=overrides.get("width", 30),
        height=overrides.get("height", 30),
        num_agents=overrides.get("num_agents", 15),
        renewables_regenerate=overrides.get("renewables_regenerate", True),
        ideology=ideology,
        cost_renewable_setup=overrides.get("cost_renewable_setup", 5.0),
        cost_extract_renewable=overrides.get("cost_extract_renewable", 1.0),
        cost_extract_nonrenewable=overrides.get("cost_extract_nonrenewable", 2.0),
        yield_per_mine_renewable=overrides.get("yield_per_mine_renewable", 4),
        yield_per_mine_nonrenewable=overrides.get("yield_per_mine_nonrenewable", 6),
        renewable_cooldown_steps=overrides.get("renewable_cooldown_steps", 5),
        renewable_overuse_trigger=overrides.get("renewable_overuse_trigger", 6),
        renewable_fatigue_decay=overrides.get("renewable_fatigue_decay", 1),
        pool_floor=overrides.get("pool_floor", 10.0),
        degrade_period=overrides.get("degrade_period", 10),
        degrade_chance=overrides.get("degrade_chance", 0.5),
        repair_energy_cost=overrides.get("repair_energy_cost", 10.0),
    )

    for _ in range(steps):
        m.step()

    model_df = m.datacollector.get_model_vars_dataframe().reset_index(names="Step")

    agents = [a for a in m.schedule.agents if hasattr(a, "energy")]
    agent_df = pd.DataFrame([{
        "ideology": getattr(a, "ideology", ""),
        "energy": getattr(a, "energy", 0.0),
        "total_collected_energy": getattr(a, "total_collected_energy", 0.0),
    } for a in agents])

    return model_df, agent_df

def summarize_runs(all_model_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in all_model_df.columns if c not in ("Step", "run")]
    grouped = all_model_df.groupby("Step")[metrics]
    mean = grouped.mean().add_suffix("_mean")
    std  = grouped.std(ddof=1).add_suffix("_std")
    return pd.concat([mean, std], axis=1).reset_index()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ideology", type=str, default="adaptive_direct")
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_prefix", type=str, default="results")
    p.add_argument("--num_agents", type=int, default=15)
    p.add_argument("--renewables_regenerate", type=int, default=1)  # 1/0
    args = p.parse_args()

    overrides = {
        "num_agents": args.num_agents,
        "renewables_regenerate": bool(args.renewables_regenerate),
    }

    all_models = []
    final_agent_summaries = []

    for r in range(args.runs):
        seed = None if args.seed is None else (args.seed + r)
        model_df, agent_df = run_once(args.steps, args.ideology, seed, overrides)
        model_df["run"] = r
        all_models.append(model_df)

        if not agent_df.empty:
            agent_df["run"] = r
            final_agent_summaries.append(agent_df.assign(Step=args.steps))

    all_model_df = pd.concat(all_models, ignore_index=True)
    all_agent_df = pd.concat(final_agent_summaries, ignore_index=True) if final_agent_summaries else pd.DataFrame()

    step_summary = summarize_runs(all_model_df)

    final_by_run = all_model_df.sort_values(["run", "Step"]).groupby("run").tail(1).reset_index(drop=True)
    metrics = [c for c in final_by_run.columns if c not in ("Step", "run")]
    final_mean = final_by_run[metrics].mean(numeric_only=True).to_frame("mean")
    final_std  = final_by_run[metrics].std(numeric_only=True, ddof=1).to_frame("std")
    final_summary = pd.concat([final_mean, final_std], axis=1)

    prefix = f"{args.out_prefix}_{args.ideology}_runs{args.runs}_steps{args.steps}"
    all_model_df.to_csv(f"{prefix}__all_model_timeseries.csv", index=False)
    step_summary.to_csv(f"{prefix}__per_step_mean_std.csv", index=False)
    final_by_run.to_csv(f"{prefix}__final_by_run.csv", index=False)
    final_summary.to_csv(f"{prefix}__final_mean_std.csv")

    print(f"\n=== Finished: {args.ideology} | runs={args.runs} | steps={args.steps} ===")
    print("\nFinal-step means and stds (across runs):")
    print(final_summary.round(3))
    print(f"\nSaved files with prefix: {prefix}")

if __name__ == "__main__":
    main()
