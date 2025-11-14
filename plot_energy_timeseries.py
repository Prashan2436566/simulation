import csv, os, argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mesa_sb3_env import MesaSB3Env

def make_env(width=30, height=30, num_agents=15, max_steps=1500):
    return MesaSB3Env(width, height, num_agents, True, "adaptive", max_steps)

def run_episode(model, venv):
    obs = venv.reset()
    steps, ren_s, non_s, cum_ren, cum_non = [], [], [], [], []
    c_ren = 0.0
    c_non = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)
        info = infos[0] if isinstance(infos, (list, tuple)) else infos

        s  = int(info.get("step", 0))
        rs = float(info.get("step_mined_renewable", 0.0))
        ns = float(info.get("step_mined_nonrenewable", 0.0))

        c_ren += rs
        c_non += ns

        steps.append(s)
        ren_s.append(rs)
        non_s.append(ns)
        cum_ren.append(c_ren)
        cum_non.append(c_non)

        if bool(dones[0]):
            break
    return np.array(steps), np.array(ren_s), np.array(non_s), np.array(cum_ren), np.array(cum_non)

def save_csv(path, steps, ren_step, non_step, cum_ren, cum_non):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step","ren_step","non_step","cum_ren","cum_non","cum_total"])
        for i in range(len(steps)):
            w.writerow([int(steps[i]), float(ren_step[i]), float(non_step[i]),
                        float(cum_ren[i]), float(cum_non[i]), float(cum_ren[i]+cum_non[i])])

def plot_concat(all_series, out_png):
    x, cum_ren, cum_non = [], [], []
    offset = 0
    for (steps, _, _, c_ren, c_non) in all_series:
        x.extend(list(steps + offset))
        cum_ren.extend(list(c_ren + (cum_ren[-1] if cum_ren else 0.0)))
        cum_non.extend(list(c_non + (cum_non[-1] if cum_non else 0.0)))
        offset = x[-1]
    x = np.array(x); cum_ren = np.array(cum_ren); cum_non = np.array(cum_non)

    plt.figure()
    plt.plot(x, cum_ren, label="Renewable (cum)")
    plt.plot(x, cum_non, label="Non-renewable (cum)")
    plt.plot(x, cum_ren + cum_non, label="Total (cum)")
    plt.xlabel("Global step (concatenated episodes)")
    plt.ylabel("Cumulative mined energy")
    plt.title("Cumulative energy over all time steps (concat episodes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"[OK] Saved: {out_png}")

def plot_align(all_series, max_steps, out_png):
    def pad_to(arr, T):
        if len(arr) >= T: return arr[:T]
        if len(arr) == 0: return np.zeros(T)
        pad = np.full(T - len(arr), arr[-1])
        return np.concatenate([arr, pad])

    cum_ren_mat, cum_non_mat = [], []
    for (steps, _, _, c_ren, c_non) in all_series:
        T = min(max_steps, len(c_ren))
        cum_ren_mat.append(pad_to(c_ren, max_steps))
        cum_non_mat.append(pad_to(c_non, max_steps))

    cum_ren_mat = np.vstack(cum_ren_mat) 
    cum_non_mat = np.vstack(cum_non_mat)

    mean_ren, std_ren = np.mean(cum_ren_mat, axis=0), np.std(cum_ren_mat, axis=0)
    mean_non, std_non = np.mean(cum_non_mat, axis=0), np.std(cum_non_mat, axis=0)
    mean_tot, std_tot = mean_ren + mean_non, np.sqrt(std_ren**2 + std_non**2)

    xs = np.arange(1, max_steps + 1)

    plt.figure()
    plt.plot(xs, mean_ren, label="Renewable (cum) mean")
    plt.fill_between(xs, mean_ren - std_ren, mean_ren + std_ren, alpha=0.2)
    plt.plot(xs, mean_non, label="Non-renewable (cum) mean")
    plt.fill_between(xs, mean_non - std_non, mean_non + std_non, alpha=0.2)
    plt.plot(xs, mean_tot, label="Total (cum) mean")
    plt.fill_between(xs, mean_tot - std_tot, mean_tot + std_tot, alpha=0.2)
    plt.xlabel("Step within episode")
    plt.ylabel("Cumulative mined energy")
    plt.title("Cumulative energy vs step (mean ± std across episodes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"[OK] Saved: {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    ap.add_argument("--max_steps", type=int, default=1500, help="Episode cap (align mode length)")
    ap.add_argument("--mode", choices=["concat","align"], default="align",
                    help="concat: one long timeline; align: mean±std over steps")
    ap.add_argument("--model", default="models/dqn_sb3_final.zip")
    ap.add_argument("--vecnorm", default="models/vecnorm.pkl")
    ap.add_argument("--out_prefix", default="energy_timeseries_all")
    args = ap.parse_args()

    # Build eval env and load model
    venv = DummyVecEnv([lambda: make_env(max_steps=args.max_steps)])
    venv = VecNormalize.load(args.vecnorm, venv)
    venv.training = False
    venv.norm_reward = False
    model = DQN.load(args.model, env=venv, device="auto")

    all_series = []
    for ep in range(args.episodes):
        series = run_episode(model, venv)
        all_series.append(series)
        csv_path = f"{args.out_prefix}_ep{ep+1}.csv"
        save_csv(csv_path, *series)
        print(f"[OK] Episode {ep+1} saved: {csv_path}")

    # Plot
    if args.mode == "concat":
        plot_concat(all_series, out_png=f"{args.out_prefix}_concat.png")
    else:
        plot_align(all_series, args.max_steps, out_png=f"{args.out_prefix}_align.png")

if __name__ == "__main__":
    main()
