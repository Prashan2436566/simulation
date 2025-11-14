import os
import argparse
from typing import Callable, List

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

from mesa_sb3_env_centralised import MesaSB3EnvCentralised


def make_env_fn(width: int, height: int, num_agents: int, max_steps: int, seed: int) -> Callable[[], MesaSB3EnvCentralised]:
    def _thunk():
        env = MesaSB3EnvCentralised(
            width=width,
            height=height,
            num_agents=num_agents,
            renewables_regenerate=True,
            ideology="adaptive",
            max_steps=max_steps,
            seed=seed,
        )
        return env
    return _thunk


def build_vec_env(
    n_envs: int,
    width: int,
    height: int,
    num_agents: int,
    max_steps: int,
    seed: int,
    monitor_path: str | None = None,
) -> DummyVecEnv:
    set_random_seed(seed)
    env_fns: List[Callable[[], MesaSB3EnvCentralised]] = [
        make_env_fn(width, height, num_agents, max_steps, seed + i) for i in range(n_envs)
    ]
    venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv, filename=monitor_path) if monitor_path else VecMonitor(venv)
    return venv


def main():
    parser = argparse.ArgumentParser(
        description="Train centralised SB3 DQN on Mesa adaptive environment with VecNormalize, CSV & TensorBoard logging."
    )
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--height", type=int, default=30)
    parser.add_argument("--num_agents", type=int, default=15)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--buffer_size", type=int, default=300_000)
    parser.add_argument("--learning_starts", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_freq", type=int, default=4)
    parser.add_argument("--gradient_steps", type=int, default=4)
    parser.add_argument("--target_update_interval", type=int, default=5_000)
    parser.add_argument("--exploration_fraction", type=float, default=0.4)
    parser.add_argument("--exploration_final_eps", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tb_logdir", type=str, default="logs/tb_centralised")
    parser.add_argument("--csv_logdir", type=str, default="logs/dqn_csv_centralised")
    parser.add_argument("--monitor_logdir", type=str, default="logs/monitor_centralised")
    parser.add_argument("--model_dir", type=str, default="models_centralised")
    parser.add_argument("--ckpt_freq", type=int, default=100_000)
    parser.add_argument("--eval_freq", type=int, default=50_000)
    parser.add_argument("--n_eval_eps", type=int, default=5)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.tb_logdir, exist_ok=True)
    os.makedirs(args.csv_logdir, exist_ok=True)
    os.makedirs(args.monitor_logdir, exist_ok=True)
    os.makedirs("logs/eval_centralised", exist_ok=True)

    print(f"[INFO] Using device: {args.device}")

    train_env = build_vec_env(
        n_envs=args.n_envs,
        width=args.width,
        height=args.height,
        num_agents=args.num_agents,
        max_steps=args.max_steps,
        seed=args.seed,
        monitor_path=os.path.join(args.monitor_logdir, "monitor"),
    )

    vecnorm_path = os.path.join(args.model_dir, "vecnorm_centralised.pkl")
    if args.resume and os.path.exists(vecnorm_path):
        train_env = VecNormalize.load(vecnorm_path, train_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = build_vec_env(
        n_envs=1,
        width=args.width,
        height=args.height,
        num_agents=args.num_agents,
        max_steps=args.max_steps,
        seed=args.seed + 10_000,
        monitor_path=None,
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    if args.resume and os.path.exists(vecnorm_path):
        eval_env = VecNormalize.load(vecnorm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    callbacks = []
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=5,
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.model_dir, "best"),
        log_path="logs/eval_centralised",
        eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
        n_eval_episodes=args.n_eval_eps,
        deterministic=False,
        callback_after_eval=stop_cb,
        warn=True,
    )
    callbacks.append(eval_cb)

    if args.ckpt_freq and args.ckpt_freq > 0:
        ckpt_cb = CheckpointCallback(
            save_freq=max(args.ckpt_freq // max(args.n_envs, 1), 1),
            save_path=args.model_dir,
            name_prefix="dqn_ckpt_centralised",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        callbacks.append(ckpt_cb)

    policy_kwargs = dict(net_arch=list(args.hidden))
    model_path = os.path.join(args.model_dir, "dqn_sb3_centralised_final.zip")

    try:
        import torch
        print("[INFO] torch.cuda.is_available() =", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("[INFO] Current GPU:", torch.cuda.get_device_name(0))
    except Exception:
        pass

    if args.resume and os.path.exists(model_path):
        print(f"[INFO] Resuming model from {model_path}")
        model = DQN.load(model_path, env=train_env, device=args.device)
        model.gamma = args.gamma
        model.learning_rate = args.learning_rate
        model.batch_size = args.batch_size
        model.train_freq = args.train_freq
        model.gradient_steps = args.gradient_steps
        model.target_update_interval = args.target_update_interval
        model.exploration_initial_eps = 1.0
        model.exploration_final_eps = args.exploration_final_eps
        model.exploration_fraction = args.exploration_fraction
    else:
        model = DQN(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=args.gamma,
            target_update_interval=args.target_update_interval,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=1.0,
            exploration_final_eps=args.exploration_final_eps,
            verbose=1,
            tensorboard_log=args.tb_logdir,
            policy_kwargs=policy_kwargs,
            device=args.device,
        )

    new_logger = configure(args.csv_logdir, ["csv", "tensorboard"])
    model.set_logger(new_logger)

    print("[INFO] Starting training (centralised)...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        log_interval=10,
        tb_log_name="DQN_adaptive_centralised",
    )

    print("[INFO] Saving model and VecNormalize stats (centralised)...")
    model.save(model_path)
    train_env.save(vecnorm_path)
    print(f"[OK] Saved model to {model_path}")
    print(f"[OK] Saved VecNormalize to {vecnorm_path}")

    print("\nWhere to look (centralised):")
    print(f"  • TensorBoard: {args.tb_logdir}/DQN_adaptive_centralised_*")
    print(f"  • CSV progress: {args.csv_logdir}/progress.csv")
    print(f"  • Monitor: {args.monitor_logdir}/monitor.csv")
    print("  • Best model: models_centralised/best/best_model.zip")
    print("  • Eval curves: logs/eval_centralised/evaluations.npz")
    if args.ckpt_freq and args.ckpt_freq > 0:
        print("  • Checkpoints: models_centralised/dqn_ckpt_centralised_*")


if __name__ == "__main__":
    main()
