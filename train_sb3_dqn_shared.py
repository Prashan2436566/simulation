import os
import argparse

import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from mesa_sb3_env import MesaSB3Env  


def make_env(seed: int = 0):
    def _init():
        env = MesaSB3Env(
            width=15,
            height=15,
            num_agents=15,      
            max_steps=400,
            seed=seed,
        )
        env = Monitor(env)      
        return env
    return _init


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    env_fn = make_env(seed=args.seed)
    venv = DummyVecEnv([env_fn])

    venv = VecNormalize(
        venv,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    model = DQN(
        policy="MlpPolicy",
        env=venv,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=0.005,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        verbose=1,
        tensorboard_log=os.path.join(args.save_dir, "tb"),
        device=args.device,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=os.path.join(args.save_dir, "checkpoints"),
        name_prefix="dqn_sb3_shared",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_env_fn = make_env(seed=args.seed + 1)
    eval_venv = DummyVecEnv([eval_env_fn])
    eval_venv = VecNormalize(
        eval_venv,
        training=False,
        norm_obs=True,
        norm_reward=False,
    )

    eval_callback = EvalCallback(
        eval_env=eval_venv,
        best_model_save_path=os.path.join(args.save_dir, "best_model"),
        log_path=os.path.join(args.save_dir, "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
#
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
    )
    model_path = os.path.join(args.save_dir, "dqn_sb3_shared_final")
    vecnorm_path = os.path.join(args.save_dir, "vecnorm_shared.pkl")

    model.save(model_path)
    venv.save(vecnorm_path)

    print(f"[INFO] Saved final shared model to: {model_path}.zip")
    print(f"[INFO] Saved shared VecNormalize stats to: {vecnorm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=str, default="modelsShared")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--buffer_size", type=int, default=80_000)
    parser.add_argument("--learning_starts", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train_freq", type=int, default=4)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--target_update_interval", type=int, default=10_000)
    parser.add_argument("--exploration_fraction", type=float, default=0.3)
    parser.add_argument("--exploration_final_eps", type=float, default=0.05)

    parser.add_argument("--checkpoint_freq", type=int, default=50_000)
    parser.add_argument("--eval_freq", type=int, default=20_000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)

    args = parser.parse_args()
    main(args)
