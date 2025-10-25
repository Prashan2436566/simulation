# train_sb3_dqn.py
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from mesa_sb3_env import MesaSB3Env
import os

def make_env():
    # Long-ish episodes so the agent can experience consequences
    # You can pass knobs that match your IdeologyModel constructor
    return MesaSB3Env(
        width=30, height=30,
        num_agents=15,
        renewables_regenerate=True,
        ideology="adaptive",   # control exactly one adaptive agent
        max_steps=1500,        # per-episode horizon (increase if needed)
        # Any additional model kwargs go here, e.g. pool_floor=10.0, degrade_chance=0.5, ...
    )

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = make_env()
    env = Monitor(env, filename="logs/monitor.csv")  # records episode rewards/lengths

    # Checkpoints every N steps
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,           # save a snapshot every 50k steps
        save_path="models",
        name_prefix="dqn_sb3",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # Optional: periodic evaluation on a fresh copy of the env
    eval_env = make_env()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/best",
        log_path="logs/eval",
        eval_freq=50_000,
        n_eval_episodes=5,
        deterministic=False,
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=200_000,
        learning_starts=10_000,      # warmup before learning
        batch_size=256,
        gamma=0.95,
        target_update_interval=10_000,
        train_freq=4,                # env steps per gradient step trigger
        gradient_steps=1,            # gradient steps per trigger (increase if you want more updates)
        exploration_fraction=0.2,    # linearly anneal epsilon over first 20% of steps
        exploration_initial_eps=0.9,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="logs/tb",
    )

    # Give it real time to learn; start with 2–5 million and scale based on speed
    model.learn(
        total_timesteps=200000,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True
    )

    model.save("models/dqn_sb3_final")
    env.close()
    eval_env.close()
