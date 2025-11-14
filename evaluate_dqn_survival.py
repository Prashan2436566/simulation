from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mesa_sb3_env import MesaSB3Env
import numpy as np

# Reload model and VecNormalize stats
venv = DummyVecEnv([lambda: MesaSB3Env(30, 30, 15, True, "adaptive", 1500)])
venv = VecNormalize.load("models/vecnorm.pkl", venv)
venv.training = False
venv.norm_reward = False

model = DQN.load("models/dqn_sb3_final.zip", env=venv, device="cuda")

# Return per-episode rewards AND lengths
ep_rewards, ep_lengths = evaluate_policy(
    model,
    venv,
    n_eval_episodes=20,
    deterministic=True,
    return_episode_rewards=True
)

print(f"Mean reward over {len(ep_rewards)} eps: {np.mean(ep_rewards):.3f} ± {np.std(ep_rewards):.3f}")
print(f"Mean survival time (steps): {np.mean(ep_lengths):.2f} ± {np.std(ep_lengths):.2f}")
