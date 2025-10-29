#python evaluate_dqn.py
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mesa_sb3_env import MesaSB3Env

# Reload model and VecNormalize stats
venv = DummyVecEnv([lambda: MesaSB3Env(30, 30, 15, True, "adaptive", 1500)])
venv = VecNormalize.load("models/vecnorm.pkl", venv)
venv.training = False
venv.norm_reward = False

model = DQN.load("models/dqn_sb3_final.zip", env=venv, device="cuda")

mean_reward, std_reward = evaluate_policy(
    model, 
    venv, 
    n_eval_episodes=20, 
    deterministic=True, 
    return_episode_rewards=False
)

print(f"Mean reward over 20 episodes: {mean_reward:.3f} ± {std_reward:.3f}")
