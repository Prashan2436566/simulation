from stable_baselines3 import DQN
from mesa_sb3_env import MesaSB3Env
import numpy as np

def main():
    env = MesaSB3Env(
        num_agents=15,
        ideology="adaptive", 
        width=30,
        height=30,
        max_steps=1500
    )
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=256,
        gamma=0.99,
        verbose=1
    )
    
    model.learn(total_timesteps=1000000)
    model.save("multi_agent_dqn")