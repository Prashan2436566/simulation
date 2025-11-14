from stable_baselines3 import DQN
from mesa_sb3_env import MesaSB3Env

# Load environment identical to training
env = MesaSB3Env(
    width=30,
    height=30,
    num_agents=15,
    renewables_regenerate=True,
    ideology="adaptive",
    max_steps=1500,
)

# Load the trained SB3 model
model = DQN.load("models/dqn_sb3_final.zip", env=env)

# Evaluate for a few episodes
for ep in range(5):
    obs, info = env.reset()
    done = truncated = False
    total_reward = 0
    steps = 0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

    print(f"Episode {ep+1}: return={total_reward:.2f}, steps={steps}")

env.close()
