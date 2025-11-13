# plot_learning_curve.py
import pandas as pd
import matplotlib.pyplot as plt

csv = "logs/monitor.csv"  
df = pd.read_csv(csv)

x = df["time/total_timesteps"]
rew = df["rollout/ep_rew_mean"]
len_ = df["rollout/ep_len_mean"]

plt.figure()
plt.plot(x, rew, label="Mean episode reward")
plt.xlabel("Total timesteps"); plt.ylabel("Reward"); plt.title("Learning curve")
plt.legend(); plt.tight_layout(); plt.savefig("learning_reward.png", dpi=160)

plt.figure()
plt.plot(x, len_, label="Mean episode length")
plt.xlabel("Total timesteps"); plt.ylabel("Steps"); plt.title("Survival over training")
plt.legend(); plt.tight_layout(); plt.savefig("learning_length.png", dpi=160)

print("Saved learning_reward.png and learning_length.png")
