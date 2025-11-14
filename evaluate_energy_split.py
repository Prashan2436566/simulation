# evaluate_energy_split.py
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mesa_sb3_env import MesaSB3Env

def make_env():
    return MesaSB3Env(30, 30, 15, True, "adaptive", 1000000)

def run_eval(model_path="models/dqn_sb3_final.zip",
             vecnorm_path="models/vecnorm.pkl",
             episodes=20):
    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False

    model = DQN.load(model_path, env=venv, device="auto")

    ep_rewards, ep_lengths = [], []
    ep_ren, ep_non, ep_tot = [], [], []

    for _ in range(episodes):
        obs = venv.reset()
        cum_rew = 0.0
        steps = 0
        last_info = {}

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)
            cum_rew += float(rewards[0])
            steps += 1

            if bool(dones[0]):  
                last_info = infos[0] if isinstance(infos, (list, tuple)) else infos
                break

        ep_rewards.append(cum_rew)
        ep_lengths.append(steps)
        ep_ren.append(float(last_info.get("ep_mined_renewable", 0.0)))
        ep_non.append(float(last_info.get("ep_mined_nonrenewable", 0.0)))
        ep_tot.append(float(last_info.get("ep_mined_total", 0.0)))

    def mu_sigma(x): return np.mean(x), np.std(x)

    mr, sr = mu_sigma(ep_rewards)
    ml, sl = mu_sigma(ep_lengths)
    mren, sren = mu_sigma(ep_ren)
    mnon, snon = mu_sigma(ep_non)
    mtot, stot = mu_sigma(ep_tot)

    print(f"Mean reward over {episodes} episodes: {mr:.3f} ± {sr:.3f}")
    print(f"Mean survival time (steps): {ml:.2f} ± {sl:.2f}")
    print(f"Energy mined (renewable):   {mren:.2f} ± {sren:.2f}")
    print(f"Energy mined (nonrenewable):{mnon:.2f} ± {snon:.2f}")
    print(f"Energy mined (total):       {mtot:.2f} ± {stot:.2f}")
    if mtot > 0:
        print(f"Renewable share: {100.0 * (mren / mtot):.1f}%")

if __name__ == "__main__":
    run_eval()
