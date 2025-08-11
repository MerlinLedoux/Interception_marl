from env_aff import DoubleChasseur
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecMonitor
import wandb
from wandb.integration.sb3 import WandbCallback

# === Prametres ===
total_timesteps = 20_000_000
run_name = "norm_rew2"

env = DoubleChasseur()
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
env = VecMonitor(env)

wandb.init(
    project="affrontement",
    name=run_name,
    config={
        "policy_type": "MlpPolicy",
        "total_timesteps": total_timesteps,
        "agent": "chasseur"
    },
    sync_tensorboard=True,
)

model = PPO(
    MlpPolicy,
    env,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/"
)
model.learn(
    total_timesteps=total_timesteps,
    callback=WandbCallback(
        gradient_save_freq=100,
        verbose=1
    )
)
model.save(run_name)
