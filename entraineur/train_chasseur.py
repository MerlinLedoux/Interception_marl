import os
import sys

sys.path.append(os.path.abspath(r"C:\Users\FX643778\Documents\Git\Interception_marl\environement"))

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

# J'ai en permanece deux warning ici mais le code fonctione trés bien
from env import Affrontement
from env_eviteur import AffrontementSingleEviteur
from env_chasseur import AffrontementSingleChasseur

save_dir = "C:/Users/FX643778/Documents/Git/Interception_marl/models/chasseur"
wandb.init(
    project="affrontement-ppo-chasseur",
    name="chasseur0",
    config={
        "policy_type": "MlpPolicy",
        "total_timesteps": 5_000,
        "agent": "chasseur"
    },
    sync_tensorboard=True,
)

env_raw = Affrontement()
env_wrapped = AffrontementSingleChasseur(env_raw)
env_monitored = Monitor(env_wrapped)

# Vectorisation + normalisation
vec_env = DummyVecEnv([lambda: env_monitored])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

# PPO + wandb callback
model = PPO("MlpPolicy", 
            vec_env, 
            verbose=1, 
            tensorboard_log="./ppo_tensorboard", 
            ent_coef=0.01)

model.learn(
    total_timesteps=5_000,
    callback=WandbCallback(
        gradient_save_freq=100,
        verbose=2
    )
)

save_path = os.path.join(save_dir, "V0_chasseur")
model.save(save_path)
vec_env.save(os.path.join(save_dir, "V0_c_vecnormalize.pkl"))

wandb.finish()