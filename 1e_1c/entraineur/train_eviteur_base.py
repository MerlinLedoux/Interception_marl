import os
import sys

sys.path.append(os.path.abspath(r"C:\Users\FX643778\Documents\Git\Interception_marl\1e_1c\environement"))

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_util import make_vec_env


# J'ai en permanece deux warning ici mais le code fonctione trés bien
from env import Affrontement
from env_eviteur_base import AffrontementSingleEviteurBase

save_dir = "C:/Users/FX643778/Documents/Git/Interception_marl/1e_1c/models/base"
run_name = "V23_base"
total_timesteps = 100_000

# Initialisation de wandb
wandb.init(
    project="affrontement-ppo",
    name=run_name,
    config={
        "policy_type": "MlpPolicy",
        "total_timesteps": total_timesteps,
        "agent": "eviteur"
    },
    sync_tensorboard=True,
)

# Environnement (avec Monitor pour wandb) 
env_raw = Affrontement()    # Environement multi agent
env_wrapped = AffrontementSingleEviteurBase(env_raw)  # Extraction d'un agent
env_monitored = Monitor(env_wrapped)    # Enveloppe l'environement dans un wrapper de suivi (pour wandb)

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
    total_timesteps=total_timesteps,
    callback=WandbCallback(
        gradient_save_freq=100,
        verbose=2
    )
)

model.save(os.path.join(save_dir, f"{run_name}"))
vec_env.save(os.path.join(save_dir, f"{run_name}_vecnormalize.pkl"))

wandb.finish()