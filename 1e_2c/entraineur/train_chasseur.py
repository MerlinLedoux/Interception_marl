import os
import sys

sys.path.append(os.path.abspath(r"C:\Users\FX643778\Documents\Git\Interception_marl\1e_1c\environement"))

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

from policy_loader import load_eviteur_policy
from env import Affrontement
from env_chasseur import AffrontementSingleChasseur

# === Paramètres ===
save_dir = "C:/Users/FX643778/Documents/Git/Interception_marl/1e_1c/models/chasseur"
os.makedirs(save_dir, exist_ok=True)

total_timesteps = 250_000  
run_name = "chasseur3_load"

# === Initialisation wandb ===
wandb.init(
    project="affrontement-chasseur",
    name=run_name,
    config={
        "policy_type": "MlpPolicy",
        "total_timesteps": total_timesteps,
        "agent": "chasseur"
    },
    sync_tensorboard=True,
)

# === Chargement du modèle eviteur (fixe) ===
eviteur_model, eviteur_env = load_eviteur_policy()

env_raw = Affrontement()
env_wrapped = AffrontementSingleChasseur(
    env_raw,
    eviteur_model=eviteur_model,
    eviteur_env=eviteur_env
)
env_monitored = Monitor(env_wrapped)

# Vectorisation + normalisation
vec_env = DummyVecEnv([lambda: env_monitored])

# vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
vec_env = VecNormalize.load("models/chasseur/chasseur2_vecnormalize.pkl", vec_env)
vec_env.training = True
vec_env.norm_reward = True


model = PPO.load("models/chasseur/chasseur2.zip", env=vec_env)
model.learn(
    total_timesteps=total_timesteps,
    callback=WandbCallback(
        gradient_save_freq=100, 
        verbose=2
    )
)

# === Sauvegarde du modèle final ===
model.save(os.path.join(save_dir, f"{run_name}"))
vec_env.save(os.path.join(save_dir, f"{run_name}_vecnormalize.pkl"))

wandb.finish()