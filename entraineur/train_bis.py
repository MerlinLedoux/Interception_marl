import os
import sys

sys.path.append(os.path.abspath(r"C:\Users\FX643778\Documents\Git\Interception_marl\environement"))

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from policy_loader import load_eviteur_policy
from env import Affrontement
from env_chasseur import AffrontementSingleChasseur

# === Configuration de l'entraînement ===
total_timesteps = 250_000
run_name = "chasseur_entrainement"
save_dir = "C:/Users/FX643778/Documents/Git/Interception_marl/models/chasseur"
os.makedirs(save_dir, exist_ok=True)

# === Initialisation wandb ===
wandb.init(
    project="affrontement-ppo-chasseur",
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

# === Préparation de l'environnement pour le chasseur ===
env_raw = Affrontement()
env_wrapped = AffrontementSingleChasseur(
    env_raw,
    eviteur_model=eviteur_model,
    eviteur_env=eviteur_env
)

vec_env = DummyVecEnv([lambda: env_wrapped])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

# === Entraînement PPO ===
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log="./ppo_tensorboard",
    ent_coef=0.01
)

model.learn(
    total_timesteps=total_timesteps,
    callback=WandbCallback(
        gradient_save_freq=100,
        verbose=2
    )
)

# === Sauvegarde du modèle entraîné ===
model.save(os.path.join(save_dir, f"{run_name}"))
vec_env.save(os.path.join(save_dir, f"{run_name}_vecnormalize.pkl"))

wandb.finish()
