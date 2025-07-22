import os
import sys

sys.path.append(os.path.abspath(r"C:\Users\FX643778\Documents\Git\Interception_marl\environement"))

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from wandb.integration.sb3 import WandbCallback

# J'ai en permanece deux warning ici mais le code fonctione trés bien
from env import Affrontement
from env_chasseur import AffrontementSingleChasseur

# === Paramètres ===
save_dir = "C:/Users/FX643778/Documents/Git/Interception_marl/models/chasseur"
os.makedirs(save_dir, exist_ok=True)

total_timesteps = 250_000  
run_name = "chasseur1"

# === Initialisation wandb ===
save_dir = "C:/Users/FX643778/Documents/Git/Interception_marl/models/chasseur"
wandb.init(
    project="affrontement-ppo-chasseur",
    name="chasseur_1",
    config={
        "policy_type": "MlpPolicy",
        "total_timesteps": total_timesteps,
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


# === Création et entraînement du modèle PPO ===
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

# === Sauvegarde du modèle final ===
model.save(os.path.join(save_dir, f"{run_name}"))
vec_env.save(os.path.join(save_dir, f"{run_name}_vecnormalize.pkl"))

wandb.finish()