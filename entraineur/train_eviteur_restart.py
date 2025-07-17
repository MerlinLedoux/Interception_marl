import os
import sys

sys.path.append(os.path.abspath(r"C:\Users\FX643778\Documents\Git\Interception_marl\environement"))

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

from env import Affrontement
from env_eviteur import AffrontementSingleAgent

# Chemins
save_dir = "C:/Users/FX643778/Documents/Git/Interception_marl/models/restart"
vecnorm_path = "C:/Users/FX643778/Documents/Git/Interception_marl/models/V7_vecnormalize.pkl"
model_path = "C:/Users/FX643778/Documents/Git/Interception_marl/models/V7_chasseur_moyen3.zip"

# wandb init
wandb.init(
    project="affrontement-ppo",
    name="chasseur_restart",
    config={
        "policy_type": "MlpPolicy",
        "total_timesteps": 250_000,
        "agent": "eviteur",
        "learning_rate" : "5e-3"
    },
    sync_tensorboard=True,
)

# Environnement
env_raw = Affrontement()
env_wrapped = AffrontementSingleAgent(env_raw, agent_id="eviteur")
env_monitored = Monitor(env_wrapped)

# Chargement de la normalisation avec l'env de base
vec_env = DummyVecEnv([lambda: env_monitored])
vec_env = VecNormalize.load(vecnorm_path, vec_env)
vec_env.training = True
vec_env.norm_reward = True

# Chargement du modèle existant
model = PPO.load(model_path, env=vec_env)
model.lr_schedule = lambda _: 5e-3

# Entraînement
model.learn(
    total_timesteps=250_000,
    callback=WandbCallback(
        gradient_save_freq=100,
        verbose=2
    )
)

# Sauvegarde
model.save(os.path.join(save_dir, "V7_1_mix"))
vec_env.save(os.path.join(save_dir, "V7_1_vecnormalize.pkl"))

wandb.finish()
