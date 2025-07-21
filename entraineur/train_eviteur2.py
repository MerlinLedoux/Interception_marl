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
from env_eviteur import AffrontementSingleEviteur
from env_chasseur import AffrontementSingleChasseur

# === Paramètres ===
save_dir = "C:/Users/FX643778/Documents/Git/Interception_marl/models/eviteur"
os.makedirs(save_dir, exist_ok=True)

total_timesteps = 100_000  
run_name = "eviteur2"

# === Initialisation wandb ===
save_dir = "C:/Users/FX643778/Documents/Git/Interception_marl/models/eviteur"
wandb.init(
    project="affrontement-ppo-eviteur",
    name="eviteur-2",
    config={
        "policy_type": "MlpPolicy",
        "total_timesteps": total_timesteps,
        "agent": "eviteur"
    },
    sync_tensorboard=True,
)

env_raw = Affrontement()
env_wrapped = AffrontementSingleEviteur(env_raw)
env_monitored = Monitor(env_wrapped)

# Vectorisation + normalisation
vec_env = DummyVecEnv([lambda: env_monitored])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

# === Création de l'environnement ===
base_env = Affrontement()
wrapped_env = AffrontementSingleEviteur(base_env)
monitored_env = Monitor(wrapped_env)

vec_env = DummyVecEnv([lambda: monitored_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
vec_env.training = True 
vec_env.norm_reward = True

# === Environnement d’évaluation ===
eval_env = DummyVecEnv([lambda: Monitor(AffrontementSingleEviteur(Affrontement()))])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
eval_env.training = False  # Important pour l'éval
eval_env.norm_reward = False

# === Callback d’évaluation (optionnel mais utile) ===
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(save_dir, "best"),
    log_path=os.path.join(save_dir, "eval_logs"),
    eval_freq=10_000,
    deterministic=True,
    render=False,
)

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
    callback=[
        WandbCallback(gradient_save_freq=100, verbose=2),
        eval_callback
    ]
)

# === Sauvegarde du modèle final ===
model.save(os.path.join(save_dir, f"{run_name}"))
vec_env.save(os.path.join(save_dir, f"{run_name}_vecnormalize.pkl"))

wandb.finish()