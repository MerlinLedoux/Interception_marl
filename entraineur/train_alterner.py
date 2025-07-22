import os
import sys

sys.path.append(os.path.abspath(r"C:\Users\FX643778\Documents\Git\Interception_marl\environement"))

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

from policy_loader import load_chasseur_policy_bis, load_eviteur_policy_bis
from env import Affrontement
from env_eviteur import AffrontementSingleEviteur
from env_chasseur import AffrontementSingleChasseur

# === Paramétres ===
save_dir = "C:/Users/FX643778/Documents/Git/Interception_marl/models/alterner"
save_dir_cha = "C:/Users/FX643778/Documents/Git/Interception_marl/models/alterner/chasseur"
save_dir_evi = "C:/Users/FX643778/Documents/Git/Interception_marl/models/alterner/eviteur"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir_cha, exist_ok=True)
os.makedirs(save_dir_evi, exist_ok=True)

train_timesteps = 50_000  
loop = 3


for k in range(1, loop+1):

    # ============================================== Entrainement du chasseur ==============================================
    
    run_name_chasseur = f"chasseur{k}_al"
    wandb.init(
        project="affrontement-chasseur",
        name=run_name_chasseur,
        config={
            "policy_type": "MlpPolicy",
            "total_timesteps": train_timesteps,
            "agent": "chasseur"
        },
        sync_tensorboard=True,
    )    

    path_norm_eviteur = f"C:/Users/FX643778/Documents/Git/Interception_marl/models/alterner/eviteur/eviteur{k-1}_al_vecnormalize.pkl"
    path_model_eviteur = f"C:/Users/FX643778/Documents/Git/Interception_marl/models/alterner/eviteur/eviteur{k-1}_al.zip"

    eviteur_model, eviteur_env = load_eviteur_policy_bis(path_norm=path_norm_eviteur, path_model=path_model_eviteur)
    env_raw = Affrontement()
    env_wrapped = AffrontementSingleChasseur(
        env_raw,
        eviteur_model=eviteur_model,
        eviteur_env=eviteur_env
    )
    env_monitored = Monitor(env_wrapped)
    vec_env = DummyVecEnv([lambda: env_monitored])

    path_norm_chasseur = f"C:/Users/FX643778/Documents/Git/Interception_marl/models/alterner/chasseur/chasseur{k-1}_al_vecnormalize.pkl"
    path_model_chasseur = f"C:/Users/FX643778/Documents/Git/Interception_marl/models/alterner/chasseur/chasseur{k-1}_al.zip"

    vec_env = VecNormalize.load(path_norm_chasseur, vec_env)
    vec_env.training = True
    vec_env.norm_reward = True

    model = PPO.load(path_model_chasseur, env=vec_env)
    model.learn(
        total_timesteps=train_timesteps,
        callback=WandbCallback(
            gradient_save_freq=100, 
            verbose=2
        )
    )

    model.save(os.path.join(save_dir_cha, f"{run_name_chasseur}"))
    vec_env.save(os.path.join(save_dir_cha, f"{run_name_chasseur}_vecnormalize.pkl"))

    wandb.finish()

    # ============================================== Entrainement de l'eviteur ==============================================

    run_name_eviteur = f"eviteur{k}_al"
    wandb.init(
        project="affrontement-eviteur",
        name=run_name_eviteur,
        config={
            "policy_type": "MlpPolicy",
            "total_timesteps": train_timesteps,
            "agent": "eviteur"
        },
        sync_tensorboard=True,
    )

    path_norm_chasseur = f"C:/Users/FX643778/Documents/Git/Interception_marl/models/alterner/chasseur/chasseur{k}_al_vecnormalize.pkl"
    path_model_chasseur = f"C:/Users/FX643778/Documents/Git/Interception_marl/models/alterner/chasseur/chasseur{k}_al.zip"

    chasseur_model, chasseur_env = load_chasseur_policy_bis(path_norm=path_norm_chasseur ,path_model=path_model_chasseur)
    env_raw = Affrontement()
    env_wrapped = AffrontementSingleEviteur(
        env_raw,
        chasseur_model=chasseur_model,
        chasseur_env=chasseur_env
    )
    env_monitored = Monitor(env_wrapped)
    vec_env = DummyVecEnv([lambda: env_monitored])

    path_norm_eviteur = f"C:/Users/FX643778/Documents/Git/Interception_marl/models/alterner/eviteur/eviteur{k-1}_al_vecnormalize.pkl"
    path_model_eviteur = f"C:/Users/FX643778/Documents/Git/Interception_marl/models/alterner/eviteur/eviteur{k-1}_al.zip"

    vec_env = VecNormalize.load(path_norm_eviteur, vec_env)
    vec_env.training = True
    vec_env.norm_reward = True

    model = PPO.load(path_model_eviteur, env=vec_env)
    model.learn(
        total_timesteps=train_timesteps,
        callback=WandbCallback(
            gradient_save_freq=100, 
            verbose=2
        )
    )

    # === Sauvegarde du modèle final ===
    model.save(os.path.join(save_dir_evi, f"{run_name_eviteur}"))
    vec_env.save(os.path.join(save_dir_evi, f"{run_name_eviteur}_vecnormalize.pkl"))

    wandb.finish()