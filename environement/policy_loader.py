# policy_loader.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import Affrontement

def load_chasseur_policy():
    from env_chasseur import AffrontementSingleChasseur
    env_raw = Affrontement()
    env_chasseur = AffrontementSingleChasseur(env_raw)
    vec_env = DummyVecEnv([lambda: env_chasseur])
    env_norm = VecNormalize.load("models/chasseur/chasseur_vecnormalize.pkl", vec_env)
    env_norm.training = False
    env_norm.norm_reward = False
    model = PPO.load("models/chasseur/chasseur.zip", env=env_norm)
    return model, env_norm

def load_eviteur_policy():
    from env_eviteur import AffrontementSingleEviteur
    env_raw = Affrontement()
    env_eviteur = AffrontementSingleEviteur(env_raw)
    vec_env = DummyVecEnv([lambda: env_eviteur])
    # env_norm = VecNormalize.load("models/base/V22_vecnormalize.pkl", vec_env)
    env_norm = VecNormalize.load("models/eviteur/eviteur3_load_vecnormalize.pkl", vec_env)
    env_norm.training = False
    env_norm.norm_reward = False
    # model = PPO.load("models/base/V22_proche_2.zip", env=env_norm)
    model = PPO.load("models/eviteur/eviteur3_load.zip", env=env_norm)
    return model, env_norm

# Version bis des deux fonctions avec l'emplacement des fichier en parametres

def load_chasseur_policy_bis(path_norm, path_model):
    from env_chasseur import AffrontementSingleChasseur
    env_raw = Affrontement()
    env_chasseur = AffrontementSingleChasseur(env_raw)
    vec_env = DummyVecEnv([lambda: env_chasseur])
    env_norm = VecNormalize.load(path_norm, vec_env)
    env_norm.training = False
    env_norm.norm_reward = False
    model = PPO.load(path_model, env=env_norm)
    return model, env_norm

def load_eviteur_policy_bis(path_norm, path_model):
    from env_eviteur import AffrontementSingleEviteur
    env_raw = Affrontement()
    env_eviteur = AffrontementSingleEviteur(env_raw)
    vec_env = DummyVecEnv([lambda: env_eviteur])
    env_norm = VecNormalize.load(path_norm, vec_env)
    env_norm.training = False
    env_norm.norm_reward = False
    model = PPO.load(path_model, env=env_norm)
    return model, env_norm