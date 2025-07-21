import gymnasium as gym
import numpy as np
import utils
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import Affrontement
from env_eviteur import AffrontementSingleEviteur

class AffrontementSingleChasseur(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.agent_id = "chasseur"
        self.observation_space = env.observation_space[self.agent_id]
        self.action_space = env.action_space[self.agent_id]

        # Chargement de la politique de l'éviteur
        env_raw = Affrontement()
        env_eviteur = AffrontementSingleEviteur(env_raw)
        vec_env = DummyVecEnv([lambda: env_eviteur])

        self.env_eviteur = VecNormalize.load("C:/Users/FX643778/Documents/Git/Interception_marl/models/reward_test/V22_vecnormalize.pkl", vec_env)
        self.env_eviteur.training = False    
        self.env_eviteur.norm_reward = False

        path_model = "C:/Users/FX643778/Documents/Git/Interception_marl/models/reward_test/V22_proche_2.zip"
        self.eviteur_model = PPO.load(path_model, env=self.env_eviteur)

        #--------------------------------------------------------------#
        # eviteur_env = AffrontementSingleEviteur(env)
        # dummy_vec_env = DummyVecEnv([lambda: eviteur_env])

        # # Création du VecNormalize propre
        # self.eviteur_norm = VecNormalize(dummy_vec_env, norm_obs=True, norm_reward=False)
        # self.eviteur_norm.training = False
        # self.eviteur_norm.norm_reward = False

        # # Chargement des stats à partir du fichier sauvegardé
        # stats_path = "C:/Users/FX643778/Documents/Git/Interception_marl/models/reward_test/V22_vecnormalize.pkl"
        # loaded_norm = VecNormalize.load(stats_path, dummy_vec_env)

        # # Copier les statistiques dans notre VecNormalize actif
        # self.eviteur_norm.obs_rms = loaded_norm.obs_rms
        # self.eviteur_norm.ret_rms = loaded_norm.ret_rms

        # # Chargement du modèle entraîné de l'éviteur
        # model_path = "C:/Users/FX643778/Documents/Git/Interception_marl/models/reward_test/V22_proche_2.zip"
        # self.eviteur_model = PPO.load(model_path, env=self.eviteur_norm)

    def reset(self, **kwargs):
        obs_dict, _ = self.env.reset(**kwargs)
        return obs_dict[self.agent_id], {}

    def step(self, action_chasseur):
        obs_dict = self.env._get_obs()

        obs_eviteur = obs_dict["eviteur"]
        obs_eviteur_norm = self.env_eviteur.normalize_obs(obs_eviteur)
        action_eviteur, _ = self.eviteur_model.predict(obs_eviteur_norm, deterministic=True)

        action_dict = {
            "eviteur": action_eviteur,
            "chasseur": action_chasseur
        }

        # print(action_dict)

        obs_dict, reward_dict, terminated, truncated, info_dict = self.env.step(action_dict)

        return (
            obs_dict[self.agent_id],
            reward_dict[self.agent_id],
            terminated[self.agent_id],
            truncated[self.agent_id],
            info_dict.get(self.agent_id, {})
        )
