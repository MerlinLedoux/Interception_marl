import gymnasium as gym
import numpy as np
import utils
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import Affrontement
from env_chasseur import AffrontementSingleChasseur

class AffrontementSingleEviteur(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.agent_id = "eviteur"
        self.observation_space = env.observation_space[self.agent_id]
        self.action_space = env.action_space[self.agent_id]

        # Chargement de la politique de l'éviteur
        env_raw = Affrontement()
        env_chasseur = AffrontementSingleChasseur(env_raw)
        vec_env = DummyVecEnv([lambda: env_chasseur])

        self.env_chasseur = VecNormalize.load("C:/Users/FX643778/Documents/Git/Interception_marl/models/chasseur/chasseur0_vecnormalize.pkl", vec_env)
        self.env_chasseur.training = False    
        self.env_chasseur.norm_reward = False

        path_model = "C:/Users/FX643778/Documents/Git/Interception_marl/models/chasseur/chasseur0.zip"
        self.chasseur_model = PPO.load(path_model, env=self.env_chasseur)


    def reset(self, **kwargs):
        obs_dict, _ = self.env.reset(**kwargs)
        return obs_dict[self.agent_id], {}

    def step(self, action_eviteur):
        obs_dict = self.env._get_obs()

        obs_chasseur = obs_dict["chasseur"]
        obs_chasseur_norm = self.env_chasseur.normalize_obs(obs_chasseur)
        action_chasseur, _ = self.chasseur_model.predict(obs_chasseur_norm, deterministic=True)

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
