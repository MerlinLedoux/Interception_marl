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
        env_raw = Affrontement()
        env_single_eviteur = AffrontementSingleEviteur(env_raw, agent_id="eviteur")
        vec_env = DummyVecEnv([lambda: env_single_eviteur])

        self.eviteur_norm = VecNormalize.load("C:/Users/FX643778/Documents/Git/Interception_marl/models/reward_test/V22_vecnormalize.pkl", vec_env)
        self.eviteur_norm.training = False    
        self.eviteur_norm.norm_reward = False

        path_model = "C:/Users/FX643778/Documents/Git/Interception_marl/models/reward_test/V22_proche_2.zip"
        self.eviteur_model = PPO.load(path_model, env=self.eviteur_norm)

    
    def _get_obs(self):
        base_env = self.unwrapped  # pour éviter les surprises avec plusieurs wrappers

        cap_chasseur = base_env.traj_chasseur[1]
        vit_chasseur = base_env.traj_chasseur[0]
        dist_eviteur = utils.red_dist(np.linalg.norm(base_env.pos_chasseur - base_env.pos_eviteur))
        cap_vers_eviteur = utils.angle_entre_cap_and_enemy(base_env.pos_chasseur, base_env.pos_eviteur, base_env.traj_chasseur[1])
        cap_eviteur = base_env.traj_eviteur[1]
        vit_eviteur = base_env.traj_eviteur[0]

        return np.array([cap_chasseur, vit_chasseur, dist_eviteur, cap_vers_eviteur, cap_eviteur, vit_eviteur], dtype=np.float32)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self._get_obs(), {}

    def step(self, action):

        action_dict = {
            "eviteur": np.array([0.0, 0.0], dtype=np.float32),
            "chasseur": np.array([0.0, 0.0], dtype=np.float32)
        }

        base_env = self.unwrapped
        obs_eviteur_raw = base_env._get_obs()["eviteur"]
        obs_eviteur = self.eviteur_norm.normalize_obs(obs_eviteur_raw) 
        action_eviteur, _ = self.eviteur_model.predict(obs_eviteur, deterministic=True)

        action_dict["eviteur"] = action_eviteur 
        action_dict["chasseur"] = action

        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        return (
            obs[self.agent_id],
            reward[self.agent_id],
            terminated[self.agent_id],
            truncated[self.agent_id],
            info.get(self.agent_id, {})
        )
