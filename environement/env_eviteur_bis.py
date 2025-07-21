import gymnasium as gym
import numpy as np
from deplacement import chasseur_simple, chasseur_moyen, chasseur_moyen_2, chasseur_moyen_3, chasseur_hard

class AffrontementSingleEviteur(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.agent_id = "eviteur"
        self.observation_space = env.observation_space[self.agent_id]
        self.action_space = env.action_space[self.agent_id]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[self.agent_id], info.get(self.agent_id, {})

    def step(self, action):
        # Création d'un dictionnaire d'action pour les deux agents
        action_dict = {
            "eviteur": np.array([0.0, 0.0], dtype=np.float32),
            "chasseur": np.array([0.0, 0.0], dtype=np.float32)
        }

        
           
        action_dict[self.agent_id] = action

        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        return (
            obs[self.agent_id],
            reward[self.agent_id],
            terminated[self.agent_id],
            truncated[self.agent_id],
            info.get(self.agent_id, {})
        )

    