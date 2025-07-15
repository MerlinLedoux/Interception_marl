import gymnasium as gym
import numpy as np

class AffrontementSingleAgent(gym.Wrapper):
    def __init__(self, env, agent_id="eviteur"):
        super().__init__(env)
        assert agent_id in ["eviteur", "chasseur"]
        self.agent_id = agent_id
        self.observation_space = env.observation_space[agent_id]
        self.action_space = env.action_space[agent_id]

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
