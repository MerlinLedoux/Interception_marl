import gymnasium as gym
import numpy as np
from deplacement import chasseur_simple, chasseur_moyen, chasseur_moyen_2, chasseur_moyen_3, chasseur_hard

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

        comportement = self.env.id % 100
        comportement = comportement // 25
        comportement = 4
        # print(comportement)

        match comportement:
            case 0:
                action_dict["chasseur"] = chasseur_simple(self.env.pos_chasseur[0], self.env.pos_chasseur[1], self.env.traj_chasseur[1], self.env.pos_eviteur[0], self.env.pos_eviteur[1])
            case 1:
                action_dict["chasseur"] = chasseur_moyen(self.env.pos_chasseur[0], self.env.pos_chasseur[1], self.env.traj_chasseur[1], self.env.pos_eviteur[0], self.env.pos_eviteur[1], self.env.traj_eviteur[1])
            case 2 :
                action_dict["chasseur"] = chasseur_moyen_2(self.env.pos_chasseur[0], self.env.pos_chasseur[1], self.env.traj_chasseur[1], self.env.traj_chasseur[0], self.env.pos_eviteur[0], self.env.pos_eviteur[1], self.env.traj_eviteur[1], self.env.traj_eviteur[0])
            case 3:
                action_dict["chasseur"] = chasseur_moyen_3(self.env.pos_chasseur[0], self.env.pos_chasseur[1], self.env.traj_chasseur[1], self.env.traj_chasseur[0], self.env.pos_eviteur[0], self.env.pos_eviteur[1], self.env.traj_eviteur[1], self.env.traj_eviteur[0])
            case 4:
                action_dict["chasseur"] = chasseur_hard(self.env.pos_chasseur[0], self.env.pos_chasseur[1], self.env.traj_chasseur[1], self.env.traj_chasseur[0], self.env.pos_eviteur[0], self.env.pos_eviteur[1], self.env.traj_eviteur[1], self.env.traj_eviteur[0])
           
        action_dict[self.agent_id] = action

        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        return (
            obs[self.agent_id],
            reward[self.agent_id],
            terminated[self.agent_id],
            truncated[self.agent_id],
            info.get(self.agent_id, {})
        )
