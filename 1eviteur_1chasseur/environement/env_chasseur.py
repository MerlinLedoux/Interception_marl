import gymnasium as gym

class AffrontementSingleChasseur(gym.Wrapper):
    def __init__(self, env, eviteur_model=None, eviteur_env=None):
        super().__init__(env)

        self.agent_id = "chasseur"
        self.observation_space = env.observation_space[self.agent_id]
        self.action_space = env.action_space[self.agent_id]

        self.env_eviteur = eviteur_env
        self.eviteur_model = eviteur_model


    def reset(self, **kwargs):
        obs_dict, _ = self.env.reset(**kwargs)
        return obs_dict[self.agent_id], {}

    def step(self, action_chasseur):
        obs_dict = self.env._get_obs()
        obs_eviteur = obs_dict["eviteur"]

        action_eviteur = [1,0]
        if self.eviteur_model:
            obs_eviteur_norm = self.env_eviteur.normalize_obs(obs_eviteur)
            action_eviteur, _ = self.eviteur_model.predict(obs_eviteur_norm, deterministic=True)

        action_dict = {
            "eviteur": action_eviteur,
            "chasseur": action_chasseur
        }

        obs_dict, reward_dict, terminated, truncated, info_dict = self.env.step(action_dict)

        return (
            obs_dict[self.agent_id],
            reward_dict[self.agent_id],
            terminated[self.agent_id],
            truncated[self.agent_id],
            info_dict.get(self.agent_id, {})
        )
