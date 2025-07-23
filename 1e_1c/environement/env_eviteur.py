import gymnasium as gym

class AffrontementSingleEviteur(gym.Wrapper):
    def __init__(self, env, chasseur_model=None, chasseur_env=None):
        super().__init__(env)

        self.agent_id = "eviteur"
        self.observation_space = env.observation_space[self.agent_id]
        self.action_space = env.action_space[self.agent_id]

        self.chasseur_env = chasseur_env
        self.chasseur_model = chasseur_model


    def reset(self, **kwargs):
        obs_dict, _ = self.env.reset(**kwargs)
        return obs_dict[self.agent_id], {}

    def step(self, action_eviteur):
        obs_dict = self.env._get_obs()
        obs_chasseur = obs_dict["chasseur"]
        action_chasseur = [0,0]
        if self.chasseur_model :
            obs_chasseur_norm = self.chasseur_env.normalize_obs(obs_chasseur)
            action_chasseur, _ = self.chasseur_model.predict(obs_chasseur_norm, deterministic=True)

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
