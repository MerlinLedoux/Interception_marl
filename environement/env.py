import gymnasium as gym
from gymnasium import spaces
import numpy as np 
import math
import random as rand

class Affrontement(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode = None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Dict({
            "eviteur" : spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                                        high=np.array([360, 10, 10, 360, 10, 360, 10], dtype=np.float32),
                                        dtype=np.float32),
            "chasseur" : spaces.Box(low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
                                        high=np.array([360, 10, 10, 360, 10], dtype=np.float32),
                                        dtype=np.float32)
        })

        self.action_space = spaces.Dict({
            "eviteur" : spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "chasseur" : spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        })

    def reset(self, seed=None, options=None):
        obs = self.get_obs()
        return obs, {}

    def get_obs(self):
        return 0

    def step(self, action_dict):
        obs = {
            "agent_a": np.random.rand(4).astype(np.float32),
            "agent_b": np.random.rand(4).astype(np.float32)
        }
        rewards = {
            "agent_a": 1.0,
            "agent_b": -1.0
        }
        terminated = {"agent_a": False, "agent_b": False}
        truncated = {"agent_a": False, "agent_b": False}
        infos = {}
        return obs, rewards, terminated, truncated, infos

    def render(self):
        pass