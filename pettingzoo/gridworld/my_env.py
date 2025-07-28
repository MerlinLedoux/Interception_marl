from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np


class GridWorldEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "grid_world_v0"}

    def __init__(self, grid_size=5, render_mode=None):
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.agents = self.possible_agents.copy()
        self.pos = {}
        self.target = (grid_size - 1, grid_size - 1)

        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.float32)
            for agent in self.possible_agents
        }

    @property
    def possible_agents(self):
        return ["agent_0", "agent_1"]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        self.pos = {
            "agent_0": np.array([0, 0]),
            "agent_1": np.array([0, self.grid_size - 1])
        }
        observations = {agent: self.pos[agent] for agent in self.agents}
        return observations, {}

    def step(self, actions):

        print(actions)

        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        for agent, action in actions.items():
            x, y = self.pos[agent]
            if action == 0:  # haut
                x = max(0, x - 1)
            elif action == 1:  # bas
                x = min(self.grid_size - 1, x + 1)
            elif action == 2:  # gauche
                y = max(0, y - 1)
            elif action == 3:  # droite
                y = min(self.grid_size - 1, y + 1)
            # action 4 = rien
            self.pos[agent] = np.array([x, y])

            if (x, y) == self.target:
                rewards[agent] = 1.0
                terminations[agent] = True

        observations = {agent: self.pos[agent] for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        tx, ty = self.target
        grid[tx][ty] = "X"
        for agent, pos in self.pos.items():
            x, y = pos
            grid[x][y] = agent[-1]
        print("\n".join(" ".join(row) for row in grid))
        print()
