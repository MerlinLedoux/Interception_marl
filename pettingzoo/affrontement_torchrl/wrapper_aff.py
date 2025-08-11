import torch
from torchrl.envs import EnvBase
from torchrl.data import CompositeSpec, BoundedTensorSpec
from pettingzoo.utils import wrappers
from typing import Optional
import numpy as np

class CustomPettingZooWrapper(EnvBase):
    def __init__(self, env, device="cpu"):
        super().__init__(device=device)

        self.env = wrappers.order_enforcing(env)
        self.possible_agents = self.env.possible_agents
        self.device = device

        self.agent_idx = 0
        self.current_agent = self.possible_agents[self.agent_idx]

        # Sample to get dimensions
        sample_obs = self.env.observation_space(self.current_agent).sample()
        sample_action = self.env.action_space(self.current_agent).sample()

        obs_dim = sample_obs.shape[0] if isinstance(sample_obs, np.ndarray) else 1
        act_dim = sample_action.shape[0] if isinstance(sample_action, np.ndarray) else 1

        obs_spec = CompositeSpec({
            agent: CompositeSpec({
                "observation": BoundedTensorSpec(
                    shape=torch.Size([obs_dim]),
                    minimum=-float("inf"),
                    maximum=float("inf"),
                    dtype=torch.float32,
                    device=device
                )
            }) for agent in self.possible_agents
        })

        act_spec = CompositeSpec({
            agent: CompositeSpec({
                "action": BoundedTensorSpec(
                    shape=torch.Size([act_dim]),
                    minimum=torch.tensor(self.env.action_space(agent).low),
                    maximum=torch.tensor(self.env.action_space(agent).high),
                    dtype=torch.float32,
                    device=device
                )
            }) for agent in self.possible_agents
        })

        self.observation_spec = obs_spec
        self.action_spec = act_spec

    def _set_seed(self, seed: Optional[int]):
        # Implémentation obligatoire
        if hasattr(self.env, "seed"):
            self.env.seed(seed)

    def _reset(self, tensordict):
        self.env.reset()
        self.agent_idx = 0
        self.current_agent = self.possible_agents[self.agent_idx]

        for agent in self.possible_agents:
            obs = self.env.observe(agent)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            tensordict.set(("next", agent, "observation"), obs_tensor)
            tensordict.set(("next", agent, "reward"), torch.tensor(0.0))
            tensordict.set(("next", agent, "done"), torch.tensor(False))
        return tensordict

    def _step(self, tensordict):
        for agent in self.possible_agents:
            if self.env.dones[agent]:
                continue

            action = tensordict.get(("action", agent, "action")).cpu().numpy()
            self.env.step(action)

            obs = self.env.observe(agent)
            reward = self.env.rewards[agent]
            done = self.env.dones[agent]

            tensordict.set(("next", agent, "observation"),
                           torch.tensor(obs, dtype=torch.float32, device=self.device))
            tensordict.set(("next", agent, "reward"),
                           torch.tensor(reward, dtype=torch.float32, device=self.device))
            tensordict.set(("next", agent, "done"),
                           torch.tensor(done, dtype=torch.bool, device=self.device))
        return tensordict
