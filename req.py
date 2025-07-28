import ray
import torch
import pettingzoo
import gymnasium
import wandb  # facultatif
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
