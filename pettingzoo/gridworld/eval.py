from my_env import GridWorldEnv
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

env = GridWorldEnv()
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

model = PPO.load("ppo_gridworld")

obs = env.reset()
for _ in range(10):
    action, _states = model.predict(obs)
    obs, rewards, dones, infos = env.step(action)
    env.render()
