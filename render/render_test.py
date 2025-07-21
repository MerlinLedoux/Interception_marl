import os
import sys

sys.path.append(os.path.abspath(r"C:\Users\FX643778\Documents\Git\Interception_marl\environement"))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import Affrontement
from env_eviteur import AffrontementSingleEviteur

env_raw = Affrontement()
env_eviteur = AffrontementSingleEviteur(env_raw)

vec_env = DummyVecEnv([lambda: env_eviteur])

env = VecNormalize.load("C:/Users/FX643778/Documents/Git/Interception_marl/models/reward_test/V22_vecnormalize.pkl", vec_env)
env.training = False    
env.norm_reward = False

path_model = "C:/Users/FX643778/Documents/Git/Interception_marl/models/reward_test/V22_proche_2.zip"
model = PPO.load(path_model, env=env)


obs = env.reset()
# print(obs)

env_raw.pos_eviteur =  np.array([358.48484312, 239.31763515])
env_raw.pos_chasseur = np.array([ 478.4415672,  -408.53484804])
env_raw.pos_objectif = np.array([918.73578221, 639.77963289])
env_raw.traj_eviteur = np.array([ 10.53226895, 245.73849697])
env_raw.traj_chasseur = np.array([  8.,         281.13294597])


# print("État physique :")
# print(f"  Pos éviteur : {env_raw.pos_eviteur}")
# print(f"  Pos chasseur : {env_raw.pos_chasseur}")
# print(f"  Pos objectif : {env_raw.pos_objectif}")
# print(f"  Cap éviteur : {env_raw.traj_eviteur}")
# print(f"  Cap chasseur : {env_raw.traj_chasseur}")

# obs = np.array([[ -0.6461649, -5.6467795, 1.8473285, -0.8174245, 2.893185, 3.3805532, 0.35541624, -10]])

obs = env._get_obs()
print(obs)
action, _ = model.predict(obs, deterministic=True)
print(action)

###=============Test et print=============###

# for k in range(200):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
    
#     if done:
#         print(k)


###=============Test environement du chasseur=============###

# env_raw2 = Affrontement()
# env_eviteur2 = AffrontementSingleEviteur(env_raw2)
# vec_env2 = DummyVecEnv([lambda: env_eviteur2])

# env_eviteur2 = VecNormalize.load("C:/Users/FX643778/Documents/Git/Interception_marl/models/reward_test/V22_vecnormalize.pkl", vec_env)
# env_eviteur2.training = False    
# env_eviteur2.norm_reward = False

# path_model = "C:/Users/FX643778/Documents/Git/Interception_marl/models/reward_test/V22_proche_2.zip"
# eviteur_model2 = PPO.load(path_model, env=env_eviteur2)

# action, _ = eviteur_model2.predict(obs, deterministic=True)
# print(action)