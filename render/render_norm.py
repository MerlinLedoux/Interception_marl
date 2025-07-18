import os
import sys

sys.path.append(os.path.abspath(r"C:\Users\FX643778\Documents\Git\Interception_marl\environement"))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import Affrontement
from env_eviteur import AffrontementSingleAgent

# Création environnement de base
env_raw = Affrontement()
env_single = AffrontementSingleAgent(env_raw, agent_id="eviteur")

# Vectorisation de l'environnement (nécessaire pour VecNormalize)
vec_env = DummyVecEnv([lambda: env_single])

# Chargement de la normalisation (adaptée au vecteur d'environnement)
# env = VecNormalize.load("C:/Users/FX643778/Documents/Git/Interception_marl/models/V10_3_long_vecnormalize.pkl", vec_env)
env = VecNormalize.load("C:/Users/FX643778/Documents/Git/Interception_marl/models/reward_test/V22_vecnormalize.pkl", vec_env)
env.training = False    
env.norm_reward = False

# Chargement du modèle
path_model = "C:/Users/FX643778/Documents/Git/Interception_marl/models/reward_test/V22_proche_2.zip"
model = PPO.load(path_model, env=env)

# Reset initial
obs = env.reset()


# Initialisation de l'affichage avec matplotlib
plt.ion()
fig, ax = plt.subplots()
sc_chasseur, = ax.plot([], [], 'ro', label="Chasseur")  # rouge
sc_eviteur, = ax.plot([], [], 'bo', label="Éviteur")    # bleu
sc_objectif, = ax.plot([], [], 'go', label="Objectif")  # vert (plus visible)

ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
ax.set_title("Simulation des agents")
ax.legend()

reward_total = 0

for i in range(200):  # nombre de steps à simuler
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    reward_total += reward

    # Récupérer les positions physiques dans env_raw (non normalisées)
    pos_chasseur = np.array(env_raw.pos_chasseur)
    pos_eviteur = np.array(env_raw.pos_eviteur)
    pos_objectif = np.array(env_raw.pos_objectif)

    # Mettre à jour les positions sur le graphique
    sc_chasseur.set_data([pos_chasseur[0]], [pos_chasseur[1]])
    sc_eviteur.set_data([pos_eviteur[0]], [pos_eviteur[1]])
    sc_objectif.set_data([pos_objectif[0]], [pos_objectif[1]])

    plt.pause(0.05)

    if done:
        print(f"Episode terminé en {i} step, le reward total a était : {reward_total[0]}.")
        break

plt.ioff()
plt.show()
