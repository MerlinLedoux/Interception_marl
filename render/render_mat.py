import os
import sys

sys.path.append(os.path.abspath(r"C:\Users\FX643778\Documents\Git\Interception_marl\environement"))

import time 
import numpy as np 
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import Affrontement
from env_eviteur import AffrontementSingleAgent

env_raw = Affrontement()
env = AffrontementSingleAgent(env_raw, agent_id="eviteur")

path_model = "C:/Users/FX643778/Documents/Git/Interception_marl/models/V3_norm.zip"
model = PPO.load(path_model)

obs, _ = env.reset()

plt.ion()
fig, ax = plt.subplots()
sc_chasseur, = ax.plot([], [], 'ro', label="Chasseur")  # rouge
sc_eviteur, = ax.plot([], [], 'bo', label="Éviteur")    # bleu
sc_objectif, = ax.plot([], [], 'bo')                    # bleu aussi mais celui la ne bouge pas

ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
ax.set_title("Simulation des agents")
ax.legend()

for _ in range(200):  # nombre de steps à simuler
    # Prédiction de l’action
    action, _ = model.predict(obs, deterministic=True)

    # print(action)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated


    # Récupérer les positions
    pos_chasseur = np.array(env_raw.pos_chasseur)
    pos_eviteur = np.array(env_raw.pos_eviteur)
    pos_objectif = np.array(env_raw.pos_objectif)

    # Mettre à jour les positions sur le graphique
    sc_chasseur.set_data([pos_chasseur[0]], [pos_chasseur[1]])
    sc_eviteur.set_data([pos_eviteur[0]], [pos_eviteur[1]])
    sc_objectif.set_data([pos_objectif[0]], [pos_objectif[1]])
    

    plt.pause(0.05)

    if done:
        print("je suis passer dans le done")
        break

plt.ioff()
plt.show()