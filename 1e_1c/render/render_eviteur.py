import os
import sys

sys.path.append(os.path.abspath(r"C:\Users\FX643778\Documents\Git\Interception_marl\1e_1c\environement"))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import Affrontement
from env_eviteur import AffrontementSingleEviteur
from policy_loader import load_chasseur_policy

# === Chargement du modèle eviteur ===
chasseur_model, chasseur_env = load_chasseur_policy()

# === Environement de l'éviteur ===
env_raw = Affrontement()
env_single = AffrontementSingleEviteur(
    env_raw,
    chasseur_model=chasseur_model,
    chasseur_env=chasseur_env
)

vec_env = DummyVecEnv([lambda: env_single])
env = VecNormalize.load("C:/Users/FX643778/Documents/Git/Interception_marl/1e_1c/models/eviteur/eviteur3_load_vecnormalize.pkl", vec_env)
env.training = False   
env.norm_reward = False

# Chargement du modèle
path_model = "C:/Users/FX643778/Documents/Git/Interception_marl/1e_1c/models/eviteur/eviteur3_load.zip"
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
    # Grande zone de print
    # print("")
    # print(f"État physique à la step {i}")
    # print(f"  Pos éviteur : {env_raw.pos_eviteur}")
    # print(f"  Pos chasseur : {env_raw.pos_chasseur}")
    # print(f"  Pos objectif : {env_raw.pos_objectif}")
    # print(f"  Cap éviteur : {env_raw.traj_eviteur}")
    # print(f"  Cap chasseur : {env_raw.traj_chasseur}")


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
