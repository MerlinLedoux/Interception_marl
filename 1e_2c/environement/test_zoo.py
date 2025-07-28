import numpy as np
from petting_zoo import AffrontementMultiZoo
import matplotlib.pyplot as plt

# Crée l'environnement
env = AffrontementMultiZoo()
obs, _ = env.reset()

# Affiche les observations initiales
print("Observations initiales :")
for agent, ob in obs.items():
    print(f"{agent} → {ob}")

# Exécution d'un épisode de test
done = {agent: False for agent in env.agents}
terminated = False
rewards_history = {agent: [] for agent in env.agents}

while not all(done.values()):
    # Génère une action aléatoire par agent
    actions = {
        agent: env.action_spaces[agent].sample()
        for agent in env.agents if not done[agent]
    }

    obs, rewards, terminations, truncations, infos = env.step(actions)

    # Affiche les récompenses
    for agent in env.agents:
        rewards_history[agent].append(rewards[agent])
        if terminations[agent] or truncations[agent]:
            done[agent] = True

# Affiche les courbes de récompense
plt.figure()
for agent in env.agents:
    plt.plot(rewards_history[agent], label=agent)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.legend()
plt.title("Récompenses des agents")
plt.show()
