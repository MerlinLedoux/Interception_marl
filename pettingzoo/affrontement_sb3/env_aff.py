import functools

from pettingzoo import ParallelEnv 
from gymnasium import spaces 
import numpy as np
import utils
import deplacement
import matplotlib.pyplot as plt

class DoubleChasseur(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "double_chasseur"}

    def __init__(self, render_mode=None):
        self.agents = ["chasseur1", "chasseur2"]    # Pour l'instant je ne met que les chasseur l'entrainement de l'éviteur ce sera pour plus tard
        self.possible_agents = self.agents[:]

        self.observation_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32) for agent in self.agents
        }

        self.action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) for agent in self.agents    
        }

        self.id = 0
        self.max_step = 200
        self.render_mode = render_mode
        self.render_initialized = False

    def reset(self, seed=None, options=None):
        self.id += 1 
        self.pos_objectif = np.random.uniform(600 , 1000, 2)
        self.pos_eviteur = np.random.uniform(0 , 200, 2)
        self.traj_eviteur = [5, utils.comp_cap(self.pos_eviteur, self.pos_objectif)]
        
        self.pos_chasseurs = []
        self.traj_chasseurs = []
        for k in range(2):
            self.pos_chasseurs.append(self.init_chasseur())
            self.traj_chasseurs.append([5, utils.comp_cap(self.pos_chasseurs[k], self.pos_eviteur)])    
        
        self.current_step = 0
        self.id += 1
        self.dist_min = 10000
        for k in range(2):
            self.dist_min = min(self.dist_min, np.linalg.norm(self.pos_eviteur - self.pos_chasseurs[k]))

        obs = self._get_obs()
        return obs, {}
    
    def _get_obs(self):

        # ========== Global ========== #
        vit_ev = self.traj_eviteur[0]
        cap_ev = self.traj_eviteur[1]
        vit_cha1 = self.traj_chasseurs[0][0]
        cap_cha1 = self.traj_chasseurs[0][1]
        vit_cha2 = self.traj_chasseurs[1][0]
        cap_cha2 = self.traj_chasseurs[1][1]        

        # ========== Eviteur ========== #
        # dist_obj = utils.red_dist(np.linalg.norm(self.pos_eviteur - self.pos_objectif))
        # cap_obj = utils.comp_cap(self.pos_eviteur, self.pos_objectif)
        dist_ev_cha1 = utils.red_dist(np.linalg.norm(self.pos_eviteur - self.pos_chasseurs[0]))
        dist_ev_cha2 = utils.red_dist(np.linalg.norm(self.pos_eviteur - self.pos_chasseurs[1]))
        # cap_ev_cha1 = utils.angle_entre_cap_and_enemy(self.pos_eviteur, self.pos_chasseurs[0], self.traj_eviteur[1])
        # cap_ev_cha2 = utils.angle_entre_cap_and_enemy(self.pos_eviteur, self.pos_chasseurs[1], self.traj_eviteur[1])

        # eviteur_obs = np.array([cap_ev, vit_ev, dist_obj, cap_obj, dist_ev_cha1, cap_ev_cha1, cap_cha1, vit_cha1, 
        #                         dist_ev_cha2, cap_ev_cha2, cap_cha2, vit_cha2], dtype=np.float32)

        # ========== Chasseurs_1 ========== #
        cap_cha1_ev = utils.angle_entre_cap_and_enemy(self.pos_chasseurs[0], self.pos_eviteur, self.traj_chasseurs[0][1])
        dist_cha1_cha2 = utils.red_dist(np.linalg.norm(self.pos_chasseurs[0] - self.pos_chasseurs[1]))
        cap_cha1_cha2 = utils.angle_entre_cap_and_enemy(self.pos_chasseurs[0], self.pos_chasseurs[1], self.traj_chasseurs[0][1])

        chasseur_obs_1 = np.array([cap_cha1, vit_cha1, dist_ev_cha1, cap_cha1_ev, cap_ev, vit_ev, 
                                   dist_cha1_cha2, cap_cha1_cha2, cap_cha2, vit_cha2], dtype=np.float32)

        chasseur_obs_1_norm = utils.normaliser_chasseur(chasseur_obs_1)

        # ========== Chasseurs_2 ========== #
        cap_cha2_ev = utils.angle_entre_cap_and_enemy(self.pos_chasseurs[1], self.pos_eviteur, self.traj_chasseurs[1][1])
        cap_cha2_cha1 = utils.angle_entre_cap_and_enemy(self.pos_chasseurs[1], self.pos_chasseurs[0], self.traj_chasseurs[1][1])

        chasseur_obs_2 = np.array([cap_cha2, vit_cha2, dist_ev_cha2, cap_cha2_ev, cap_ev, vit_ev, 
                                   dist_cha1_cha2, cap_cha2_cha1, cap_cha1, vit_cha1], dtype=np.float32)
        
        chasseur_obs_2_norm = utils.normaliser_chasseur(chasseur_obs_2)

        # return {"eviteur": eviteur_obs, "chasseur1": chasseur_obs_1, "chasseur2": chasseur_obs_2}
        return {"chasseur1": chasseur_obs_1_norm, "chasseur2": chasseur_obs_2_norm}
    

    def step(self, actions):
        self.current_step +=1
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Déplacement des navire 
        actions["chasseur1"] = np.clip(actions["chasseur1"], -1.0, 1.0)
        actions["chasseur2"] = np.clip(actions["chasseur2"], -1.0, 1.0)


        # Je suis en train de faire des test pour voir comment l'agent reagit si je cap la vitesst de l'eviteur plus bas
        nxe, nye, nce, nve = deplacement.move(self.pos_eviteur[0], self.pos_eviteur[1], self.traj_eviteur[1], self.traj_eviteur[0], 1, 0, 8)
        nxc1, nyc1, ncc1, nvc1 = deplacement.move(self.pos_chasseurs[0][0], self.pos_chasseurs[0][1], self.traj_chasseurs[0][1], self.traj_chasseurs[0][0], actions["chasseur1"][0], actions["chasseur1"][1], 8)
        nxc2, nyc2, ncc2, nvc2 = deplacement.move(self.pos_chasseurs[1][0], self.pos_chasseurs[1][1], self.traj_chasseurs[1][1], self.traj_chasseurs[1][0], actions["chasseur2"][0], actions["chasseur2"][1], 8)

        self.pos_eviteur = np.array([nxe, nye])
        self.traj_eviteur = np.array([nve, nce])
        self.pos_chasseurs[0] = np.array([nxc1, nyc1])
        self.traj_chasseurs[0] = np.array([nvc1, ncc1])
        self.pos_chasseurs[1] = np.array([nxc2, nyc2])
        self.traj_chasseurs[1] = np.array([nvc2, ncc2])

        temp1 = np.linalg.norm(self.pos_eviteur - self.pos_chasseurs[0])
        temp2 = np.linalg.norm(self.pos_eviteur - self.pos_chasseurs[1])
        self.dist_min = min(self.dist_min, temp1, temp2)

        # Check nombre de step
        if self.current_step >= self.max_step:
            truncations = {"chasseur1": True, "chasseur2": True}
            terminations = {"chasseur1": False, "chasseur2": False}
            rewards = {"chasseur1": -1, "chasseur2": -1}
        else :
            truncations = {"chasseur1": False, "chasseur2": False}
            # Calcul des rewards
            if np.linalg.norm(self.pos_eviteur - self.pos_objectif) < 30:

                if self.dist_min > 150:
                    proche = -5
                elif self.dist_min < 50:
                    proche = 5
                else:
                    proche = -((self.dist_min / 10) - 10)

                terminations = {"chasseur1": True, "chasseur2": True}
                rewards = {"chasseur1": (-10 + proche)/10, "chasseur2": (-10 + proche)/10}

            elif (np.linalg.norm(self.pos_chasseurs[0] - self.pos_eviteur) < 30 or np.linalg.norm(self.pos_chasseurs[1] - self.pos_eviteur) < 30):
                terminations = {"chasseur1": True, "chasseur2": True}
                rewards = {"chasseur1": 1, "chasseur2": 1}

            else:
                terminations = {"chasseur1": False, "chasseur2": False}
                rewards = {"chasseur1": -1/200, "chasseur2": -1/200}
       
        obs = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        return obs, rewards, terminations, truncations, infos


    def init_chasseur(self):
        while True:
            pos = np.random.uniform(0, 1000, 2)
            if pos[0] > 600 or pos[1] > 600:
                break
        return pos


    def render(self):
        if not self.render_initialized:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.sc_eviteur, = self.ax.plot([], [], 'bo', label="Éviteur")
            self.sc_chasseurs, = self.ax.plot([], [], 'ro', label="Chasseurs")
            self.ax.set_xlim(0, 1000)
            self.ax.set_ylim(0, 1000)
            self.ax.set_title("Positions des agents")
            self.ax.legend()
            self.render_initialized = True

        # Récupérer positions
        eviteur = self.pos_eviteur
        chasseurs = self.pos_chasseurs  # liste de tuples ou np.array

        chasseurs = np.array(chasseurs)

        # Mettre à jour les positions
        self.sc_eviteur.set_data([eviteur[0]], [eviteur[1]])
        self.sc_chasseurs.set_data(chasseurs[:, 0], chasseurs[:, 1])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)

    def close(self):
        print("Fermeture de l'environnement")