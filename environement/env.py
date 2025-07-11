import gymnasium as gym
from gymnasium import spaces
import numpy as np 
import utils
from deplacement import move

class Affrontement(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode = None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Dict({
            "eviteur" : spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                                        high=np.array([360, 10, 10, 360, 10, 360, 10], dtype=np.float32),
                                        dtype=np.float32),
            "chasseur" : spaces.Box(low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
                                        high=np.array([360, 10, 10, 360, 10], dtype=np.float32),
                                        dtype=np.float32)
        })
        self.action_space = spaces.Dict({
            "eviteur" : spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "chasseur" : spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        })
        self.reset()

    def reset(self, seed=None, options=None):
        self.pos_objectif = np.random.uniform(600 , 1000, 2)
        self.pos_eviteur = np.random.uniform(0 , 200, 2)
        self.traj_eviteur = [5, utils.comp_cap(self.pos_eviteur, self.pos_objectif)]
        self.pos_chasseur = self.init_chasseur()
        self.traj_chasseur = [5, utils.comp_cap(self.pos_chasseur, self.pos_eviteur)]
        self.current_step = 0

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        cap_eviteur = self.traj_eviteur[1]
        vit_eviteur = self.traj_eviteur[0]
        dist_obj = np.linalg.norm(self.pos_eviteur - self.pos_objectif)
        cap_obj = utils.comp_cap(self.pos_eviteur, self.pos_objectif)
        dist_chasseur = np.linalg.norm(self.pos_eviteur - self.pos_chasseur)
        cap_vers_chasseur = utils.angle_entre_cap_and_enemy(self.pos_eviteur, self.pos_chasseur, self.traj_eviteur[1])
        cap_chasseur = self.traj_chasseur[1]
        vit_chasseur = self.traj_chasseur[0]
        dist_eviteur = np.linalg.norm(self.pos_chasseur - self.pos_eviteur)
        cap_vers_eviteur = utils.angle_entre_cap_and_enemy(self.pos_chasseur, self.pos_eviteur, self.traj_chasseur[1])

        # Eviteur
        eviteur_obs = np.array([cap_eviteur, vit_eviteur, dist_obj, cap_obj, dist_chasseur, cap_vers_chasseur, 
                                cap_chasseur, vit_chasseur], dtype=np.float32)
        
        # Chasseur
        chasseur_obs = np.array([cap_chasseur, vit_chasseur, dist_eviteur, cap_vers_eviteur, cap_eviteur, vit_eviteur],
                                 dtype=np.float32)

        return {"eviteur": eviteur_obs, "chasseur": chasseur_obs}


    def step(self, action):
        self.agent_pos = np.clip(self.agent_pos + action * 0.1, -1, 1)
        reward = -np.linalg.norm(self.agent_pos - self.enemy_pos)  # se rapprocher de l'ennemi
        terminated = False
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

    def step(self, action_dict):
        self.current_step +=1
        
        # Memoire
        dist_evit_obj_mem = np.linalg.norm(self.pos_eviteur - self.pos_objectif)
        dist_cha_evit_mem = np.linalg.norm(self.pos_chasseur - self.pos_eviteur)

        # Déplacement des navire 
        move(self.pos_eviteur[0], self.pos_eviteur[1], self.traj_eviteur[0], self.traj_eviteur[1], action_dict["eviteur"][0], action_dict["eviteur"][1], 10)
        move(self.pos_chasseur[0], self.pos_chasseur[1], self.traj_chasseur[0], self.traj_chasseur[1], action_dict["chasseur"][0], action_dict["chasseur"][1], 10)

        # Calcul des rewards
        if np.linalg.norm(self.pos_eviteur - self.pos_objectif) < 30:
            terminated = {"eviteur": True, "chasseur": True}
            truncated = {"eviteur": False, "chasseur": False}
            rewards = {"eviteur": 100, "chasseur": -30}

        elif np.linalg.norm(self.pos_chasseur - self.pos_eviteur) < 30:
            terminated = {"eviteur": True, "chasseur": True}
            truncated = {"eviteur": False, "chasseur": False}
            rewards = {"eviteur": -30, "chasseur": 100}

        else:
            terminated = {"eviteur": False, "chasseur": False}
            truncated = {"eviteur": False, "chasseur": False}
            
        


        # terminated = {"agent_a": False, "agent_b": False}
        # truncated = {"agent_a": False, "agent_b": False}


        obs = self._get_obs()
        infos = {}
        return obs, rewards, terminated, truncated, infos
    
    def step(self, action):
        self.current_step += 1
        prev_dist = utils.distance(self.pos_joueur, self.pos_cible)

        self._apply_action(action)
        self._move_ships()

        obs = self._get_obs()

        done, reward, _ = self._compute_reward(action, prev_dist)

        # Ajout d'un limitaion sur le nombre max de step par épisode pour ne pas avoir des episode a 7000 step en début d'entrainement
        if self.current_step > self.max_step:
            done = True

        terminated = done
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def init_chasseur():
        while True:
            pos = np.random.uniform(0, 1000, 2)
            if pos[0] > 600 and pos[1] > 600:
                break
        return pos