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
        self.max_step = 500

        self.observation_space = spaces.Dict({
            "eviteur" : spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                                        high=np.array([360, 10, 10, 360, 10, 360, 360, 10], dtype=np.float32),
                                        dtype=np.float32),
            "chasseur" : spaces.Box(low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
                                        high=np.array([360, 10, 10, 360, 360, 10], dtype=np.float32),
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
        dist_obj = utils.red_dist(np.linalg.norm(self.pos_eviteur - self.pos_objectif))
        cap_obj = utils.comp_cap(self.pos_eviteur, self.pos_objectif)
        dist_chasseur = utils.red_dist(np.linalg.norm(self.pos_eviteur - self.pos_chasseur))
        cap_vers_chasseur = utils.angle_entre_cap_and_enemy(self.pos_eviteur, self.pos_chasseur, self.traj_eviteur[1])
        cap_chasseur = self.traj_chasseur[1]
        vit_chasseur = self.traj_chasseur[0]
        dist_eviteur = utils.red_dist(np.linalg.norm(self.pos_chasseur - self.pos_eviteur))
        cap_vers_eviteur = utils.angle_entre_cap_and_enemy(self.pos_chasseur, self.pos_eviteur, self.traj_chasseur[1])

        # Eviteur
        eviteur_obs = np.array([cap_eviteur, vit_eviteur, dist_obj, cap_obj, dist_chasseur, cap_vers_chasseur, 
                                cap_chasseur, vit_chasseur], dtype=np.float32)
        
        # Chasseur
        chasseur_obs = np.array([cap_chasseur, vit_chasseur, dist_eviteur, cap_vers_eviteur, cap_eviteur, vit_eviteur],
                                 dtype=np.float32)

        return {"eviteur": eviteur_obs, "chasseur": chasseur_obs}


    def step(self, action_dict):
        self.current_step +=1
        
        # Memoire
        dist_evit_obj_mem = np.linalg.norm(self.pos_eviteur - self.pos_objectif)
        dist_cha_evit_mem = np.linalg.norm(self.pos_chasseur - self.pos_eviteur)

        # Déplacement des navire 
        nxe, nye, nce, nve = move(self.pos_eviteur[0], self.pos_eviteur[1], self.traj_eviteur[0], self.traj_eviteur[1], action_dict["eviteur"][0], action_dict["eviteur"][1], 10)
        nxc, nyc, ncc, nvc = move(self.pos_chasseur[0], self.pos_chasseur[1], self.traj_chasseur[0], self.traj_chasseur[1], action_dict["chasseur"][0], action_dict["chasseur"][1], 10)

        self.pos_eviteur = nxe, nye
        self.traj_eviteur = nve, nce 
        self.pos_chasseur = nxc, nyc
        self.traj_chasseur = nvc, ncc

        # Check nombre de step
        if self.current_step >= self.max_steps:
            truncated = {"eviteur": True, "chasseur": True}
            terminated = {"eviteur": False, "chasseur": False}
            rewards = {"eviteur": 0, "chasseur": 0}
        else :
            truncated = {"eviteur": False, "chasseur": False}
            # Calcul des rewards
            if np.linalg.norm(self.pos_eviteur - self.pos_objectif) < 30:
                terminated = {"eviteur": True, "chasseur": True}
                rewards = {"eviteur": 100, "chasseur": -30}

            elif np.linalg.norm(self.pos_chasseur - self.pos_eviteur) < 30:
                terminated = {"eviteur": True, "chasseur": True}
                rewards = {"eviteur": -30, "chasseur": 100}

            else:
                terminated = {"eviteur": False, "chasseur": False}
                dist_evit_obj_new = np.linalg.norm(self.pos_eviteur - self.pos_objectif)
                dist_cha_evit_new = np.linalg.norm(self.pos_chasseur - self.pos_eviteur)
                rewards = {"eviteur": dist_evit_obj_mem - dist_evit_obj_new, "chasseur": dist_cha_evit_mem - dist_cha_evit_new}
       
        obs = self._get_obs()
        infos = {}
        return obs, rewards, terminated, truncated, infos
    

    def render(self):
        pass

    def init_chasseur(self):
        while True:
            pos = np.random.uniform(0, 1000, 2)
            if pos[0] > 600 and pos[1] > 600:
                break
        return pos