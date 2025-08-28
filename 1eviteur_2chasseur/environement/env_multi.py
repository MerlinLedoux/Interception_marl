import gymnasium as gym
from gymnasium import spaces
import numpy as np 
import utils
import deplacement

class AffrontementMulti(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode = None):
        super().__init__()
        self.render_mode = render_mode
        self.max_step = 200
        self.id = 0
        self.nbr_chasseurs = 2

        self.observation_space = spaces.Dict({
            "eviteur" : spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                                        high=np.array([360, 10, 10, 360, 10, 360, 360, 10, 10, 360, 360, 10], dtype=np.float32),
                                        dtype=np.float32),

            "chasseur1" : spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                                        high=np.array([360, 10, 10, 360, 360, 10, 10, 360, 360, 10], dtype=np.float32),
                                        dtype=np.float32),

            "chasseur2" : spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                                        high=np.array([360, 10, 10, 360, 360, 10, 10, 360, 360, 10], dtype=np.float32),
                                        dtype=np.float32)
        })
        
        self.action_space = spaces.Dict({
            "eviteur" : spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "chasseur1" : spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "chasseur2" : spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        })
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos_objectif = np.random.uniform(600 , 1000, 2)
        self.pos_eviteur = np.random.uniform(0 , 200, 2)
        self.traj_eviteur = [5, utils.comp_cap(self.pos_eviteur, self.pos_objectif)]
        
        self.pos_chasseurs = []
        self.traj_chasseurs = []
        for k in range(self.nbr_chasseurs):
            self.pos_chasseurs.append(self.init_chasseur())
            self.traj_chasseurs.append([5, utils.comp_cap(self.pos_chasseurs[k], self.pos_eviteur)])    
        
        self.current_step = 0
        self.id += 1
        self.dist_min = 10000
        for k in range(self.nbr_chasseurs):
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
        dist_obj = utils.red_dist(np.linalg.norm(self.pos_eviteur - self.pos_objectif))
        cap_obj = utils.comp_cap(self.pos_eviteur, self.pos_objectif)
        dist_ev_cha1 = utils.red_dist(np.linalg.norm(self.pos_eviteur - self.pos_chasseurs[0]))
        dist_ev_cha2 = utils.red_dist(np.linalg.norm(self.pos_eviteur - self.pos_chasseurs[1]))
        cap_ev_cha1 = utils.angle_entre_cap_and_enemy(self.pos_eviteur, self.pos_chasseurs[0], self.traj_eviteur[1])
        cap_ev_cha2 = utils.angle_entre_cap_and_enemy(self.pos_eviteur, self.pos_chasseurs[1], self.traj_eviteur[1])

        eviteur_obs = np.array([cap_ev, vit_ev, dist_obj, cap_obj, dist_ev_cha1, cap_ev_cha1, cap_cha1, vit_cha1, 
                                dist_ev_cha2, cap_ev_cha2, cap_cha2, vit_cha2], dtype=np.float32)

        # ========== Chasseurs_1 ========== #
        cap_cha1_ev = utils.angle_entre_cap_and_enemy(self.pos_chasseurs[0], self.pos_eviteur, self.traj_chasseurs[0][1])
        dist_cha1_cha2 = utils.red_dist(np.linalg.norm(self.pos_chasseurs[0] - self.pos_chasseurs[1]))
        cap_cha1_cha2 = utils.angle_entre_cap_and_enemy(self.pos_chasseurs[0], self.pos_chasseurs[1], self.traj_chasseurs[0][1])

        chasseur_obs_1 = np.array([cap_cha1, vit_cha1, dist_ev_cha1, cap_cha1_ev, cap_ev, vit_ev, 
                                   dist_cha1_cha2, cap_cha1_cha2, cap_cha2, vit_cha2], dtype=np.float32)

        # ========== Chasseurs_2 ========== #
        cap_cha2_ev = utils.angle_entre_cap_and_enemy(self.pos_chasseurs[1], self.pos_eviteur, self.traj_chasseurs[1][1])
        cap_cha2_cha1 = utils.angle_entre_cap_and_enemy(self.pos_chasseurs[1], self.pos_chasseurs[0], self.traj_chasseurs[1][1])

        chasseur_obs_2 = np.array([cap_cha2, vit_cha2, dist_ev_cha2, cap_cha2_ev, cap_ev, vit_ev, 
                                   dist_cha1_cha2, cap_cha2_cha1, cap_cha1, vit_cha1], dtype=np.float32)


        return {"eviteur": eviteur_obs, "chasseur1": chasseur_obs_1, "chasseur2": chasseur_obs_2}


    def step(self, action_dict):
        self.current_step +=1
        # print(self.current_step)

        # Memoire
        dist_evit_obj_mem = np.linalg.norm(self.pos_eviteur - self.pos_objectif)
        # dist_cha1_evit_mem = np.linalg.norm(self.pos_chasseurs[0] - self.pos_eviteur)
        # dist_cha2_evit_mem = np.linalg.norm(self.pos_chasseurs[1] - self.pos_eviteur)

        # Déplacement des navire 
        action_dict["eviteur"] = np.clip(action_dict["eviteur"], -1.0, 1.0)
        action_dict["chasseur1"] = np.clip(action_dict["chasseur1"], -1.0, 1.0)
        action_dict["chasseur2"] = np.clip(action_dict["chasseur2"], -1.0, 1.0)

        nxe, nye, nce, nve = deplacement.move(self.pos_eviteur[0], self.pos_eviteur[1], self.traj_eviteur[1], self.traj_eviteur[0], action_dict["eviteur"][0], action_dict["eviteur"][1], 12)
        nxc1, nyc1, ncc1, nvc1 = deplacement.move(self.pos_chasseurs[0][0], self.pos_chasseurs[0][1], self.traj_chasseurs[0][1], self.traj_chasseurs[0][0], action_dict["chasseur1"][0], action_dict["chasseur1"][1], 8)
        nxc2, nyc2, ncc2, nvc2 = deplacement.move(self.pos_chasseurs[1][0], self.pos_chasseurs[1][1], self.traj_chasseurs[1][1], self.traj_chasseurs[1][0], action_dict["chasseur2"][0], action_dict["chasseur2"][1], 8)

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
            truncated = {"eviteur": True, "chasseur1": True, "chasseur2": True}
            terminated = {"eviteur": False, "chasseur1": False, "chasseur2": False}
            rewards = {"eviteur": -100, "chasseur1": 100, "chasseur2": 100}
        else :
            truncated = {"eviteur": False, "chasseur1": False, "chasseur2": False}
            # Calcul des rewards
            if np.linalg.norm(self.pos_eviteur - self.pos_objectif) < 30:
                terminated = {"eviteur": True, "chasseur1": True, "chasseur2": True}
                rewards = {"eviteur": 100, "chasseur1": -100 - self.dist_min, "chasseur2": -100 - self.dist_min}

            elif (np.linalg.norm(self.pos_chasseurs[0] - self.pos_eviteur) < 30 or np.linalg.norm(self.pos_chasseurs[1] - self.pos_eviteur) < 30):
                terminated = {"eviteur": True, "chasseur1": True, "chasseur2": True}
                rewards = {"eviteur": -100, "chasseur1": 100, "chasseur2": 100}

            else:
                proche = np.linalg.norm(self.pos_chasseurs[0] - self.pos_eviteur) < 80 or np.linalg.norm(self.pos_chasseurs[1] - self.pos_eviteur) < 80
                pen_pro = 1 if proche else 0
                terminated = {"eviteur": False, "chasseur1": False, "chasseur2": False}
                dist_evit_obj_new = np.linalg.norm(self.pos_eviteur - self.pos_objectif)
                rewards = {"eviteur": (dist_evit_obj_mem - dist_evit_obj_new)/10 - pen_pro - 0.2, "chasseur1": 0, "chasseur2": 0}
       
        obs = self._get_obs()
        infos = {}
        return obs, rewards, terminated, truncated, infos
    

    def render(self):
        pass

    def init_chasseur(self):
        while True:
            pos = np.random.uniform(0, 1000, 2)
            if pos[0] > 600 or pos[1] > 600:
                break
        return pos