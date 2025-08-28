import typing
from typing import Callable, Dict, List
import random as rand

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World, Line, Box
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

EVI_SPAWN = [[(-2,-1),(1,2)],    [(-1,0),(1,2)],    [(0,1),(1,2)],     [(1,2),(1,2)],
             [(1,2),(1,2)],      [(1,2),(0,1)],     [(1,2),(-1,0)],    [(1,2),(-2,-1)],
             [(1,2),(-2,-1)],    [(0,1),(-2,-1)],   [(-1,0),(-2,-1)],  [(-2,-1),(-2,-1)],
             [(-2,-1),(-2,-1)],  [(-2,-1),(-1,0)],  [(-2,-1),(0,1)],   [(-2,-1),(1,2)]]
OBS_SPAWN = [[(-2,2),(-1,1)], [(-1,1),(-2,2)]]


# ============================== Fonction obligatoire ============================== #

class Scenario(BaseScenario):

    def init_params(self, **kwargs):
        pass

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.init_params(**kwargs)
        self.batch_dim = batch_dim
        self.device = device
        self.visualize_semidims = False                             # Affichage de la limite du terrain
        self.plot_grid = False                                      # Affichage de la grille (utile pout les environement discret)
        self.n_agents = kwargs.pop("n_agents", 1)
        self.n_chasseurs = kwargs.pop("n_chasseurs", 2)
        self.n_obstacles = kwargs.pop("n_obstacles", 10)             
        self.collisions = kwargs.pop("collisions", True)            # Activation ou non des collision
        # self.init_background()                                      # Initialisation du fond
        
        self._render_field = True
        self.world_spawning_x = kwargs.pop("world_spawning_x", 2)   # X-coordinate limit for entities spawning
        self.world_spawning_y = kwargs.pop("world_spawning_y", 2)   # Y-coordinate limit for entities spawning
        self.enforce_bounds = kwargs.pop("enforce_bounds", False)   # Limitaion ou non de l'espace de déplacemnet

        # Hyperparametre de l'agent
        self.lidar_range = kwargs.pop("lidar_range", 0.35)
        self.agent_radius = kwargs.pop("agent_radius", 0.03)
        self.comms_range = kwargs.pop("comms_range", 0)
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 12)
        
        # self.shared_rew_cha = kwargs.pop("shared_rew_cha", True)        # les chasseurs partagent
        # self.shared_rew_evi = kwargs.pop("shared_rew_evi", True)       # les évitants ne partagent pas
        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)

        self.time_malus = kwargs.pop("time_malus", 0.01)
        self.final_reward = kwargs.pop("final_reward", 5)
        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -0.2)
        self.eviteur_threat_penalty = kwargs.pop("eviteur_threat_penalty", 0.5)
        
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.min_distance_between_entities = 1
        self.min_collision_distance = 0.005

        
        # Dimension de l'environement
        if self.enforce_bounds:
            self.x_semidim = self.world_spawning_x
            self.y_semidim = self.world_spawning_y
        else:
            self.x_semidim = None
            self.y_semidim = None

        # Make world
        world = World(batch_dim, device, substeps=2, x_semidim=self.x_semidim, y_semidim=self.y_semidim,)

        # Generation de l'éviteur (pour l'instant le nombre d'éviteur est fixé a 1)
        self.goals = []

        for i in range(self.n_agents):
            eviteur = Agent(
                name=f"eviteur",
                collide=self.collisions,
                color=(0.2, 0.2, 1),
                max_speed=0.25,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                sensors=(
                    [
                        Lidar(
                            world,
                            n_rays=self.n_lidar_rays,
                            max_range=self.lidar_range,
                            entity_filter=lambda e:e.collide,
                        ),
                    ]
                    if self.collisions
                    else None
                ),
            )

            eviteur.type = "eviteur"
            eviteur.pos_rew = torch.zeros(batch_dim, device=device)
            eviteur.agent_collision_rew = eviteur.pos_rew.clone()
            world.add_agent(eviteur)

            # Ajout de l'objectif de l'éviteur
            goal = Landmark(
                name=f"goal_eviteur",
                collide=False,
                color=(0.2, 0.2, 1),
            )
            world.add_landmark(goal)
            eviteur.goal = goal
            self.goals.append(goal)

        # --- Ajout d'un chasseur ---
        for i in range(self.n_chasseurs):
            chasseur = Agent(
                name=f"chasseur_{i}",
                collide=self.collisions,
                color=(1, 0, 0),  # rouge
                max_speed=0.20,   # un peu plus lent ou plus rapide selon la difficulté
                shape=Sphere(radius=self.agent_radius),
            )
  
            chasseur.type = "chasseur"
            world.add_agent(chasseur)

        # Add obstacles
        self.obstacles = ([])
        for i in range(self.n_obstacles):
            obstacle = Landmark(name=f"obstacle_{i}",
                                collide=True,
                                color=Color.BLACK,
                                shape=Sphere(radius=0.1))
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = torch.zeros(batch_dim, device=device)

        return world


    def reset_world_at(self, env_index: int = None):
        world = self.world
        
        index_spawn_agent = rand.randint(0,15)
        bound_agent = EVI_SPAWN[index_spawn_agent]
        ajout = rand.randint(0,3)
        index_spawn_obj = (((index_spawn_agent // 4) + 2) % 4) * 4 + ajout
        bound_obj = EVI_SPAWN[index_spawn_obj]
        index_obs = 0 if ((index_spawn_agent // 4) == 0 or (index_spawn_agent // 4) == 2) else 1
        bound_obs = OBS_SPAWN[index_obs] 

        # === 1. Réinitialiser l'éviteur ===
        eviteurs = [a for a in self.world.agents if a.type == "eviteur"]
        ScenarioUtils.spawn_entities_randomly(
            eviteurs,
            world=world,
            env_index=env_index,
            min_dist_between_entities=0.0,
            x_bounds=bound_agent[0],
            y_bounds=bound_agent[1],
        )

        ScenarioUtils.spawn_entities_randomly(
            self.goals,
            self.world,
            env_index,
            min_dist_between_entities=0.0,
            x_bounds=bound_obj[0],
            y_bounds=bound_obj[1],
        )

        ScenarioUtils.spawn_entities_randomly(
            self.obstacles,
            self.world,
            env_index,
            min_dist_between_entities=0.0,
            x_bounds=bound_obs[0],
            y_bounds=bound_obs[1],
        )

        chasseur = [a for a in self.world.agents if a.type == "chasseur"]
        ScenarioUtils.spawn_entities_randomly(
            chasseur,
            self.world,
            env_index,
            min_dist_between_entities=0.0,
            x_bounds=(bound_obj[0][0] - 1, bound_obj[0][1] + 1),
            y_bounds=(bound_obj[1][0] - 1, bound_obj[1][1] + 1),
        )

        for i, agent in enumerate(self.world.agents):
            agent.goal = self.goals[0]  # Solution temporaire
            if env_index is None:
                agent.pos_shaping = (torch.linalg.vector_norm(agent.state.pos - agent.goal.state.pos, dim=1,) * self.pos_shaping_factor)
            else:
                agent.pos_shaping[env_index] = (torch.linalg.vector_norm(agent.state.pos[env_index] - agent.goal.state.pos[env_index]) * self.pos_shaping_factor)



    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            eviteur = next(a for a in self.world.agents if a.type == "eviteur")

            # --- Récompenses positionnelles et final reward ---
            for a in self.world.agents:
                if a.type == "eviteur" :
                    self.pos_rew += self.eviteur_reward(a)

            # --- Récompense finale si l'éviteur atteint son objectif ---
            mask = torch.norm(eviteur.state.pos - eviteur.goal.state.pos) < 0.05
            self.final_rew[mask] = self.final_reward

        # --- Collision avec les obstacles ---
        collision_obs_rew = torch.zeros(self.batch_dim, device=self.device)
        for obs in self.obstacles:
            if self.world.collides(obs, agent):
                collision_obs_rew += self.agent_collision_penalty

        # --- Récompense finale à renvoyer ---
        reward = self.pos_rew + self.final_rew + collision_obs_rew
        return reward



    def eviteur_reward(self, agent: Agent):
        goal_dist = torch.norm(agent.state.pos - agent.goal.state.pos, dim=-1)
        reward = self.pos_shaping_factor * (agent.pos_shaping - goal_dist)
        agent.pos_shaping = goal_dist

        # Malus si proche d'un chasseur
        for other in self.world.agents:
            if other.type == "chasseur":
                dist = torch.norm(agent.state.pos - other.state.pos, dim=-1)
                if dist < 0.07:
                    reward -= self.eviteur_threat_penalty  
                    print("bonjour")

        return reward


    def observation(self, agent: Agent):
        vecteur_obs = []
        
        if agent.type == "eviteur":
            vecteur_obs.append(agent.state.pos - agent.goal.state.pos) # position de l'objectif
            chasseurs = [a for a in self.world.agents if a.type == "chasseur"]
            for chasseur in chasseurs:
                vecteur_obs.append(agent.state.pos - chasseur.state.pos) # position des chasseur

        lidar_obs = []
        if self.collisions and agent.sensors:
            lidar = agent.sensors[0]
            lidar_obs = [lidar._max_range - lidar.measure()]

        obs = torch.cat([agent.state.pos, agent.state.vel] + vecteur_obs + lidar_obs, dim=-1,)

        return obs
    

    def done(self):
        eviteur = next(a for a in self.world.agents if a.type == "eviteur")

        reached_goal = torch.linalg.vector_norm(
            eviteur.state.pos - eviteur.goal.state.pos, dim=-1
        ) < eviteur.shape.radius  # Tensor [batch_size] bool

        # Capture condition : distance avec un chasseur < seuil
        captured = torch.zeros_like(reached_goal, dtype=torch.bool)
        for chasseur in [a for a in self.world.agents if a.type == "chasseur"]:
            captured |= torch.linalg.vector_norm(
                eviteur.state.pos - chasseur.state.pos, dim=-1
            ) < (eviteur.shape.radius + chasseur.shape.radius)


        return reached_goal | captured  # => Tensor [batch_size] bool



    def info(self, agent: Agent) -> Dict[str, torch.Tensor]: 
        return {
            "pos_rew_eviteur": self.pos_rew,
            "final_rew_eviteur": self.final_rew,
        }
    

    def pre_step(self):
        eviteur = next(a for a in self.world.agents if a.type == "eviteur")
        for chasseur in [a for a in self.world.agents if a.type == "chasseur"]:
            direction = eviteur.state.pos - chasseur.state.pos
            norm = torch.norm(direction, dim=-1, keepdim=True) + 1e-8
            direction = direction / norm
            chasseur.state.vel[:] = direction * chasseur.max_speed



if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=False,
        save_render=False,
        display_info=True,
        
    )
