import os 
import sys

sys.path.append(os.path.abspath(r"C:\Users\FX643778\Documents\Git\Interception_marl\1e_2c\environement"))

import wandb
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from affrontement_env import env_creator
from ray.tune.callback import Callback

class CustomWandbCallback(Callback):
    def __init__(self, project, entity=None):
        self.project = project
        self.entity = entity
        self.run = None

    def on_experiment_start(self, **info):
        self.run = wandb.init(project=self.project, entity=self.entity, config=info.get("config", {}))

    def on_trial_result(self, iteration, trials, trial, result, **info):
        # print(result.keys()) #Permet de commaitre tous les champs de metric dans result
        wandb.log(result)

    def on_experiment_end(self, **info):
        if self.run:
            self.run.finish()

# Enregistrement
register_env("affrontement_v0", env_creator)

# Configuration RLlib
config = (
    PPOConfig()
    .environment("affrontement_v0", env_config={"render_mode": None})
    .framework("torch")
    .env_runners(num_env_runners=1)        # runners = environement en parallele pour accelerer l'apprentissage
    .training(
        train_batch_size=512,
        lr=2e-5,
        gamma=0.99,
        lambda_=0.9,
        use_gae=True,
        clip_param=0.4,
        grad_clip=None,
        entropy_coeff=0.1,
        vf_loss_coeff=0.25,
    )
    .multi_agent(
        policies={"chasseur_policy", "eviteur_policy"},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: (
            "chasseur_policy" if "chasseur" in agent_id else "eviteur_policy")  # Deux politique une pour les chasseur et une pour les eviteur
    )
)

# Entraînement
tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        callbacks=[CustomWandbCallback(project="affrontement", entity="ton_username")],
        stop={"training_iteration": 50}
    )
).fit()