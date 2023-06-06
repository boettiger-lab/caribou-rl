from envs import two_three_fishing
from envs import growth_functions
from parameters import parameters
import callback_fn
from ray.rllib.algorithms import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env
import os
import pandas as pd
import numpy as np
import torch

## GLOBALS:
P = parameters()
_DEFAULT_PARAMS = P.parameters()

iterations = 250
## SETTING UP RL ALGO

register_env("two_three_fishing", two_three_fishing.twoThreeFishing)

config = ppo.PPOConfig()
config.training(vf_clip_param = 50.0)
config.num_envs_per_worker=20
config = config.resources(num_gpus=torch.cuda.device_count())
config.framework_str="torch"
config.create_env_on_local_worker = True
config.env="two_three_fishing"
#
config.env_config["parameters"] = _DEFAULT_PARAMS
config.env_config["growth_fn"] = growth_functions.K_limit_rx_drift_growth
config.env_config["fluctuating"] = True
config.env_config["initial_pop"] = parameters().init_state()
agent = config.build()

#agent = PPOTrainer(config=config)
#


iterations = 200
checkpoint = ("cache_two_three/checkpoint_000{}".format(iterations))

if not os.path.exists(checkpoint): # train only if no trained agent saved
  for _ in range(iterations):
    print(f"iteration {_}", end = "\r")
    agent.train()
  checkpoint = agent.save("cache")

agent.restore(checkpoint)

stats = agent.evaluate() # built-in method to evaluate agent on eval env
print(stats["evaluation"])






