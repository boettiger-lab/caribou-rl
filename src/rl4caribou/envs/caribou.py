import numpy as np


# pop = elk, caribou, wolves
# Caribou Scenario
def dynamics(pop, effort, harvest_fn, p, timestep=1):
    pop = harvest_fn(pop, effort)
    X, Y, Z = pop[0], pop[1], pop[2]

    K = p["K"]  # - 0.2 * np.sin(2 * np.pi * timestep / 3200)
    D = p["D"] + 0.5 * np.sin(2 * np.pi * timestep / 3200)
    beta = p["beta"] + 0.2 * np.sin(2 * np.pi * timestep / 3200)

    X += (
        p["r_x"] * X * (1 - (X + p["tau_xy"] * Y) / K)
        - (1 - D) * beta * Z * (X**2) / (p["v0"] ** 2 + X**2)
        + p["sigma_x"] * X * np.random.normal()
    )

    Y += (
        p["r_y"] * Y * (1 - (Y + p["tau_yx"] * X) / K)
        - D * beta * Z * (Y**2) / (p["v0"] ** 2 + Y**2)
        + p["sigma_y"] * Y * np.random.normal()
    )

    Z += (
        p["alpha"]
        * beta
        * Z
        * (
            (1 - D) * (X**2) / (p["v0"] ** 2 + X**2)
            + D * (Y**2) / (p["v0"] ** 2 + Y**2)
        )
        - p["dH"] * Z
        + p["sigma_z"] * Z * np.random.normal()
    )

    pop = np.array([X, Y, Z], dtype=np.float32)
    pop = np.clip(pop, [0, 0, 0], [np.Inf, np.Inf, np.Inf])
    return pop


initial_pop = [0.5, 0.5, 0.2]

parameters = {
    "r_x": np.float32(0.13),
    "r_y": np.float32(0.2),
    "K": np.float32(1),
    "beta": np.float32(0.1),
    "v0": np.float32(0.1),
    "D": np.float32(0.8),
    "tau_yx": np.float32(0.7),
    "tau_xy": np.float32(0.2),
    "alpha": np.float32(0.4),
    "dH": np.float32(0.03),
    "sigma_x": np.float32(0.05),
    "sigma_y": np.float32(0.05),
    "sigma_z": np.float32(0.05),
}


def harvest(pop, effort):
    q0 = 0.5  # catchability coefficients -- erradication is impossible
    q2 = 0.5
    pop[0] = pop[0] * (1 - effort[0] * q0)  # pop 0, elk
    pop[2] = pop[2] * (1 - effort[1] * q2)  # pop 2, wolves
    return pop


def utility(pop, effort):
    benefits = 0.5 * pop[1]  # benefit from Caribou
    costs = 0.00001 * (effort[0] + effort[1])  # cost to culling
    if np.any(pop <= 0.01):
        benefits -= 1
    return benefits - costs


import gymnasium as gym


class Caribou(gym.Env):
    """A 3-species ecosystem model with two control actions"""

    def __init__(self, config=None):
        config = config or {}

        ## these parameters may be specified in config
        self.Tmax = config.get("Tmax", 800)
        self.max_episode_steps = self.Tmax
        self.threshold = config.get("threshold", np.float32(1e-4))
        self.init_sigma = config.get("init_sigma", np.float32(1e-3))
        self.training = config.get("training", True)
        self.initial_pop = config.get("initial_pop", initial_pop)
        self.parameters = config.get("parameters", parameters)
        self.dynamics = config.get("dynamics", dynamics)
        self.harvest = config.get("harvest", harvest)
        self.utility = config.get("utility", utility)
        self.observe = config.get(
            "observe", lambda state: state
        )  # default to perfectly observed case
        self.bound = 2 * self.parameters["K"]

        self.action_space = gym.spaces.Box(
            np.array([-1, -1], dtype=np.float32),
            np.array([1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            np.array([-1, -1, -1], dtype=np.float32),
            np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.reset(seed=config.get("seed", None))

    def reset(self, *, seed=None, options=None):
        self.timestep = 0
        self.initial_pop += np.multiply(
            self.initial_pop, np.float32(self.init_sigma * np.random.normal(size=3))
        )
        self.state = self.state_units(self.initial_pop)
        info = {}
        return self.observe(self.state), info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        pop = self.population_units()  # current state in natural units
        effort = (action + 1.0) / 2

        # harvest and recruitment
        reward = self.utility(pop, effort)
        nextpop = self.dynamics(
            pop, effort, self.harvest, self.parameters, self.timestep
        )

        self.timestep += 1
        terminated = bool(self.timestep > self.Tmax)

        # in training mode only: punish for population collapse
        if any(pop <= self.threshold) and self.training:
            terminated = True
            reward -= 50 / self.timestep

        self.state = self.state_units(nextpop)  # transform into [-1, 1] space
        observation = self.observe(self.state)  # same as self.state
        return observation, reward, terminated, False, {}

    def state_units(self, pop):
        self.state = 2 * pop / self.bound - 1
        self.state = np.clip(
            self.state,
            np.repeat(-1, self.state.__len__()),
            np.repeat(1, self.state.__len__()),
        )
        return np.float32(self.state)

    def population_units(self):
        pop = (self.state + 1) * self.bound / 2
        return np.clip(
            pop, np.repeat(0, pop.__len__()), np.repeat(np.Inf, pop.__len__())
        )


# verify that the environment is defined correctly
# from stable_baselines3.common.env_checker import check_env
# env = s3a2()
# check_env(env, warn=True)
