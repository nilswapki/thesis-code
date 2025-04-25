from ml_collections import ConfigDict
from typing import Tuple
import gym
from gym.envs.registration import register
from configs.envs.terminal_fns import finite_horizon_terminal

env_name_fn = lambda l: f"network-defender-{l}-v0"

def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    length = config.env_name

    env_name = env_name_fn(length)
    register(
        env_name,
        entry_point="envs.network-defender.network-defender2:NetworkDefenderEnv",
        kwargs=dict(
            n_nodes=config.n_nodes,
            extra_edge_prob=config.extra_edge_prob,
            noise_mean=config.noise_mean,
            noise_95_interval=config.noise_95_interval,
            recursive=config.recursive,
        ),
        max_episode_steps=length,  # NOTE: has to define it here
    )

    del config.create_fn
    return config, env_name

def get_config():
    config = ConfigDict()
    config.create_fn = create_fn

    config.env_type = "network-defender"
    config.terminal_fn = finite_horizon_terminal

    config.eval_interval = 100
    config.save_interval = 5000
    config.eval_episodes = 5

    config.n_nodes = 15
    config.extra_edge_prob = 0.2
    config.noise_mean = 0.4  # 0.4
    config.noise_95_interval = 0.25  # 0.25
    config.recursive = False

    # [1, 2, 5, 10, 30, 50, 100, 300, 500, 1000]
    config.env_name = 100

    return config