from ml_collections import ConfigDict
from typing import Tuple
import gym
from gym.envs.registration import register
from configs.envs.terminal_fns import finite_horizon_terminal

env_name_fn = lambda l: f"network-defender-{l}-v0"

def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    length = config.env_name
    n_nodes = config.n_nodes
    extra_edge_prob = config.extra_edge_prob
    num_critical_nodes = config.num_critical_nodes

    env_name = env_name_fn(length)
    register(
        env_name,
        entry_point="envs.network-defender.network-defender:NetworkDefenderEnv",
        kwargs=dict(
            n_nodes=n_nodes,
            extra_edge_prob=extra_edge_prob,
            num_critical_nodes=num_critical_nodes,
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

    config.eval_interval = 10
    config.save_interval = 1000
    config.eval_episodes = 2

    config.n_nodes = 300
    config.extra_edge_prob = 0.01
    config.num_critical_nodes = 5

    # [1, 2, 5, 10, 30, 50, 100, 300, 500, 1000]
    config.env_name = 100

    return config