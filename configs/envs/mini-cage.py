from ml_collections import ConfigDict
from typing import Tuple
import gym
from gym.envs.registration import register
from configs.envs.terminal_fns import finite_horizon_terminal

env_name_fn = lambda l: f"mini-cage-{l}-v0"

def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    length = config.env_name
    env_name = env_name_fn(length)
    register(
        env_name,
        entry_point="envs.mini_CAGE.SimplifiedCAGEWrapper:SimplifiedCAGEWrapper",
        max_episode_steps=length,  # NOTE: has to define it here
    )

    del config.create_fn
    return config, env_name

def get_config():
    config = ConfigDict()
    config.create_fn = create_fn

    config.env_type = "mini-cage"
    config.terminal_fn = finite_horizon_terminal

    config.eval_interval = 100
    config.save_interval = 1000
    config.eval_episodes = 6

    # [1, 2, 5, 10, 30, 50, 100, 300, 500, 1000]
    config.env_name = 100

    return config