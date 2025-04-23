from ml_collections import ConfigDict
from typing import Tuple
import gym
from gym.envs.registration import register
from configs.envs.terminal_fns import finite_horizon_terminal

env_name_fn = lambda l: f"minimal-test-{l}-v0"

def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    length = config.env_name
    env_name = env_name_fn(length)
    register(
        env_name,
        entry_point="envs.minimal-test.minimal-test:MinimalTestEnv",  # Adjust this path if needed
        max_episode_steps=length,
    )

    del config.create_fn
    return config, env_name

def get_config():
    config = ConfigDict()
    config.create_fn = create_fn

    config.env_type = "minimal-test"
    config.terminal_fn = finite_horizon_terminal

    config.eval_interval = 100
    config.save_interval = 500
    config.eval_episodes = 1

    config.env_name = 100  # episode length

    return config
