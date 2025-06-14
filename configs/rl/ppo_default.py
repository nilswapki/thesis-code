from configs.rl.name_fns import name_fn
from ml_collections import ConfigDict
from typing import Tuple

# Define a custom name function for PPO
def ppo_name_fn(config: ConfigDict, *args) -> Tuple[ConfigDict, str]:
    config, name = name_fn(config)
    return config, name + f"clip-{config.clip_param}/"

# Function to get the default configuration for PPO
def get_config():
    config = ConfigDict()
    config.name_fn = ppo_name_fn

    config.algo = "ppo"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4

    config.config_actor = ConfigDict()
    config.config_actor.hidden_dims = (256, 256)

    config.config_critic = ConfigDict()
    config.config_critic.hidden_dims = (256, 256)

    config.discount = 0.99
    config.tau = 0.005

    config.clip_param = 0.2
    config.ppo_epochs = 10
    config.mini_batch_size = 64
    config.entropy_coef = 0.01

    config.replay_buffer_size = 1e6
    config.replay_buffer_num_episodes = 1e3

    return config