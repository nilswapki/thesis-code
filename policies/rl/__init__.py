from .sacd import SACD
from .dqn import DQN
from .ppo import PPO
from .ppo2 import PPO2

RL_ALGORITHMS = {
    SACD.name: SACD,
    DQN.name: DQN,
    PPO.name: PPO,
    PPO2.name: PPO2,
}
