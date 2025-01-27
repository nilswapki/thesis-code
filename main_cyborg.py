from CybORG_plus.mini_CAGE import (
    SimplifiedCAGE, Meander_minimal, B_line_minimal, Blue_sleep, React_restore_minimal)
from CybORG_plus.mini_CAGE.RLagents import BlueRLAgent
from CybORG_plus.Debugged_CybORG.CybORG.CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG_plus.mini_CAGE.SimplifiedCAGEWrapper import SimplifiedCAGEWrapper
from CybORG_plus.mini_CAGE.custom_monitor import Monitor
import datetime

import numpy as np


if __name__ == '__main__':

    seed = 42  # random.randint(1, 100)
    np.random.seed(seed)
    log_path = f'results/run-{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    # initialise environment
    num_envs = 1
    red_agent = B_line_minimal()
    env = SimplifiedCAGEWrapper(num_envs=num_envs, num_nodes=13, red_agent=red_agent, episode_length=100)
    env = Monitor(env=env, filename=log_path, info_keywords=("invalid_share",))
    state, _ = env.reset()

    # initialise the agents
    blue_agent = BlueRLAgent(env, verbose=0)

    reward_log = []
    blue_action_log = []
    total_reward = np.zeros(num_envs)

    blue_agent.train(timesteps=3000)

    """
    for i in range(100):

        blue_observation = state['Blue']
        red_observation = state['Red']

        actions = {
            "red": red_agent.get_action(observation=state["Red"]),
            "blue": blue_agent.get_action(observation=state["Blue"]),
        }

        state, reward, done, info = env.step(actions)
        total_reward += reward['Blue'].reshape(-1)
        reward_log.append(reward['Blue'].reshape(-1))
        blue_action_log.append(actions['blue'])


    print(np.stack(blue_action_log, axis=1))
    print(f'Total Reward: {total_reward}')
    
    """