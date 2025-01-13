from CybORG_plus.mini_CAGE import (
    SimplifiedCAGE, Meander_minimal, React_restore_minimal)
from CybORG_plus.mini_CAGE.RLagents import BlueRLAgent
from CybORG_plus.Debugged_CybORG.CybORG.CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG_plus.mini_CAGE.SimplifiedCAGEWrapper import SimplifiedCAGEWrapper

import numpy as np


if __name__ == '__main__':

    seed = 42  # random.randint(1, 100)
    np.random.seed(seed)

    # initialise environment
    num_envs = 1
    #env = SimplifiedCAGE(num_envs=1)
    env = SimplifiedCAGEWrapper(num_envs=num_envs, num_nodes=13)
    state, _ = env.reset()

    # initialise the agents
    red_agent = Meander_minimal()
    blue_agent = BlueRLAgent(env)  # TODO: replace this with RL agent

    reward_log = []
    blue_action_log = []
    total_reward = np.zeros(num_envs)

    for i in range(100):
        print('----------------')

        blue_observation = state['Blue']
        red_observation = state['Red']

        actions = {
            "red": red_agent.get_action(observation=state["Red"]),
            "blue": blue_agent.get_action(observation=state["Blue"]),
        }

        state, reward, done, info = env.step(actions)
        total_reward += reward['Blue'].reshape(-1)
        reward_log.append(reward['Blue'].reshape(-1))
        print('Reward ', reward['Blue'].reshape(-1))
        blue_action_log.append(actions['blue'])


    print(np.stack(blue_action_log, axis=1))
    print(f'Total Reward: {total_reward}' )