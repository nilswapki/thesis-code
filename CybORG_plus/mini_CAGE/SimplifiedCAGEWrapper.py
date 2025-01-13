import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .minimal import SimplifiedCAGE


class SimplifiedCAGEWrapper(gym.Env):
    def __init__(self, num_envs=1, num_nodes=13, remove_bugs=False):
        super().__init__()
        self.env = SimplifiedCAGE(num_envs, num_nodes, remove_bugs)
        self.num_envs = num_envs
        self.num_nodes = num_nodes

        # Define the action space for both agents (red and blue)
        self.action_space = spaces.MultiDiscrete([(4 * num_nodes) + 1] * num_envs)

        # Define the observation space based on the environment's state
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(num_nodes*6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Reset the environment
        state, info = self.env.reset()
        return state, info
        #return {'state': state, 'info': info}, {}

    def step(self, actions):
        # Parse red and blue actions
        red_action = actions['red']
        blue_action = actions['blue']

        # Take a step in the environment
        next_state, reward, done, info = self.env.step(red_action, blue_action)
        ##reward = np.array([reward['Red'], reward['Blue']]).T  # Adjust reward structure for both agents

        # Convert the "done" signal into a Gym-compatible format
        terminated = np.all(done)
        truncated = False  # Assuming no truncation logic in the base environment

        return next_state, reward, terminated, info

    def render(self, mode='human'):
        # Rendering is optional and depends on the base environment's capabilities
        print("Rendering not implemented for SimplifiedCAGE.")

    def close(self):
        # Perform cleanup if needed
        pass
