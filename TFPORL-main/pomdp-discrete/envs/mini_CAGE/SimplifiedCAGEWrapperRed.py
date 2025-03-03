import gymnasium as gym
from gym import spaces
import numpy as np
from .minimal import SimplifiedCAGE
from .baseline_agents import React_restore_minimal, AdaptiveDefender


class SimplifiedCAGEWrapperRed(gym.Env):
    def __init__(self, num_envs=1, num_nodes=13, remove_bugs=True, blue_agents=None, episode_length=100, verbose=False):
        super().__init__()
        self.env = SimplifiedCAGE(num_envs, remove_bugs)
        self.num_envs = num_envs
        self.num_nodes = num_nodes
        self.verbose = verbose

        self.episode_length = episode_length
        self.eval_length = episode_length
        self.steps_taken = 0

        if blue_agents is not None:
            self.blue_agents = blue_agents
        else:
            self.blue_agents = [React_restore_minimal()]
        self.current_blue_agent_index = 0

        # Define the action space for RED agent
        self.action_space = spaces.Discrete(self.env.num_subnets+self.env.num_nodes*len(self.env.red_actions[2:])+1)

        # Define the observation space based on the environment's state
        self.observation_space = spaces.Box(-1.0, 1.0, shape=((self.env.num_nodes*3)+1,), dtype=np.float32)

    def reset(self, seed=None, options=None):

        # Rotate to the next blue agent
        self.current_blue_agent_index = (self.current_blue_agent_index + 1) % len(self.blue_agents)
        self.blue_agent = self.blue_agents[self.current_blue_agent_index]

        # Reset stats
        self.steps_taken = 0
        state, info = self.env.reset()
        return np.array(state['Red']).reshape(-1, 1), info
        #return {'state': state, 'info': info}, {}

    def step(self, red_action):
        self.steps_taken += 1
        # Parse red and blue actions
        blue_action = self.blue_agent.get_action(observation=self.env._process_state(self.env.state, self.env.current_decoys)['Blue'])

        # converting int64 to np.array
        red_action = np.array([[red_action]]) if np.issubdtype(type(red_action), np.integer) else red_action

        if self.verbose:
            self.env.describe_action_blue(blue_action)
            self.env.describe_action_red(red_action)

        # Take a step in the environment
        next_state, reward, done, info = self.env.step(red_action, blue_action)
        ##reward = np.array([reward['Red'], reward['Blue']]).T  # Adjust reward structure for both agents

        # Convert the "done" signal into a Gym-compatible format
        terminated = np.all(done)
        truncated = False  # Assuming no truncation logic in the base environment

        if self.steps_taken == self.episode_length:
            terminated = True

        return next_state['Red'], reward['Red'], terminated, truncated, info

    def render(self, mode='human'):
        # Rendering is optional and depends on the base environment's capabilities
        print("Rendering not implemented for SimplifiedCAGE.")

    def close(self):
        # Perform cleanup if needed
        pass

    def seed(self, seed=None):
        # Set the seed using the reset method
        self.reset(seed=seed)
        if hasattr(self.action_space, 'seed'):
            self.action_space.seed(seed)
        if hasattr(self.observation_space, 'seed'):
            self.observation_space.seed(seed)

    def eval(self):
        # TODO: Add evaluation-specific logic here if needed
        pass
