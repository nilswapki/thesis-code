import gymnasium as gym
from gym import spaces
import numpy as np
from .minimal import SimplifiedCAGE
from .baseline_agents import Meander_minimal


class SimplifiedCAGEWrapper(gym.Env):
    def __init__(self, num_envs=1, remove_bugs=True, red_agent=None, episode_length=100, verbose=False):
        super().__init__()
        self.name = 'mini-cage'
        self.env = SimplifiedCAGE(num_envs, remove_bugs)
        self.num_envs = num_envs
        self.verbose = verbose

        self.episode_length = episode_length
        self.eval_length = episode_length
        self.steps_taken = 0

        if red_agent is not None:
            self.red_agent = red_agent
        else:
            self.red_agent = Meander_minimal()

        # Define the action space for BLUE agent
        self.action_space = spaces.Discrete((4 * self.env.num_nodes) + 1)

        # Define the observation space based on the environment's state
        # 2n scan activity, 2n host safety, n prior scans, n decoy info --> 6n
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.env.num_nodes*6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Reset the environment
        self.steps_taken = 0
        state, info = self.env.reset()
        return np.array(state['Blue']).reshape(-1, 1), info
        #return {'state': state, 'info': info}, {}

    def step(self, blue_action):
        self.steps_taken += 1
        # Parse red and blue actions
        red_action = self.red_agent.get_action(observation=self.env._process_state(self.env.state, self.env.current_decoys)['Red'])

        # converting int64 to np.array
        blue_action = np.array([[blue_action]]) if np.issubdtype(type(blue_action), np.integer) else blue_action

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

        return next_state['Blue'], reward['Blue'], terminated, truncated, info

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
