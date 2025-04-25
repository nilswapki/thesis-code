import gymnasium as gym
from gym import spaces
import numpy as np


class MinimalTestEnv(gym.Env):
    def __init__(self, num_envs=1):
        super().__init__()
        self.name = 'minimal-test'
        self.num_envs = num_envs
        self.num_features = 5
        self.steps_taken = 0
        self.episode_length = 10

        self.action_space = spaces.Discrete(2)  # 0 or 1
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_features,), dtype=np.float32)

        self.state = None

    def reset(self, *, seed=None, options=None):
        print("New Episode")
        super().reset(seed=seed)
        self.steps_taken = 0
        self.state = np.random.uniform(-1.0, 1.0, size=(self.num_features,))
        return self.state.reshape(-1, 1), {}

    def step(self, action):
        self.steps_taken += 1

        # Reward depends only on feature[2] and action
        if action == 1:
            reward = 10 * self.state[2]  # only feature[2] matters
        else:
            reward = -10 * self.state[2]

        # New random state
        self.state = np.random.uniform(-1.0, 1.0, size=(self.num_features,))
        terminated = self.steps_taken >= self.episode_length
        truncated = False
        return self.state.reshape(1, -1), reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
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
