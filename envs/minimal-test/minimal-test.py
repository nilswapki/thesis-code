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
        self.episode_length = 100

        self.action_space = spaces.Discrete(2)  # 0 or 1
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_features,), dtype=np.float32)

        self.state = None
        # Assign each feature a different oscillation speed
        self.freqs = np.linspace(0.05, 0.25, self.num_features)  # Different speeds for each feature
        self.time = 0  # Global time counter

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0
        self.time = 0
        self.state = np.sin(self.freqs * self.time * 2 * np.pi)
        return self.state.reshape(-1, 1), {}

    def step(self, action):
        self.steps_taken += 1
        self.time += 1

        # Update state: each feature is a sine wave oscillating differently
        self.state = np.sin(self.freqs * self.time * 2 * np.pi)

        # Reward depends strongly on feature[2] and feature[4]
        feature2 = self.state[2]
        feature4 = self.state[4]

        if action == 1:
            reward = 5 * feature2 + 3 * feature4
        else:
            reward = -5 * feature2 + 2 * feature4

        terminated = self.steps_taken >= self.episode_length
        truncated = False
        return self.state.reshape(1, -1), reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.reset(seed=seed)
        if hasattr(self.action_space, 'seed'):
            self.action_space.seed(seed)
        if hasattr(self.observation_space, 'seed'):
            self.observation_space.seed(seed)

    def eval(self):
        pass
