import gymnasium as gym
from gym import spaces
import numpy as np
from .minimal import SimplifiedCAGE
from .baseline_agents import Meander_minimal, B_line_minimal, Blue_sleep

import sys, os
from main_eval_red import initialize_learner_with_flags
from torchkit import pytorch_utils as ptu
import torch

class SimplifiedCAGEWrapper(gym.Env):
    def __init__(self, num_envs=1, remove_bugs=True, red_agents=None, episode_length=100, verbose=False):
        super().__init__()
        self.name = 'mini-cage'
        self.env = SimplifiedCAGE(num_envs, remove_bugs)
        self.num_envs = num_envs
        self.verbose = verbose

        self.episode_length = episode_length
        self.eval_length = episode_length
        self.steps_taken = 0

        if red_agents is not None:
            self.red_agents = red_agents
        else:
            self.red_agents = [B_line_minimal()]
        self.current_red_agent_index = 0

        # Define the action space for BLUE agent
        self.action_space = spaces.Discrete((4 * self.env.num_nodes) + 1)

        # Define the observation space based on the environment's state
        # 2n scan activity, 2n host safety, n prior scans, n decoy info --> 6n
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.env.num_nodes*6,), dtype=np.float32)

        self.adversarial = True
        if self.adversarial:
            self.learner_red = initialize_learner_with_flags(save_dir='logs_results/mini-cage-red/attacker_30000/seed-5')
            self.action_red, self.reward_red, self.internal_state_red = self.learner_red.agent.get_initial_info(
                self.learner_red.config_seq.sampled_seq_len
            )

    def reset(self, seed=None, options=None):

        # Rotate to the next red agent
        self.current_red_agent_index = (self.current_red_agent_index + 1) % len(self.red_agents)
        self.red_agent = self.red_agents[self.current_red_agent_index]

        # Reset the environment
        self.steps_taken = 0
        state, info = self.env.reset()
        return np.array(state['Blue']).reshape(-1, 1), info
        #return {'state': state, 'info': info}, {}

    def step(self, blue_action):
        self.steps_taken += 1
        if self.adversarial:
            obs_red = self.env._process_state(self.env.state, self.env.current_decoys)['Red']
            obs_red = ptu.from_numpy(obs_red).reshape(-1, 1) if obs_red.shape[0] == 1 else ptu.from_numpy(obs_red)
            self.action_red, self.internal_state_red = self.learner_red.agent.act(
                prev_internal_state=self.internal_state_red,
                prev_action=self.action_red,
                reward=self.reward_red,
                obs=obs_red,
                deterministic=True,
            )
        else:
            self.action_red = self.red_agent.get_action(observation=self.env._process_state(self.env.state, self.env.current_decoys)['Red'])

        # converting int64 to np.array
        blue_action = np.array([[blue_action]]) if np.issubdtype(type(blue_action), np.integer) else blue_action

        if self.verbose:
            self.env.describe_action_blue(blue_action)
            self.env.describe_action_red(self.action_red)

        # Take a step in the environment
        if self.adversarial:
            action_red_shaped = torch.argmax(self.action_red, dim=1).cpu().numpy().reshape(1, 1)
        else: action_red_shaped = self.action_red
        next_state, reward, done, info = self.env.step(action_red_shaped, blue_action)
        if self.adversarial:
            self.reward_red = torch.tensor(reward['Red']).reshape(1, 1)
        ##reward = np.array([reward['Red'], reward['Blue']]).T  # Adjust reward structure for both agents

        # Convert the "done" signal into a Gym-compatible format
        terminated = np.all(done)
        truncated = False  # Assuming no truncation logic in the base environment

        if self.steps_taken == self.episode_length:
            terminated = True

        return next_state['Blue'], reward['Blue'], terminated, truncated, info

    def describe_feature(self, feature_index):
        return self.env.describe_feature(feature_index)

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
