from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from .test_agent import Base_agent
import torch
import numpy as np

class BlueRLAgent():
    def __init__(self, env, model_path=None, verbose=1):
        """
        Initialize the RL agent.
        :param env: The environment to train the agent.
        :param model_path: Path to a pre-trained model (optional).
        """
        self.env = env #DummyVecEnv([lambda: env])  # Wrap for RL compatibility
        if model_path:
            self.model = PPO.load(model_path)
        else:
            self.model = PPO("MlpPolicy", self.env, verbose=verbose)

    def train(self, timesteps=100):
        """Train the RL agent."""
        self.model.learn(total_timesteps=timesteps, callback=ProgressBarCallback())


    def get_action(self, observation, action_mask=None):
        """
        Return a valid action from the RL policy, respecting the action mask.
        :param observation: Current observation from the environment.
        :param action_mask: Binary mask indicating valid actions (1=valid, 0=invalid).
        :return: A valid action selected by the RL policy.
        """
        """
        # Convert observation to a PyTorch tensor
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)

        # Predict raw actions (e.g., logits or probabilities)
        action_distribution = self.model.policy.get_distribution(observation_tensor)
        actions_probs = action_distribution.distribution.probs

        # Apply the action mask
        masked_logits = action_probs + (1 - action_mask) * -1e9  # Set invalid actions to a large negative value

        # Compute action probabilities
        action_probabilities = np.exp(masked_logits) / np.sum(np.exp(masked_logits))

        # Sample or select a valid action
        action = np.random.choice(len(action_mask), p=action_probabilities)
        """

        action, _ = self.model.predict(observation)

        return action
