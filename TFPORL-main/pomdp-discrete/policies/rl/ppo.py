"""
Proximal Policy Optimization (PPO) implementation.
This class implements the PPO algorithm, which is a policy gradient method for reinforcement learning.
Attributes:
    name (str): The name of the algorithm.
    continuous_action (bool): Indicates if the action space is continuous.
    use_target_actor (bool): Indicates if a target actor network is used.
    clip_param (float): The clipping parameter for PPO.
    ppo_epochs (int): The number of epochs for PPO updates.
    mini_batch_size (int): The size of mini-batches for PPO updates.
Methods:
    __init__(clip_param=0.2, ppo_epochs=10, mini_batch_size=64, **kwargs):
        Initializes the PPO algorithm with the given parameters.
    build_actor(input_size, action_dim, hidden_sizes, **kwargs):
        Builds the actor network.
    build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        Builds the critic network.
    select_action(actor, observ, deterministic: bool):
        Selects an action based on the actor network and the observation.
    forward_actor(actor, observ):
        Computes the action logits and probabilities from the actor network.
    critic_loss(markov_actor: bool, markov_critic: bool, actor, actor_target, critic, critic_target, observs, actions, rewards, dones, gamma, next_observs):
        Computes the loss for the critic network.
    actor_loss(markov_actor: bool, markov_critic: bool, actor, actor_target, critic, critic_target, observs, actions=None, rewards=None):
        Computes the loss for the actor network.
    update_others(current_log_probs):
        Updates other parameters such as entropy and coefficient.
Comments:
    # Step 1: Initialize the PPO algorithm with the given parameters.
    # Step 2: Build the actor network.
    # Step 3: Build the critic network.
    # Step 4: Select an action based on the actor network and the observation.
    # Step 5: Compute the action logits and probabilities from the actor network.
    # Step 6: Compute the loss for the critic network.
    # Step 7: Compute the loss for the actor network.
    # Step 8: Update other parameters such as entropy and coefficient.
Missing:
    - The code does not include the training loop where the PPO updates are performed.
    - There is no method for updating the actor and critic networks using the computed losses.
    - The code does not handle the storage and sampling of experience tuples (observations, actions, rewards, etc.).
"""

import torch
import numpy as np
from torch.optim import Adam
from .base import RLAlgorithmBase
from torchkit.networks import FlattenMlp
import torch.nn.functional as F
import torchkit.pytorch_utils as ptu


class PPO(RLAlgorithmBase):
    name = "ppo"
    continuous_action = False
    use_target_actor = False

    def __init__(self, clip_param=0.2, mini_batch_size=64, **kwargs):
        super().__init__(**kwargs)
        self.clip_param = clip_param
        self.mini_batch_size = mini_batch_size
        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.optimizer = Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, **kwargs):
        return FlattenMlp(
            input_size=input_size, output_size=action_dim, hidden_sizes=hidden_sizes
        )

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        critic = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )
        return critic, critic

    def select_action(self, actor, observ, deterministic: bool):
        action_logits = actor(observ)
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action_prob = F.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_prob, num_samples=1)
        return action

    @staticmethod
    def forward_actor(actor, observ):
        action_logits = actor(observ)
        action_prob = F.softmax(action_logits, dim=-1)
        return action_logits, action_prob

    def critic_loss(
        self,
        markov_actor: bool,
        markov_critic: bool,
        actor,
        actor_target,
        critic,
        critic_target,
        observs,
        actions,
        rewards,
        dones,
        gamma,
        next_observs,
    ):
        with torch.no_grad():
            value_pred = critic(next_observs).squeeze(-1)
            q_target = rewards + (1.0 - dones) * gamma * value_pred

        value_pred = critic(observs).squeeze(-1)
        critic_loss = F.mse_loss(value_pred, q_target)

        return (value_pred, value_pred), q_target

    def actor_loss(
        self,
        markov_actor: bool,
        markov_critic: bool,
        actor,
        actor_target,
        critic,
        critic_target,
        observs,
        actions=None,
        rewards=None,
    ):
        action_logits, action_probs = self.forward_actor(actor, observs)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)

        ratio = torch.exp(log_probs - actions)
        surr1 = ratio * rewards
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * rewards
        actor_loss = -torch.min(surr1, surr2).mean()

        return actor_loss, log_probs

    def update_others(self, current_log_probs):
        return {"entropy": -current_log_probs, "coef": 1.0}