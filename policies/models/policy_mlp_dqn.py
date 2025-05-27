"""
DQN MLP agent (adjusted from SAC/TD3 style agent)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchkit.pytorch_utils as ptu
from policies.models import *
from policies.rl import RL_ALGORITHMS


class ModelFreeOffPolicy_DQN_MLP(nn.Module):
    """
    DQN Agent using an MLP Q-network
    Only builds a critic (no separate actor)
    """

    ARCH = "markov"
    Markov_Actor = True
    Markov_Critic = True

    def __init__(
        self,
        obs_dim,
        action_dim,
        config_rl,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = config_rl.discount
        self.tau = config_rl.tau

        self.algo = RL_ALGORITHMS[config_rl.algo](**config_rl, action_dim=action_dim)

        # Build Q-network
        self.qf1 = self.algo.build_critic(
            input_size=obs_dim,
            action_dim=action_dim,
            hidden_sizes=config_rl.config_critic.hidden_dims,
        )
        self.qf1_optim = Adam(self.qf1.parameters(), lr=config_rl.critic_lr)

        # Target Q-network
        self.qf1_target = copy.deepcopy(self.qf1)

        # No actor needed for DQN
        self.qf2 = None
        self.qf2_optim = None
        self.policy = None
        self.policy_optim = None
        self.policy_target = None

    @torch.no_grad()
    def act(self, obs, deterministic=True, epsilon=0.0):
        """
        Take an action from the current Q-network.
        Epsilon-greedy exploration if deterministic=False.
        """
        return self.algo.select_action(qf=self.qf1, observ=obs.reshape(1, -1), deterministic=deterministic)

    def update(self, batch):
        """
        Perform one DQN update step: TD learning on the critic
        """
        observs, next_observs = batch["obs"], batch["obs2"]
        actions, rewards, dones = batch["act"], batch["rew"], batch["term"]

        ### 1. Critic loss
        q_pred = self.qf1(observs).gather(1, actions.long()).squeeze(-1)

        with torch.no_grad():
            # Target: max_a' Q_target(next_state, a')
            next_q = self.qf1_target(next_observs).max(dim=-1)[0]
            q_target = rewards + (1 - dones) * self.gamma * next_q

        critic_loss = F.mse_loss(q_pred, q_target)

        # Update critic
        self.qf1_optim.zero_grad()
        critic_loss.backward()
        self.qf1_optim.step()

        # Soft update target network
        self.soft_target_update()

        outputs = {
            "critic_loss": critic_loss.item(),
            "q1": q_pred.mean().item(),
        }
        return outputs

    def soft_target_update(self):
        """
        Soft update the target Q-network
        """
        ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
