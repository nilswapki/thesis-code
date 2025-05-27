from .base import RLAlgorithmBase
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torchkit.networks import FlattenMlp
from policies.models.actor import MarkovPolicyBase
from typing import Any, Tuple


class PPO(RLAlgorithmBase):
    name = "ppo"
    continuous_action = False
    use_target_actor = False

    def __init__(self, clip_param=0.2, entropy_coef=0.01, **kwargs):
        super().__init__(**kwargs)
        self.clip_param = clip_param
        self.learning_rate = kwargs.get("learning_rate", 3e-4)
        self.entropy_coef = entropy_coef

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes) -> MarkovPolicyBase:
        return FlattenMlp(input_size=input_size, output_size=action_dim, hidden_sizes=hidden_sizes)

    @staticmethod
    def build_critic(input_size, hidden_sizes, **kwargs) -> Tuple[Any, Any]:
        critic = FlattenMlp(input_size=input_size, output_size=1, hidden_sizes=hidden_sizes)
        return critic, critic  # PPO usually uses a single critic, but returning two for consistency

    def select_action(self, actor, observ, deterministic: bool) -> Any:
        action_logits = actor(observ)

        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            dist = Categorical(logits=action_logits)  # Use logits directly
            action = dist.sample()

        action = F.one_hot(
            action.long(), num_classes=action_logits.shape[-1]
        ).float()  # (*, A)

        return action

    @staticmethod
    def forward_actor(actor, observ) -> Tuple[Any, Any]:
        action_logits = actor(observ)
        dist = Categorical(logits=action_logits)  # No need for softmax
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

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
            next_observs=None,  # used in markov_critic
    ):
        with torch.no_grad():
            if markov_critic:
                # Compute value estimate for next observations (Markov case)
                next_v_target = critic_target(next_observs)  # (B, 1)
            else:
                # Compute value estimate for next hidden states (Non-Markov case)
                next_v_target = critic_target(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    current_actions=None,
                )  # (T+1, B, 1)

            q_target = rewards + (1.0 - dones) * gamma * next_v_target[0]  # (T, B, 1)

            if not markov_critic:
                q_target = q_target[1:]  # Ignore last timestep (T, B, 1)

        # Compute current value estimates
        if markov_critic:
            v_pred = critic(observs)  # (B, 1)
        else:
            v_pred = critic(
                prev_actions=actions[:-1],
                rewards=rewards[:-1],
                observs=observs[:-1],
                current_actions=None,
            )  # (T, B, 1)

        return v_pred, q_target

    def actor_loss(
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
    ) -> Tuple[Any, Any]:

        _, log_probs = actor(prev_actions=actions, rewards=rewards, observs=observs)

        with torch.no_grad():
            value_pred = critic(prev_actions=actions, rewards=rewards, observs=observs, current_actions=None)[0].squeeze(-1)
            advantage = rewards.squeeze(-1) - value_pred

        ratio = torch.exp(log_probs - log_probs.detach())  # PPO ratio calculation
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()

        action_probs = log_probs.exp()  # Convert log_probs back to probabilities
        dist = Categorical(probs=action_probs)  # Define categorical distribution

        entropy = dist.entropy().mean()
        actor_loss -= self.entropy_coef * entropy  # Encourage exploration

        return actor_loss, log_probs.unsqueeze(-1)

    def update_others(self, current_log_probs):
        return {"entropy": -current_log_probs, "coef": 1.0}
