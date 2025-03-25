from main_eval import initialize_learner_with_flags
from policies.learner import Learner

import shap
import timeshap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from timeshap.explainer.local_methods import local_report, calc_local_report
from timeshap.explainer.global_methods import global_report
from timeshap.utils import calc_avg_event
import matplotlib
from matplotlib import pyplot as plt
import torch
import torchkit.pytorch_utils as ptu
import utils.helpers as utl
import altair_saver

matplotlib.use('TkAgg')  # Use the standard interactive backend


def explain(learner: Learner):
    # Evaluate the model
    learner.agent.eval()  # Set the agent to evaluation mode

    # Reset the environment
    action, reward, internal_state = learner.agent.get_initial_info(learner.config_seq.sampled_seq_len)
    obs, _ = learner.eval_env.reset()
    obs = ptu.from_numpy(obs)
    trajectory = []  # Store trajectory for TimeSHAP in numpy array

    done = False
    while not done:
        # Store the current observation for TimeSHAP
        trajectory.append(obs.clone().detach())

        device = 'mps:0'
        action = action.to(device)
        reward = reward.to(device)
        obs = obs.to(device)

        # Select action
        action, internal_state = learner.agent.act(
            prev_internal_state=internal_state,
            prev_action=action,
            reward=reward,
            obs=obs,
            deterministic=True,
        )

        action_index = torch.argmax(action, dim=-1).item()
        print(f"Action: {action_index}, Reward: {reward}, Done: {done}")

        # Step the environment
        next_obs, reward, done, info = utl.env_step(learner.eval_env, action.squeeze(dim=0))
        obs = next_obs.clone()

    # === TimeSHAP Explainability === #
    print("\nCalculating TimeSHAP explanations...")

    # Define the model wrapper that works with TimeSHAP
    def model_wrapper(obs_batch):
        print('Model Wrapper called')
        with torch.no_grad():
            actions = []
            action, reward, internal_state = learner.agent.get_initial_info(learner.config_seq.sampled_seq_len)

            # Convert obs_batch to tensor (ensure it's the correct shape)
            obs_batch = torch.tensor(obs_batch, dtype=torch.float32)

            # Iterate over batch dimension (#samples)
            for sample in range(obs_batch.shape[0]):
                internal_state = learner.agent.get_initial_info(learner.config_seq.sampled_seq_len)[
                    2]  # Reset internal state for each sample
                sample_actions = []

                # Iterate over sequence length
                for i in range(obs_batch.shape[1]):
                    obs_adjusted = obs_batch[sample, i, :].unsqueeze(0)  # Shape: (1, 1, #features)

                    device = 'mps:0'
                    action = action.to(device)
                    reward = reward.to(device)
                    obs_adjusted = obs_adjusted.to(device)


                    action, internal_state = learner.agent.act(
                        prev_internal_state=internal_state,  # shape (1,1,1,64)
                        prev_action=action,  # shape (1,1,1,56)
                        reward=reward,  # shape (1,1)
                        obs=obs_adjusted,  # shape (1,1,40)
                        deterministic=True,
                    )

                    # Select last time step from the sequence (T dimension)
                    last_action = action #action[:, :, -1, :]  # Shape: (1, 1, 56)

                    # Squeeze out batch and sample dimensions -> (56,)
                    last_action = last_action.squeeze(0)
                    sample_actions.append(torch.argmax(last_action, dim=-1).item())

                # Append the final action of each sample
                actions.append(sample_actions[-1])

            return np.array(actions).reshape(-1, 1)  # Ensure shape (#samples, 1)

    # Convert trajectory to numpy array
    trajectory = np.stack([obs.cpu().numpy() for obs in trajectory])
    trajectory = trajectory[:20]

    model_features = [i for i in range(obs.shape[0])]
    # The plotting dictionary should map features to themselves if there are no custom labels
    plot_features = {f: learner.eval_env.describe_feature(feature_index=f) for f in model_features}
    avg_event = calc_avg_event(data=pd.DataFrame(trajectory.squeeze(2)), numerical_feats=model_features, categorical_feats=[]).astype(int)
    # Prepare TimeSHAP input dictionaries
    pruning_dict = {'tol': 0.1}
    event_dict = {'rs': 42, 'nsamples': 100}
    feature_dict = {'rs': 42, 'nsamples': 100, 'feature_names': model_features, 'plot_features': plot_features}
    cell_dict = {'rs': 42, 'nsamples': 100, 'top_x_feats': 3, 'top_x_events': 3}

    data = np.transpose(trajectory, (2, 0, 1))

    # Generate local report and plot using TimeSHAP
    pruning_data, event_data, feature_data, cell_level = calc_local_report(
        f=model_wrapper,
        data=data,
        pruning_dict=pruning_dict,
        event_dict=event_dict,
        feature_dict=feature_dict,
        cell_dict=cell_dict,
        baseline=avg_event
    )

    plot_pruning(pruning_data, cutoff_t=pruning_dict['tol'])
    plot_event(event_data)

    plt.show()

def plot_pruning(pruning_data, cutoff_t):
    # Convert to DataFrame
    df = pd.DataFrame(pruning_data)

    # Separate data for plotting
    greater_t = df[df["Coalition"] == "Sum of contribution of events > t"]
    lesser_t = df[df["Coalition"] == "Sum of contribution of events ≤ t"]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.4

    ax.bar(greater_t["t (event index)"] - width / 2, greater_t["Shapley Value"], width=width, label="> t", color='blue', alpha=0.6)
    ax.bar(lesser_t["t (event index)"] + width / 2, lesser_t["Shapley Value"], width=width, label="≤ t", color='orange', alpha=0.6)

    plt.axhline(y=cutoff_t, color='red', linestyle='--', label=f'Cutoff (t={cutoff_t})')

    plt.xlabel("Event Index (t)")
    plt.ylabel("Shapley Value")
    plt.title("Pruning Contribution by Event Index")
    plt.legend()
    plt.savefig('pruning_contribution.png')


def plot_event(event_data, sort=False):
    if sort:
        # Sort features by absolute Shapley Value (most important first)
        event_data_sorted = event_data.sort_values(by="Shapley Value", key=abs, ascending=True)
    else:
        event_data_sorted = event_data

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x="Shapley Value",
        y="Feature",
        data=event_data_sorted,
        palette="coolwarm"
    )

    # Add title and labels
    plt.title("Feature Importance based on Shapley Values")
    plt.xlabel("Shapley Value")
    plt.ylabel("Feature")

    # Add a vertical line at zero for reference
    plt.axvline(x=0, color="black", linestyle="--", alpha=0.7)
    plt.savefig('event_importance.png')


if __name__ == "__main__":
    learner = initialize_learner_with_flags(save_dir='logs_results/mini-cage/final/standard/lstm/seed-1')
    explain(learner)
