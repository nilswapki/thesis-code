from main_eval import initialize_learner_with_flags
from policies.learner import Learner

import shap
import timeshap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from timeshap.explainer.local_methods import local_report, calc_local_report
from timeshap.explainer.global_methods import calc_global_explanations
from timeshap.utils import calc_avg_event
import matplotlib
from matplotlib import pyplot as plt
import torch
import torchkit.pytorch_utils as ptu
import utils.helpers as utl
import altair_saver
import re
from sklearn.preprocessing import minmax_scale

matplotlib.use('TkAgg')  # Use the standard interactive backend


def generate_trajs(learner: Learner, num_trajs: int):
    # Evaluate the model
    learner.agent.eval()  # Set the agent to evaluation mode
    trajs = []

    for i in range(num_trajs):
        print(f"Generating trajectory {i + 1}/{num_trajs}")
        # Reset the environment
        action, reward, internal_state = learner.agent.get_initial_info(learner.config_seq.sampled_seq_len)
        obs, _ = learner.eval_env.reset()
        if obs.shape[0] == 1:
            obs = obs.reshape(-1, 1)
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

            # Step the environment
            next_obs, reward, done, info = utl.env_step(learner.eval_env, action.squeeze(dim=0))
            obs = next_obs.clone()

        trajectory = np.stack([obs.cpu().numpy() for obs in trajectory])
        trajectory = np.transpose(trajectory, (2, 0, 1))
        trajs.append(trajectory[:, :, :])

    return trajs


def model_wrapper(obs_batch):
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


def explain(learner: Learner, num_trajs: int = 3, last_k: int = 30, top_k: int = 20, model="", tag=""):
    # Evaluate the model
    learner.agent.eval()  # Set the agent to evaluation mode

    # Generate trajectories
    trajectories = generate_trajs(learner, num_trajs=num_trajs+1)

    model_features = [f'{i}' for i in range(trajectories[0].shape[2])]
    # The plotting dictionary should map features to themselves if there are no custom labels
    plot_features = {f'{f}': f'Feature {f}' for f in range(trajectories[0].shape[2])}  # learner.eval_env.describe_feature(feature_index=f)
    #avg_data = pd.DataFrame(np.concatenate([traj.squeeze(0) for traj in trajectories], axis=0))
    avg_data = pd.DataFrame(trajectories.pop(0).squeeze(0))

    avg_data.columns = avg_data.columns.astype(str)
    avg_event = calc_avg_event(data=avg_data,
                               numerical_feats=model_features, categorical_feats=[]).astype(int)
    # Prepare TimeSHAP input dictionaries
    pruning_dict = {'tol': 0.001}
    event_dict = {'rs': 42, 'nsamples': 100}
    feature_dict = {'rs': 42, 'nsamples': 100, 'feature_names': model_features, 'plot_features': plot_features}
    cell_dict = {'rs': 42, 'nsamples': 100, 'top_x_feats': 1, 'top_x_events': 1}

    events = []
    features = []

    for i, traj in enumerate(trajectories):
        print(f"Explaining Trajectory {i + 1}/{num_trajs}")
        num_events = traj.shape[1]  # includes the pruned event slot
        num_features = traj.shape[2]  # includes the pruned feature slot

        # Run TimeSHAP
        _, event_data, feature_data, cell_data = calc_local_report(
            f=model_wrapper,
            data=traj,
            pruning_dict=pruning_dict,
            event_dict=event_dict,
            feature_dict=feature_dict,
            cell_dict=cell_dict,
            baseline=avg_event
        )

        # Initialize arrays with zeros (or np.nan if you prefer)
        event_array = np.zeros(num_events)
        feature_array = np.zeros(num_features)

        # === Handle events ===
        for _, row in event_data.iterrows():
            label = row["Feature"]
            if label == "Pruned Features" or label == "Pruned Events":
                idx = 0
                #array_idx = num_events - 1  # last column
            else:
                match = re.search(r"Event (-?\d+)", label)
                if match:
                    idx = int(match.group(1))
                    #array_idx = idx if idx >= 0 else num_events + idx
                else:
                    print(f"Unrecognized event label: {label}")
                    idx = 0
                    continue

            if 0 != idx:
                event_array[idx] = abs(row["Shapley Value"])
            elif label == "Pruned Features" or label == "Pruned Events":
                print(f"Invalid event index: {idx} for label: {label}")

        # === Handle features ===
        for _, row in feature_data.iterrows():
            label = row["Feature"]
            if label == "Pruned Features" or label == "Pruned Events":
                idx = -1
                #array_idx = num_events - 1  # last column
            else:
                idx = int(label)

            if -1 != idx:
                feature_array[idx] = abs(row["Shapley Value"])
            elif label == "Pruned Features" or label == "Pruned Events":
                print(f"Invalid feature index: {idx} for label: {label}")

        events.append(event_array)
        features.append(feature_array)

    # Convert to np arrays for easy slicing/plotting
    events = np.array(events)  # shape: (num_trajs, num_events)
    features = np.array(features)  # shape: (num_trajs, num_features)

    """
    data_list = []
    for i, traj in enumerate(trajectories):
        df = pd.DataFrame(traj.squeeze(0))  # Shape (100, 78)
        df["episode_id"] = i  # Assign unique ID for each trajectory
        data_list.append(df)
    df_all = pd.concat(data_list, ignore_index=True)  # Merge into one DataFrame

    prun_indexes, event_data, feat_data = calc_global_explanations(
        f=model_wrapper,
        data=df_all,
        pruning_dict=pruning_dict,
        event_dict=event_dict,
        feature_dict=feature_dict,
        baseline=avg_event,
        entity_col="episode_id",
    )
    """

    plot_event_multi(events, last_k, save_path=f"explainability/{learner.FLAGS.config_env.env_type}_{model}_{tag}_events_last{last_k}_traj{num_trajs}.png")

    # Compute mean absolute Shapley values
    importance = np.mean(np.abs(features), axis=0)
    top_indices = np.argsort(importance)[-top_k:][::-1]

    # Slice features and names
    features_top = features[:, top_indices]
    feature_names_top = [plot_features[str(idx)] for idx in top_indices]

    plot_feature_multi(features_top, feature_names_top, save_path=f"explainability/{learner.FLAGS.config_env.env_type}_{model}_{tag}_features_top{top_k}_traj{num_trajs}.png")

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


def plot_event_single(event_data, sort=False):
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


def plot_event_multi(events, last_k=10, save_path="shapley_event_plot.png"):
    events = np.array(events)
    events = events[:, -last_k:]  # keep all trajs, last k timesteps

    # Min-max scaling
    event_min = np.min(events)
    event_max = np.max(events)
    if event_min != event_max:
        events = (events - event_min) / (event_max - event_min)
    else:
        events = np.zeros_like(events)

    num_trajs, num_events = events.shape

    # X-axis: event indices from -N+1 to 0
    x_labels = list(range(-num_events + 1, 1))

    plt.figure(figsize=(12, 6))

    # Plot individual Shapley values
    for i, x in enumerate(x_labels):
        y_vals = events[:, i]
        plt.scatter(
            [x] * len(y_vals), y_vals,
            color='mediumturquoise',
            alpha=0.4,
            s=100,
            edgecolor='none',
            label='Shapley Value' if i == 0 else None
        )

    # Plot mean Shapley values
    mean_vals = np.mean(events, axis=0)
    plt.scatter(
        x_labels, mean_vals,
        color='orangered',
        s=100,
        zorder=3,
        label='Mean'
    )

    # Formatting
    plt.xticks(x_labels)
    plt.xlabel("Event index", fontsize=12)
    plt.ylabel("Shapley Value", fontsize=12)
    plt.title("Shapley Value", fontsize=14, fontweight='bold')

    # Grid: horizontal only, dotted and semi-transparent
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.grid(axis='x', visible=False)

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_multi(features, feature_names=None, save_path="shapley_feature_plot.png"):
    features = np.array(features)

    # Min-max scaling
    feature_min = np.min(features)
    feature_max = np.max(features)
    if feature_min != feature_max:
        features = (features - feature_min) / (feature_max - feature_min)
    else:
        features = np.zeros_like(features)

    num_trajs, num_features = features.shape

    # Set y-axis labels
    if feature_names is None:
        y_labels = [f"Feature {i}" for i in range(num_features)]
    else:
        y_labels = list(feature_names)

    y_positions = np.arange(num_features)

    plt.figure(figsize=(10, max(6, num_features * 0.4)))

    # Plot individual Shapley values (horizontal scatter)
    for i, y in enumerate(y_positions):
        x_vals = features[:, i]
        plt.scatter(
            x_vals, [y] * len(x_vals),
            color='mediumturquoise',
            alpha=0.4,
            s=100,
            edgecolor='none',
            label='Shapley Value' if i == 0 else None
        )

    # Plot mean Shapley values (horizontal red points)
    mean_vals = np.mean(features, axis=0)
    plt.scatter(
        mean_vals, y_positions,
        color='orangered',
        s=100,
        zorder=3,
        label='Mean'
    )

    # Formatting
    plt.yticks(y_positions, y_labels)
    plt.xlabel("Shapley Value", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title("Feature-wise Shapley Values", fontsize=14, fontweight='bold')

    # Grid: vertical only, dotted
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.grid(axis='y', visible=False)

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    learner = initialize_learner_with_flags(save_dir='logs_results/network-defender/final/lstm/seed-1')
    explain(learner, num_trajs=2, last_k=10, top_k=5, model="mamba", tag="test")
