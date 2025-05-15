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

if not torch.cuda.is_available():
    matplotlib.use('TkAgg')  # Use the standard interactive backend


def generate_trajs(learner: Learner, num_trajs: int):
    # Evaluate the model
    learner.agent.eval()  # Set the agent to evaluation mode
    trajs = []
    infiltrations = []
    restorations = []

    for i in range(num_trajs):
        print(f"Generating trajectory {i + 1}/{num_trajs}")
        # Reset the environment
        action, reward, internal_state = learner.agent.get_initial_info(learner.config_seq.sampled_seq_len)
        obs, info = learner.eval_env.reset()
        if obs.shape[0] == 1:
            obs = obs.reshape(-1, 1)
        obs = ptu.from_numpy(obs)

        trajectory = []  # Store trajectory for TimeSHAP in numpy array
        infiltrations_episode = []  # Store infiltrations for this episode
        restorations_episode = []

        done = False
        while not done:
            # Store the current observation for TimeSHAP
            trajectory.append(obs.clone().detach())

            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
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

            if learner.FLAGS.config_env.env_type == "network-defender":
                infiltrations_episode.append(info["newly_infiltrated"])
                restorations_episode.append(info["restoration_occured"])

        trajectory = np.stack([obs.cpu().numpy() for obs in trajectory])
        trajectory = np.transpose(trajectory, (2, 0, 1))
        trajs.append(trajectory[:, :, :])

        if learner.FLAGS.config_env.env_type == "network-defender":
            infiltrations.append(infiltrations_episode)
            restorations.append(restorations_episode)

    return trajs #, infiltrations, restorations


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

                if torch.cuda.is_available():
                    device = 'cuda:0'
                else:
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
    trajectories = generate_trajs(learner, num_trajs=num_trajs)

    model_features = [f'{i}' for i in range(trajectories[0].shape[2])]
    # The plotting dictionary should map features to themselves if there are no custom labels
    if learner.FLAGS.config_env.env_type == "mini-cage":
        plot_features = {f'{f}': learner.eval_env.describe_feature(feature_index=f) for f in range(trajectories[0].shape[2])}  # learner.eval_env.describe_feature(feature_index=f)
    else:
        plot_features = {f'{f}': f'Feature {f}' for f in range(trajectories[0].shape[2])}

    #avg_data = pd.DataFrame(np.concatenate([traj.squeeze(0) for traj in trajectories], axis=0))
    #avg_data = pd.DataFrame(trajectories.pop(0).squeeze(0))

    all_ones = True
    if not all_ones:
        stacked = np.concatenate(trajectories, axis=0)
        reshaped = stacked.reshape(-1, stacked.shape[-1])
        avg_data = pd.DataFrame(reshaped)

        avg_data.columns = avg_data.columns.astype(str)
        avg_event = calc_avg_event(data=avg_data,
                                   numerical_feats=model_features, categorical_feats=[]).astype(float)
    else:
        avg_event = pd.DataFrame(np.ones((1, 78)))
        avg_event.columns = avg_event.columns.astype(str)

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

    # Save events
    events_save_path = f"explainability/{learner.FLAGS.config_env.env_type}_{model}_{tag}_events_last{last_k}_traj{num_trajs}.npy"
    np.save(events_save_path, events)

    # Save features_top
    features_save_path = f"explainability/{learner.FLAGS.config_env.env_type}_{model}_{tag}_features_top{top_k}_traj{num_trajs}.npz"
    np.savez(features_save_path,
             features=features,
             plot_features=plot_features)

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

    plot_feature_multi(features, plot_features, top_k, save_path=f"explainability/{learner.FLAGS.config_env.env_type}_{model}_{tag}_features_top{top_k}_traj{num_trajs}.png")

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


def plot_event_multi(events=None, last_k=30, load_path=None, save_path="shapley_event_plot.png"):
    if events is None:
        assert load_path is not None, "Either events or load_path must be provided."
        events = np.load(load_path)
        save_path = load_path.replace(".npy", ".png")

    events = np.array(events)
    events = events[:, -last_k:]  # keep all trajs, last k timesteps

    # Scaling factor to shift event 0 mean to 0.5
    #scale = 0.5 / np.max(mean_vals) if np.max(mean_vals) != 0 else 1.0
    #events = events * scale


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

    plt.figure(figsize=(16, 6))

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

    mean_vals = np.mean(events, axis=0)
    # Plot mean Shapley values
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
    plt.title("Event-wise Shapley Values", fontsize=14, fontweight='bold')

    # Grid: horizontal only, dotted and semi-transparent
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.grid(axis='x', visible=False)

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_multi(features=None, plot_features=None, top_k=10, load_path=None, save_path="shapley_feature_plot.png"):
    if features is None:
        assert load_path is not None, "Either events or load_path must be provided."
        loaded = np.load(load_path, allow_pickle=True)
        features = loaded['features']
        plot_features = loaded['plot_features'].item()  # because it's a saved dict
        plot_features = {k: v.replace('Event', 'Node') if isinstance(v, str) else v for k, v in plot_features.items()}
        save_path = load_path.replace(".npz", ".png")
    else:
        features = np.array(features)

    # Compute mean absolute Shapley values
    importance = np.mean(np.abs(features), axis=0)
    top_indices = np.argsort(importance)[-top_k:][::-1]

    # Slice features and names
    features = features[:, top_indices]
    feature_names = [plot_features[str(idx)] for idx in top_indices]

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


def plot_infiltration_timing(infiltrations, save_path="infiltration_timing_plot.png"):
    """
    infiltrations: List of lists (episodes × timesteps) containing lists of newly infiltrated nodes
    """
    # Flatten into a list of (episode, timestep) where infiltration happened
    infiltration_times = []
    for ep_idx, episode in enumerate(infiltrations):
        for t_idx, newly_infiltrated in enumerate(episode):
            if newly_infiltrated:  # if list is not empty
                infiltration_times.extend([t_idx] * len(newly_infiltrated))

    if len(infiltration_times) == 0:
        print("No infiltrations recorded.")
        return

    infiltration_times = np.array(infiltration_times)

    infiltration_counts = np.zeros(100, dtype=int)
    for t in infiltration_times:
        infiltration_counts[t] += 1

    # Plotting
    fig, ax = plt.subplots(figsize=(16, 6))
    bars = ax.bar(np.arange(50), infiltration_counts[50:], color='mediumturquoise', edgecolor='black', alpha=0.7)

    # Set custom x-ticks
    ax.set_xticks(np.arange(0, 50))  # 50 bars
    ax.set_xticklabels([str(-i) for i in range(49, -1, -1)])  # From -49 to 0

    # Axis labels
    ax.set_xlabel("Timesteps (relative to end)", fontsize=12)
    ax.set_ylabel("Number of Infiltrations", fontsize=12)
    ax.set_title("Infiltrations in Last 50 Timesteps", fontsize=14, fontweight='bold')

    # Grid styling
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.grid(axis='x', visible=False)

    plt.tight_layout()
    plt.show()


def plot_infiltration_nodes(infiltrations, save_path=""):
    node_count = np.zeros(15, dtype=int)
    for episode in infiltrations:
        for newly_infiltrated in episode:
            for node in newly_infiltrated:
                node_count[node] += 1

    num_nodes = len(node_count)
    node_ids = np.arange(num_nodes)

    # Sort by infiltration count (ascending for "most at bottom")
    sort_idx = np.argsort(-node_count)
    sorted_counts = node_count[sort_idx]
    sorted_labels = [f"Node {i}" for i in sort_idx]

    # Plot
    plt.figure(figsize=(10, max(6, num_nodes * 0.4)))
    y_positions = np.arange(num_nodes)

    plt.barh(y_positions, sorted_counts, color='mediumturquoise', edgecolor='black', alpha=0.7)
    plt.yticks(y_positions, sorted_labels)
    plt.xlabel("Number of Infiltrations", fontsize=12)
    plt.ylabel("Node", fontsize=12)
    plt.title("Node-wise Infiltration Frequency", fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_variance_timesteps(trajs):
    # Stack into (num_trajs, 100, 15)
    stacked = np.concatenate(trajs, axis=0)  # shape: (num_trajs, 100, 15)

    # Compute variance across trajectories: axis=0
    var_across_trajs = np.var(stacked, axis=0)  # shape: (100, 15)

    # Average variance over the 15 features (nodes) per timestep
    mean_variance_per_timestep = np.mean(var_across_trajs, axis=1)  # shape: (100,)

    # Bar plot
    plt.figure(figsize=(16, 5))
    plt.bar(np.arange(50), mean_variance_per_timestep[50:], color='mediumturquoise', edgecolor='black', alpha=0.7)

    plt.xlabel("Timestep (relative to end)", fontsize=12)
    plt.ylabel("Avg Variance Across Nodes", fontsize=12)
    plt.title("Observation Variance Across Trajectories (Last 50 Timesteps)", fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    #plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_restoration_occured(restorations):
    # Flatten into a list of (episode, timestep) where infiltration happened
    restoration_times = []
    for ep_idx, episode in enumerate(restorations):
        for t_idx, restored in enumerate(episode):
            if restored:  # if list is not empty
                restoration_times.extend([t_idx])

    if len(restoration_times) == 0:
        print("No infiltrations recorded.")
        return

    infiltration_times = np.array(restoration_times)

    restoration_counts = np.zeros(100, dtype=int)
    for t in restoration_times:
        restoration_counts[t] += 1

    # Plotting
    fig, ax = plt.subplots(figsize=(16, 6))
    bars = ax.bar(np.arange(50), restoration_counts[50:], color='mediumturquoise', edgecolor='black', alpha=0.7)

    # Set custom x-ticks
    ax.set_xticks(np.arange(0, 50))  # 50 bars


    # Axis labels
    ax.set_xlabel("Timesteps (relative to end)", fontsize=12)
    ax.set_ylabel("Number of Restorations", fontsize=12)
    ax.set_title("Restorations in Last 50 Timesteps", fontsize=14, fontweight='bold')
    ax.set_ylim(bottom=50)
    # Grid styling
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.grid(axis='x', visible=False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    learner = initialize_learner_with_flags(save_dir='logs_results/mini-cage/final/standard/mlp/seed-1')
    explain(learner, num_trajs=100, last_k=50, top_k=15, model="mlp", tag="final_ones")

    #trajs, infiltrations, restorations = generate_trajs(learner, num_trajs=100)
    #plot_infiltration_nodes(infiltrations, save_path="infiltration_timing_plot.png")
    #plot_infiltration_timing(infiltrations)
    #plot_variance_timesteps(trajs)
    #plot_restoration_occured(restorations)

    #plot_event_multi(load_path='explainability/network-defender_mlp_final-avg_events_last50_traj100.npy', last_k=50)
    #plot_feature_multi(load_path='explainability/network-defender_lru_final_features_top10_traj100.npz', top_k=15)

