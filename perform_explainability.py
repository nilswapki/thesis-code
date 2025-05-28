from helper_eval import initialize_learner_with_flags
from policies.learner import Learner

import numpy as np
import pandas as pd
from timeshap.explainer.local_methods import calc_local_report
from timeshap.utils import calc_avg_event
import matplotlib
from matplotlib import pyplot as plt
import torch
import torchkit.pytorch_utils as ptu
import utils.helpers as utl
import re

# this can be changed if needed
if not torch.cuda.is_available():
    matplotlib.use('TkAgg')  # Use the standard interactive backend


def generate_trajs(learner: Learner, num_trajs: int):

    """
    Generates trajectories by simulating the agent's behavior in the evaluation environment.

    Args:
        learner (Learner): The learner object containing the agent and environment configurations.
        num_trajs (int): The number of trajectories to generate.

    Returns:
        trajs (list): A list of trajectories, where each trajectory is a numpy array of observations.
    """

    # Evaluate the model
    learner.agent.eval()  # Set the agent to evaluation mode
    trajs = []
    infiltrations = []
    restorations = []

    for i in range(num_trajs):
        print(f"Generating trajectory {i + 1}/{num_trajs}")
        # Get the initial action, reward, and internal state from the agent
        action, reward, internal_state = learner.agent.get_initial_info(learner.config_seq.sampled_seq_len)
        # Reset the environment
        obs, info = learner.eval_env.reset()

        # Convert observation to tensor and ensure correct shape
        if obs.shape[0] == 1:
            obs = obs.reshape(-1, 1)
        obs = ptu.from_numpy(obs)

        trajectory = []  # Store trajectory for TimeSHAP in numpy array
        infiltrations_episode = []  # Store infiltrations for this episode
        restorations_episode = []  # Store restorations for this episode

        done = False
        while not done:
            # Store the current observation for TimeSHAP
            trajectory.append(obs.clone().detach())

            # Convert tensors to the appropriate device
            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'mps:0'
            action = action.to(device)
            reward = reward.to(device)
            obs = obs.to(device)

            # Get the next action from the agent
            action, internal_state = learner.agent.act(
                prev_internal_state=internal_state,
                prev_action=action,
                reward=reward,
                obs=obs,
                deterministic=True,
            )

            # Step the environment
            next_obs, reward, done, info = utl.env_step(learner.eval_env, action.squeeze(dim=0))
            obs = next_obs.clone()

            if learner.FLAGS.config_env.env_type == "network-defender":
                infiltrations_episode.append(info["newly_infiltrated"])
                restorations_episode.append(info["restoration_occured"])

        # Store the last observation
        trajectory = np.stack([obs.cpu().numpy() for obs in trajectory])
        # Ensure trajectory is in the correct shape (T, B, F)
        trajectory = np.transpose(trajectory, (2, 0, 1))
        # Append the trajectory to the list
        trajs.append(trajectory[:, :, :])

        if learner.FLAGS.config_env.env_type == "network-defender":
            infiltrations.append(infiltrations_episode)
            restorations.append(restorations_episode)

        return trajs


def model_wrapper(obs_batch: np.ndarray):

    """
    Wraps the model's inference logic for use in explainability (e.g., TimeSHAP).

    Args:
        obs_batch (np.ndarray): A NumPy array of shape (batch_size, sequence_length, num_features),
            containing the input observation sequences for multiple samples.

    Returns:
        np.ndarray: A NumPy array of shape (batch_size, 1), containing the final predicted action
        index for each sample in the batch.
    """

    with torch.no_grad():
        actions = []
        # Get the initial action, reward, and internal state from the agent
        action, reward, internal_state = learner.agent.get_initial_info(learner.config_seq.sampled_seq_len)

        # Convert obs_batch to tensor (ensure it has the correct shape)
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

                # Get the next action from the agent
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


def explain(learner: Learner, num_trajs: int = 3, last_k: int = 30, top_k: int = 20, model="", tag="", all_ones=False):

    """
    Generates and saves TimeSHAP-based explainability visualizations for a given RL agent.

    This function evaluates the agent in the provided learner setup, extracts a number of trajectories,
    and computes Shapley values for both temporal events (timesteps) and input features using TimeSHAP.
    It then saves and visualizes these attributions to help understand which timesteps and features were
    most influential in the agent's decision-making.

    Args:
        learner (Learner): The learner object containing the agent and environment configurations.
        num_trajs (int, optional): Number of trajectories to generate for explainability. Defaults to 3.
        last_k (int, optional): Number of last events to consider for event-wise Shapley value plots. Defaults to 30.
        top_k (int, optional): Number of top features to consider for feature-wise Shapley value plots. Defaults to 20.
        model (str, optional): Model name used for saving explainability results. Defaults to an empty string.
        tag (str, optional): Tag to differentiate explainability runs, used in output file names. Defaults to an empty string.

    Returns:
        None
    """

    # Evaluate the model
    learner.agent.eval()  # Set the agent to evaluation mode

    # Generate trajectories
    trajectories = generate_trajs(learner, num_trajs=num_trajs)

    # Generate model features
    model_features = [f'{i}' for i in range(trajectories[0].shape[2])]
    # The plotting dictionary should map features to themselves if there are no custom labels
    if learner.FLAGS.config_env.env_type == "mini-cage":
        # Use the environment's feature descriptions
        plot_features = {f'{f}': learner.eval_env.describe_feature(feature_index=f) for f in range(trajectories[0].shape[2])}  # learner.eval_env.describe_feature(feature_index=f)
    else:
        plot_features = {f'{f}': f'Feature {f}' for f in range(trajectories[0].shape[2])}

    if not all_ones:  # if we want to calculate the average event
        # Prepare the average event data
        stacked = np.concatenate(trajectories, axis=0)
        reshaped = stacked.reshape(-1, stacked.shape[-1])
        avg_data = pd.DataFrame(reshaped)
        avg_data.columns = avg_data.columns.astype(str)
        # Calculate the average event
        avg_event = calc_avg_event(data=avg_data,
                                   numerical_feats=model_features, categorical_feats=[]).astype(float)
    else:  # if we want to use an all-ones event
        avg_event = pd.DataFrame(np.ones((1, 78)))
        avg_event.columns = avg_event.columns.astype(str)

    # Prepare TimeSHAP input dictionaries
    pruning_dict = {'tol': 0.001}
    event_dict = {'rs': 42, 'nsamples': 100}
    feature_dict = {'rs': 42, 'nsamples': 100, 'feature_names': model_features, 'plot_features': plot_features}
    cell_dict = {'rs': 42, 'nsamples': 100, 'top_x_feats': 1, 'top_x_events': 1}

    events = []
    features = []

    # Iterate over trajectories and compute Shapley values
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
        for _, row in event_data.iterrows():  # iterate over rows in the DataFrame
            label = row["Feature"]  # get the label from the "Feature" column
            if label == "Pruned Features" or label == "Pruned Events":  # check for special labels
                idx = 0
            else:
                match = re.search(r"Event (-?\d+)", label)  # match against the Event regex pattern
                if match:
                    idx = int(match.group(1))  # extract the event index from the label
                else:  # if the label does not match the expected format
                    print(f"Unrecognized event label: {label}")
                    continue

            if 0 != idx:  # if idx is not the pruned event slot
                event_array[idx] = abs(row["Shapley Value"])  # store the absolute Shapley value
            elif label == "Pruned Features" or label == "Pruned Events":  # if it's a pruned event
                print(f"Invalid event index: {idx} for label: {label}")

        # === Handle features ===
        for _, row in feature_data.iterrows():  # iterate over rows in the DataFrame
            label = row["Feature"]  # get the label from the "Feature" column
            if label == "Pruned Features" or label == "Pruned Events":  # check for special labels
                idx = -1  # use -1 for pruned features/events
            else:
                idx = int(label)

            if -1 != idx:  # if idx is not the pruned feature slot
                feature_array[idx] = abs(row["Shapley Value"])  # store the absolute Shapley value
            elif label == "Pruned Features" or label == "Pruned Events":  # if it's a pruned feature
                print(f"Invalid feature index: {idx} for label: {label}")

        events.append(event_array)
        features.append(feature_array)

    # Convert to np arrays for easy slicing/plotting
    events = np.array(events)  # shape: (num_trajs, num_events)
    features = np.array(features)  # shape: (num_trajs, num_features)

    # Save events
    events_save_path = f"final_results/explainability/{learner.FLAGS.config_env.env_type}_{model}_{tag}_events_last{last_k}_traj{num_trajs}.npy"
    np.save(events_save_path, events)

    # Save features_top
    features_save_path = f"final_results/explainability/{learner.FLAGS.config_env.env_type}_{model}_{tag}_features_top{top_k}_traj{num_trajs}.npz"
    np.savez(features_save_path,
             features=features,
             plot_features=plot_features)

    # Plotting
    plot_event_multi(events, last_k, save_path=f"final_results/explainability/{learner.FLAGS.config_env.env_type}_{model}_{tag}_events_last{last_k}_traj{num_trajs}.png")
    plot_feature_multi(features, plot_features, top_k, save_path=f"final_results/explainability/{learner.FLAGS.config_env.env_type}_{model}_{tag}_features_top{top_k}_traj{num_trajs}.png")
    plt.show()


def plot_pruning(pruning_data, cutoff_t):
    """
    Visualizes which events were pruned based on their Shapley value contributions.

    Args:
        pruning_data (pd.DataFrame or list of dict): Output from the TimeSHAP pruning step,
            containing Shapley value contributions grouped by event index.
        cutoff_t (float): The cutoff threshold (typically determined via a pruning tolerance)
            used to distinguish significant from negligible contributions.

    Returns:
        None. Displays and saves the plot as 'pruning_contribution.png'.
    """

    # Convert to DataFrame
    df = pd.DataFrame(pruning_data)

    # Separate data for plotting
    greater_t = df[df["Coalition"] == "Sum of contribution of events > t"]
    lesser_t = df[df["Coalition"] == "Sum of contribution of events ≤ t"]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.4

    # Plot bars for greater and lesser t
    ax.bar(greater_t["t (event index)"] - width / 2, greater_t["Shapley Value"], width=width, label="> t", color='blue', alpha=0.6)
    ax.bar(lesser_t["t (event index)"] + width / 2, lesser_t["Shapley Value"], width=width, label="≤ t", color='orange', alpha=0.6)

    # Add horizontal line for cutoff
    plt.axhline(y=cutoff_t, color='red', linestyle='--', label=f'Cutoff (t={cutoff_t})')

    plt.xlabel("Event Index (t)")
    plt.ylabel("Shapley Value")
    plt.title("Pruning Contribution by Event Index")
    plt.legend()
    plt.savefig('pruning_contribution.png')


def plot_event_multi(events=None, last_k=30, load_path=None, save_path="shapley_event_plot.png"):
    """
    Plots event-wise Shapley values for multiple trajectories.

    Args:
    events (np.ndarray, optional): A NumPy array of shape (num_trajectories, num_events),
        where each row contains the Shapley values for one trajectory. If None, `load_path`
        must be provided to load the data from file.
    last_k (int, optional): Number of most recent events to include in the plot. Defaults to 30.
    load_path (str, optional): Path to a `.npy` file containing precomputed event-level Shapley
        values. Only used if `events` is None.
    save_path (str, optional): File path to save the generated plot. Defaults to "shapley_event_plot.png".

    Returns:
        None. The function saves the plot to `save_path` and optionally displays it.
    """
    if events is None:  # if events are not provided, load from file
        assert load_path is not None, "Either events or load_path must be provided."
        events = np.load(load_path)
        save_path = load_path.replace(".npy", ".png")

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

    plt.figure(figsize=(16, 6))

    # Plot individual Shapley values
    for i, x in enumerate(x_labels):
        y_vals = events[:, i]
        plt.scatter(
            [x] * len(y_vals), y_vals,  # x is the event index, y is the Shapley value
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

    """
    Plots feature-wise Shapley values for multiple trajectories.

    Args:
        features (np.ndarray, optional): A NumPy array of shape (num_trajectories, num_features),
            where each row contains the feature-level Shapley values for one trajectory. If None,
            `load_path` must be provided to load the data from file.
        plot_features (dict, optional): A dictionary mapping feature indices (as strings) to
            human-readable labels used for plotting.
        top_k (int, optional): Number of top features (by average importance) to display. Defaults to 10.
        load_path (str, optional): Path to a `.npz` file containing precomputed feature-level Shapley
            values. Only used if `features` is None.
        save_path (str, optional): File path to save the generated plot. Defaults to "shapley_feature_plot.png".

    Returns:
        None. The function saves the plot to `save_path` and optionally displays it.
    """

    if features is None:  # if features are not provided, load from file
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


if __name__ == "__main__":
    # Initialize the learner with flags
    learner = initialize_learner_with_flags(save_dir='final_results/mini-cage/final/lstm/seed-1')

    # Generate and explain trajectories
    explain(learner, num_trajs=5, last_k=50, top_k=15, model="lstm", tag="test")

    # Uncomment to plot events and features from saved files
    #plot_event_multi(load_path='explainability/network-defender_mlp_final-avg_events_last50_traj100.npy', last_k=50)
    #plot_feature_multi(load_path='explainability/network-defender_lru_final_features_top10_traj100.npz', top_k=15)

