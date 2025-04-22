import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
import pandas as pd
import io
import re
import pickle
import os
from ml_collections import config_flags
from absl import app, flags
import numpy as np

def plot_reward(data, window_size=10):
    """
    Plots the reward over episodes with a sliding window for smoothing.

    Args:
        data (pd.DataFrame): Data containing the reward column 'r'.
        window_size (int): Size of the sliding window for smoothing.
    """
    # Extract the reward column (r) from the dataframe
    rewards = data['return']

    # Calculate the moving average for smoothing
    smoothed_rewards = rewards.rolling(window=window_size, min_periods=1).mean()

    # Plot the rewards
    plt.figure(figsize=(12, 7))
    plt.plot(rewards, label='Reward (Original)', color='lightblue', alpha=0.6)
    plt.plot(smoothed_rewards, label=f'Reward (Smoothed, Window={window_size})', color='blue', linewidth=2)

    # Add labels, title, and legend
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.title('Reward Over Time (With Smoothing)', fontsize=16)
    plt.legend(fontsize=12)

    # Lighten the grid and background for better visibility
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.gca().set_facecolor('whitesmoke')

    return plt.gcf()

def plot_invalid_share(data, window_size=10):
    """
    Plots the invalid share over episodes with a sliding window for smoothing.

    Args:
        data (pd.DataFrame): Data containing the invalid share column.
        window_size (int): Size of the sliding window for smoothing.
    """
    # Extract the invalid share column from the dataframe
    invalid_share = data['invalid_share']

    # Calculate the moving average for smoothing
    smoothed_invalid_share = invalid_share.rolling(window=window_size, min_periods=1).mean()

    # Plot the invalid share
    plt.figure(figsize=(12, 7))
    plt.plot(invalid_share, label='Invalid Share (Original)', color='lightcoral', alpha=0.6)
    plt.plot(smoothed_invalid_share, label=f'Invalid Share (Smoothed, Window={window_size})', color='red', linewidth=2)

    # Add labels, title, and legend
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Invalid Share', fontsize=14)
    plt.title('Invalid Share Over Time (With Smoothing)', fontsize=16)
    plt.legend(fontsize=12)

    # Lighten the grid and background for better visibility
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.gca().set_facecolor('whitesmoke')

    # Save the figure and show
    plt.savefig('invalid_share_plot_smoothing.png')
    plt.show()

def parse_flags(file_path):
    extracted = {}

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            # Handle flags like --noshared_encoder
            if line.startswith("--"):
                if "=" in line:
                    key, value = line.split("=")
                    key = key.strip("-")
                    value = value.strip()

                    # Convert numeric values
                    if value.replace(".", "", 1).isdigit():
                        value = float(value) if "." in value else int(value)

                    extracted[key] = value
                else:
                    # Store flags without values as True
                    extracted[line.strip("-")] = True
            else:
                # Handle normal key-value pairs
                key_value = line.split(":")
                if len(key_value) == 2:
                    key, value = key_value
                    key = key.strip()
                    value = value.strip()

                    # Convert numeric values
                    if value.replace(".", "", 1).isdigit():
                        value = float(value) if "." in value else int(value)

                    extracted[key] = value

    # Extract specific keys
    return extracted


def aggregate_main_metrics(folder_path, window_size=10):
    """
    Aggregates main metrics from progress CSV files across seeds.
    Computes statistics like overall mean, standard deviation,
    final values, max, min, and median. Saves the aggregated metrics to a text file
    in the folder_path.
    """
    # Define the groups of features and a mapping to human-friendly names.
    train_features = ["env_steps", "return", "invalid_actions_blue", "invalid_actions_red", "length", "FPS", "time", "restorations", "infiltrations"]
    eval_features = ["env_steps_eval", "return_eval", "invalid_actions_blue_eval", "invalid_actions_red_eval", "length_eval", "FPS_eval", "time_eval", "restorations_eval", "infiltrations_eval"]
    stats_features = ["env_steps", "critic_loss", "q", "critic_grad_norm", "critic_seq_grad_norm"]

    feature_mapping = {
        "env_steps": "Environment Steps",
        "return": "Reward",
        "invalid_actions_blue": "Invalid Actions (Blue)",
        "invalid_actions_red": "Invalid Actions (Red)",
        "restorations": "Unnecessary Restorations",
        "infiltrations": "Infiltrations",
        "length": "Episode Length",
        "FPS": "Frames per Second",
        "time": "Training Time",
        "env_steps_eval": "Environment Steps Eval",
        "return_eval": "Reward Eval",
        "invalid_actions_blue_eval": "Invalid Actions (Blue) Eval",
        "invalid_actions_red_eval": "Invalid Actions (Red) Eval",
        "restorations_eval": "Unnecessary Restorations Eval",
        "infiltrations_eval": "Infiltrations Eval",
        "length_eval": "Episode Length Eval",
        "FPS_eval": "Frames per Second Eval",
        "time_eval": "Training Time Eval",
        "critic_loss": "Critic Loss",
        "q": "Q-Value",
        "critic_grad_norm": "Critic Gradient Norm",
        "critic_seq_grad_norm": "Sequential Critic Gradient Norm"
    }

    # Combine all features into one list.
    all_features = list(set(train_features + eval_features + stats_features))
    
    # Find subfolders (each assumed to be a seed folder)
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    # Parse configuration from the first subfolder to create a save name.
    config_file = os.path.join(folder_path, subfolders[0], "flags.txt")
    config_dict = parse_flags(config_file)
    save_name = f'{config_dict["env_type"]}-{config_dict["train_episodes"]}-{config_dict["algo"]}-{config_dict["sequential_model"]}-{config_dict["seeds"]}'

    # Dictionary to store aggregated results for each feature.
    aggregated_results = {}

    # Process each feature across all seeds.
    for feature in all_features:
        # Choose the correct CSV file based on the feature category.
        if feature in train_features:
            file = 'progress_train.csv'
        elif feature in eval_features:
            file = 'progress_eval.csv'
        elif feature in stats_features:
            file = 'progress_stats.csv'
        else:
            continue
        
        # List to collect the feature data (as a smoothed Series) for each seed.
        feature_data_list = []

        for subfolder in subfolders:
            csv_path = os.path.join(folder_path, subfolder, file)
            data_seed = load_csv(csv_path)
            if data_seed is not None and feature in data_seed.columns:
                entries = data_seed[feature]
                # Apply rolling smoothing.
                smoothed = entries.rolling(window=window_size, min_periods=1).mean()
                feature_data_list.append(smoothed)
        
        # Only aggregate if we got data from at least one seed.
        if feature_data_list:
            # Align all series to the maximum length among seeds.
            max_length = max(len(series) for series in feature_data_list)
            aligned_series = [series.reindex(range(max_length), fill_value=np.nan) for series in feature_data_list]
            df_combined = pd.concat(aligned_series, axis=1)
            
            # Compute metrics along the episode axis.
            mean_series = df_combined.mean(axis=1)
            std_series = df_combined.std(axis=1)
            final_values = df_combined.iloc[-1, :]  # Last entry from each seed.
            
            aggregated_results[feature] = {
                "Feature Name": feature_mapping.get(feature, feature),
                "Overall Mean (Avg over Episodes)": np.nanmean(mean_series),
                "Overall Std (Avg over Episodes)": np.nanmean(std_series),
                "Final Mean (Last Episode)": np.nanmean(final_values),
                "Final Std": np.nanstd(final_values),
                "Max Mean": np.nanmax(mean_series),
                "Min Mean": np.nanmin(mean_series),
                "Median Mean": np.nanmedian(mean_series)
            }

    # Build output text from aggregated metrics.
    output_lines = []
    output_lines.append(f"Aggregated Metrics for {save_name}\n")
    for feature, metrics in aggregated_results.items():
        output_lines.append(f"Feature: {metrics['Feature Name']} ({feature})")
        output_lines.append(f"  Overall Mean (Avg over Episodes): {metrics['Overall Mean (Avg over Episodes)']:.4f}")
        output_lines.append(f"  Overall Std (Avg over Episodes): {metrics['Overall Std (Avg over Episodes)']:.4f}")
        output_lines.append(f"  Final Mean (Last Episode): {metrics['Final Mean (Last Episode)']:.4f}")
        output_lines.append(f"  Final Std: {metrics['Final Std']:.4f}")
        output_lines.append(f"  Max Mean: {metrics['Max Mean']:.4f}")
        output_lines.append(f"  Min Mean: {metrics['Min Mean']:.4f}")
        output_lines.append(f"  Median Mean: {metrics['Median Mean']:.4f}")
        output_lines.append("")  # blank line between features

    # Save the aggregated metrics to a text file in the same location as the plots.
    output_path = os.path.join(folder_path, f"{save_name}_aggregated_metrics.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"Aggregated metrics saved to {output_path}")
    return aggregated_results



def plot_feature(folder_path, feature, window_size=10):
    data = []

    train_features = ["env_steps", "return", "invalid_actions_blue", "invalid_actions_red", "length", "FPS", "time", "restorations", "infiltrations"]
    eval_features = ["env_steps_eval", "return_eval", "invalid_actions_blue_eval", "invalid_actions_red_eval", "length_eval", "FPS_eval", "time_eval", "restorations_eval", "infiltrations_eval"]
    stats_features = ["env_steps", "critic_loss", "q", "critic_grad_norm", "critic_seq_grad_norm"]

    feature_mapping = {
    "env_steps": "Environment Steps",
    "return": "Reward",
    "invalid_actions_blue": "Invalid Actions (Blue)",
    "invalid_actions_red": "Invalid Actions (Red)",
    "restorations": "Unnecessary Restorations",
    "infiltrations": "Infiltrations",
    "length": "Episode Length",
    "FPS": "Frames per Second",
    "time": "Training Time",
    "env_steps_eval": "Environment Steps Eval",
    "return_eval": "Reward Eval",
    "invalid_actions_blue_eval": "Invalid Actions (Blue) Eval",
    "invalid_actions_red_eval": "Invalid Actions (Red) Eval",
    "restorations_eval": "Unnecessary Restorations Eval",
    "infiltrations_eval": "Infiltrations Eval",
    "length_eval": "Episode Length Eval",
    "FPS_eval": "Frames per Second Eval",
    "time_eval": "Training Time Eval",
    "critic_loss": "Critic Loss",
    "q": "Q-Value",
    "critic_grad_norm": "Critic Gradient Norm",
    "critic_seq_grad_norm": "Sequential Critic Gradient Norm"
}

    if feature in train_features:
        file = 'progress_train.csv'
    elif feature in eval_features:
        file = 'progress_eval.csv'
    elif feature in stats_features:
        file = 'progress_stats.csv'
    else:
        print(f"Unknown Feature: {feature}")
        return -1

    feature_name = feature_mapping.get(feature, feature)
    print(type(os))
    print(os)
    # Check if the folder contains subfolders (multiple seeds) or directly the files
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    config_dict = parse_flags(os.path.join(os.path.join(folder_path, subfolders[0], "flags.txt")))
    save_name = f'{config_dict["env_type"]}-{config_dict["train_episodes"]}-{config_dict["algo"]}-{config_dict["sequential_model"]}-{config_dict["seeds"]}'

    print(f"Plotting {feature_name}")
    if subfolders:
        for subfolder in subfolders:
            csv_path = os.path.join(folder_path, subfolder, file)
            data_seed = load_csv(csv_path)
            if data_seed is not None:
                data.append(data_seed)
    else:
        csv_path = os.path.join(folder_path, file)
        data_seed = load_csv(csv_path)
        if data_seed is not None:
            data.append(data_seed)

    
    plt.figure(figsize=(12, 7))
    plt.tight_layout()
    
    all_entries = []
    all_smoothed_entries = []
    max_length = max(len(seed_data[feature]) for seed_data in data)

    for seed_data in data:
        entries = seed_data[feature]
        smoothed_entries = entries.rolling(window=window_size, min_periods=1).mean()
        
        all_entries.append(entries.reindex(range(max_length), fill_value=np.nan))
        all_smoothed_entries.append(smoothed_entries.reindex(range(max_length), fill_value=np.nan))
        
        #plt.plot(entries, label=f'{feature_name}', color='lightblue', alpha=0.3)
        #plt.plot(smoothed_entries, label=f'{feature_name} (Smoothed, Window={window_size})', color='blue', alpha=0.6)

    mean_smoothed = pd.concat(all_smoothed_entries, axis=1).mean(axis=1)
    std_smoothed = pd.concat(all_smoothed_entries, axis=1).std(axis=1)

    plt.plot(mean_smoothed, color='#1f77b4', linewidth=2, label=f'Mean {feature_name} (Smoothed)')
    plt.fill_between(range(max_length), mean_smoothed - std_smoothed, mean_smoothed + std_smoothed, color='#1f77b4', alpha=0.3, label='Standard Deviation')

    # Add labels, title, and legend
    plt.xlabel('Episodes')
    plt.ylabel(feature_name)
    plt.title(f'{feature_name} Over Time for {save_name}')
    plt.legend()

    # Lighten the grid and background for better visibility
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.gca().set_facecolor('whitesmoke')
    
    plt.savefig(folder_path + feature_name + "-" + save_name)
    return plt.gcf()


def plot_models(superfolder_path, feature, window_size=10):
    """
    Plots a comparison of a given feature across different models.
    Each model folder in the superfolder should contain seed subfolders
    with the CSV logs.

    Args:
        superfolder_path (str): Path to the superfolder containing model folders.
        feature (str): The feature to plot.
        window_size (int): Smoothing window size for rolling average.
    """
    # Mapping from CSV feature names to human-readable names.
    feature_mapping = {
        "env_steps": "Environment Steps",
        "return": "Reward",
        "invalid_actions_blue": "Invalid Actions (Blue)",
        "invalid_actions_red": "Invalid Actions (Red)",
        "restorations": "Unnecessary Restorations",
        "infiltrations": "Infiltrations",
        "length": "Episode Length",
        "FPS": "Frames per Second",
        "time": "Training Time",
        "env_steps_eval": "Environment Steps Eval",
        "return_eval": "Reward Eval",
        "invalid_actions_blue_eval": "Invalid Actions (Blue) Eval",
        "invalid_actions_red_eval": "Invalid Actions (Red) Eval",
        "restorations_eval": "Unnecessary Restorations Eval",
        "infiltrations_eval": "Infiltrations Eval",
        "length_eval": "Episode Length Eval",
        "FPS_eval": "Frames per Second Eval",
        "time_eval": "Training Time Eval",
        "critic_loss": "Critic Loss",
        "q": "Q-Value",
        "critic_grad_norm": "Critic Gradient Norm",
        "critic_seq_grad_norm": "Sequential Critic Gradient Norm"
    }

    # Determine which CSV file to load based on feature group.
    train_features = ["env_steps", "return", "invalid_actions_blue", "invalid_actions_red",
                      "length", "FPS", "time", "restorations", "infiltrations"]
    eval_features = ["env_steps_eval", "return_eval", "invalid_actions_blue_eval", "invalid_actions_red_eval",
                     "length_eval", "FPS_eval", "time_eval", "restorations_eval", "infiltrations_eval"]
    stats_features = ["env_steps", "critic_loss", "q", "critic_grad_norm", "critic_seq_grad_norm"]

    if feature in train_features:
        csv_file = 'progress_train.csv'
    elif feature in eval_features:
        csv_file = 'progress_eval.csv'
    elif feature in stats_features:
        csv_file = 'progress_stats.csv'
    else:
        print(f"Unknown Feature: {feature}")
        return None

    feature_name = feature_mapping.get(feature, feature)

    # Each model should have its own folder inside the superfolder.
    model_folders = [d for d in os.listdir(superfolder_path)
                     if os.path.isdir(os.path.join(superfolder_path, d))]

    if not model_folders:
        print("No model folders found in the superfolder.")
        return None

    plt.figure(figsize=(12, 7))
    plt.tight_layout()

    # Get a colormap to assign distinct colors to each model.
    colors = plt.cm.tab10.colors

    # Loop over each model folder.
    for i, model_folder in enumerate(sorted(model_folders)):
        model_path = os.path.join(superfolder_path, model_folder)
        # Each model folder should have seed subfolders.
        seed_folders = [d for d in os.listdir(model_path)
                        if os.path.isdir(os.path.join(model_path, d))]

        model_data = []
        for seed_folder in seed_folders:
            csv_path = os.path.join(model_path, seed_folder, csv_file)
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path, comment='#')
                    model_data.append(df)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")

        if len(model_data) == 0:
            print(f"No valid data found for model {model_folder}")
            continue

        all_entries = []
        all_smoothed_entries = []
        max_length = max(len(seed_data[feature]) for seed_data in model_data)

        for seed_data in model_data:
            entries = seed_data[feature]
            smoothed_entries = entries.rolling(window=window_size, min_periods=1).mean()

            all_entries.append(entries.reindex(range(max_length), fill_value=np.nan))
            all_smoothed_entries.append(smoothed_entries.reindex(range(max_length), fill_value=np.nan))

        # Aggregate across seeds for this model.
        mean_smoothed = pd.concat(all_smoothed_entries, axis=1).mean(axis=1)
        std_smoothed = pd.concat(all_smoothed_entries, axis=1).std(axis=1)

        # Plot the aggregated mean and std for the model.
        color = colors[i % len(colors)]
        plt.plot(mean_smoothed, color=color, linewidth=2,
                 label=f'{model_folder} {feature_name}')
        plt.fill_between(range(max_length),
                         mean_smoothed - std_smoothed,
                         np.minimum(mean_smoothed + std_smoothed, 0),  # maximum 0 for miniCAGE
                         color=color, alpha=0.3)

    # Final plot settings.
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel(feature_name, fontsize=14)
    plt.title(f'{feature_name} Over Time Comparison Across Models', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.gca().set_facecolor('whitesmoke')

    save_path = os.path.join(superfolder_path, f'comparison_{feature_name}.png')
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")
    return plt.gcf()


def load_csv(path):
    try:
        return pd.read_csv(path, comment='#')
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None


if __name__ == '__main__':
    folder_path = 'TFPORL-main/pomdp-discrete/logs_results/network-defender/final/'

    plot_models(folder_path, feature='return', window_size=100)
    #plot_feature(folder_path=folder_path, feature='return')
    #aggregate_main_metrics(folder_path=folder_path)


 

