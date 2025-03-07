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


def load_csv(path):
    try:
        return pd.read_csv(path, comment='#')
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None


if __name__ == '__main__':
    folder_path = 'TFPORL-main/pomdp-discrete/logs/mini-cage/100/lstm/2025-03-07-14:46:59'

    plot_feature(folder_path=folder_path, feature='return')


 

