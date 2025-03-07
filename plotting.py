import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import pandas as pd
import io
import re
import pickle
import os
from ml_collections import config_flags
from absl import app, flags

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


if __name__ == '__main__':
    base_path = 'TFPORL-main/pomdp-discrete/logs/network-defender/100/mlp/'
    file_path = '2025-03-06-14:00:36'

    # Load the pkl file
    config_dict = parse_flags(os.path.join(base_path + file_path, "flags.txt"))

    save_name = f'{config_dict["env_type"]}-{config_dict["train_episodes"]}-{config_dict["algo"]}-{config_dict["seeds"]}'

    data = pd.read_csv(base_path + file_path + '/progress_train.csv', comment='#')

    reward_fig = plot_reward(data, window_size=50)


    plt.savefig('z_plots/' + save_name + '.png')


    #plot_invalid_share(data, window_size=50)

