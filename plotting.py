import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import pandas as pd
import io


def plot_reward(data, window_size=10):
    """
    Plots the reward over episodes with a sliding window for smoothing.

    Args:
        data (pd.DataFrame): Data containing the reward column 'r'.
        window_size (int): Size of the sliding window for smoothing.
    """
    # Extract the reward column (r) from the dataframe
    rewards = data['r']

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

    # Save the figure and show
    plt.savefig('reward_plot_smoothing.png')
    plt.show()


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


if __name__ == '__main__':
    data = pd.read_csv('results/run-2025-01-17_14-45-17.monitor.csv', comment='#')

    plot_reward(data, window_size=50)
    plot_invalid_share(data, window_size=50)

