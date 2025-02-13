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


if __name__ == '__main__':
    base_path = 'TFPORL-main/pomdp-discrete/logs/network-defender/100/mlp/'
    file_path = '2025-02-11-15:20:22'

    data = pd.read_csv(base_path + file_path + '/progress_train.csv', comment='#')

    reward_fig = plot_reward(data, window_size=50)

    # Save the figure and show
    flags_path = base_path + file_path + '/flags.txt'
    with open(flags_path, 'r') as f:
        flags_content = f.read()

        # Extract n_nodes, extra_edge_prob, and num_critical_nodes from flags.txt
        import re
        n_nodes = re.search(r'n_nodes:\s*(\d+)', flags_content).group(1)
        extra_edge_prob = re.search(r'extra_edge_prob:\s*([\d.]+)', flags_content).group(1)
        num_critical_nodes = re.search(r'num_critical_nodes:\s*(\d+)', flags_content).group(1)

        file_info = f"{file_path.replace('/', '_').replace(':', '-')}_n{n_nodes}_e{extra_edge_prob}_c{num_critical_nodes}_{base_path.split('/')[-2]}"
        plt.savefig(f'z_plots/reward_plot_smoothing_{file_info}.png')


    #plot_invalid_share(data, window_size=50)

