from helper_eval import initialize_learner_with_flags
from policies.learner import Learner
import numpy as np
import os
import re
import time


def evaluate(learner: Learner, save_dir: str, episodes: int = 10):
    """
        Evaluates the performance of a given learner over a specified number of episodes.

        Args:
            learner (Learner): The learner object to be evaluated.
            save_dir (str): Directory where evaluation results will be saved.
            episodes (int, optional): Number of episodes to evaluate. Defaults to 10.

        Returns:
            tuple: A tuple containing:
                - returns_per_episode (list): List of returns for each episode.
                - infos (dict): Dictionary containing additional evaluation information.
        """

    returns_per_episode, success_rate, total_steps, trajs, infos = learner.evaluate(episodes=episodes)
    print('Evaluation Completed')

    # Calculate mean and std of returns
    mean_return = np.mean(returns_per_episode)
    std_return = np.std(returns_per_episode)

    # Prepare the data to save
    output_lines = []
    output_lines.append(f"Evaluation Results - {episodes} Episodes\n")
    output_lines.append(f"Mean Return: {mean_return}\n")
    output_lines.append(f"Standard Deviation of Return: {std_return}\n")
    output_lines.append(f"Maximum Return: {np.max(returns_per_episode)}\n")
    output_lines.append(f"Minimum Return: {np.min(returns_per_episode)}\n")
    output_lines.append("Info Keywords:\n")
    for key, value in infos.items():
        output_lines.append(f"- {key}: {value}\n")

    # Construct the file path with the number of episodes
    filename = f"evaluation_results_{episodes}_episodes.txt"
    file_path = os.path.join(save_dir, filename)

    # Save the results to the txt file
    with open(file_path, 'w') as file:
        file.writelines(output_lines)

    print("------------------------*------------------------")
    for line in output_lines:
        print(line, end='')
    print(f"Results saved to {file_path}")
    print('\n')

    return returns_per_episode, infos


def evaluation_pipe(dir, episodes, tag):
    """
    Executes an evaluation pipeline for agents stored in subdirectories of a given directory.

    Args:
        dir (str): The directory containing subfolders with saved agent models.
        episodes (int): The number of episodes to evaluate each agent.
        tag (str): A tag to differentiate evaluation runs, used in the output file name.

    Raises:
        ValueError: If `dir` is not a valid directory.
        ValueError: If `episodes` is not a positive integer.
        ValueError: If `tag` is not a non-empty string.

    Returns:
        None
    """

    # check if dir is a valid directory
    if not os.path.isdir(dir):
        raise ValueError(f"The directory {dir} does not exist or is not a valid directory.")

    # check if episodes is a valid integer
    if not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("Episodes must be a positive integer.")

    # check if tag is a valid string
    if not isinstance(tag, str) or not tag:
        raise ValueError("Tag must be a non-empty string.")

    # Initialize lists to store results
    all_rewards = []
    all_infiltrations = []
    all_restorations = []
    all_times = []

    agent = None

    # Iterate over all subfolders in save_dir
    for subfolder in os.listdir(dir):
        subfolder_path = os.path.join(dir, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Evaluating agent in subfolder: {subfolder}")
            if agent is None:
                agent = initialize_learner_with_flags(save_dir=subfolder_path)
            else:
                save_path = os.path.join(subfolder_path, "save")
                agent_files = [f for f in os.listdir(save_path)]
                if agent_files:
                    file_paths = [os.path.join(save_path, f) for f in agent_files]
                    model_path = max(file_paths, key=os.path.getctime)
                else: raise ValueError("No valid agent_*.pt files found in the save directory.")
                agent.load_model(model_path)
            start_time = time.time()
            rewards, infos = evaluate(learner=agent, save_dir=subfolder_path, episodes=episodes)
            end_time = time.time()
            all_rewards.append(rewards)
            all_times.append(end_time - start_time)
            if agent.train_env.name == "network-defender":
                all_restorations.append(infos['restorations_eval'])
                all_infiltrations.append(infos['infiltrations_eval'])

    # save mean and std of mean_rewards in a txt file
    output_lines = []
    output_lines.append(f"Mean reward over all agents: {np.mean([np.mean(rewards) for rewards in all_rewards])}\n")
    output_lines.append(f"Standard Deviation of mean reward: {np.std([np.mean(rewards) for rewards in all_rewards])}\n")
    output_lines.append(f"Maximum Reward: {np.max([np.max(rewards) for rewards in all_rewards])}\n")
    output_lines.append(f"Minimum Reward: {np.min([np.min(rewards) for rewards in all_rewards])}\n")
    output_lines.append("\n")
    if agent.train_env.name == "network-defender":
        output_lines.append(f"Mean Infiltrations: {np.mean(all_infiltrations)}\n")
        output_lines.append(f"Standard Deviation of Infiltrations: {np.std(all_infiltrations)}\n")
        output_lines.append(f"Maximum Infiltrations: {np.max(all_infiltrations)}\n")
        output_lines.append(f"Minimum Infiltrations: {np.min(all_infiltrations)}\n")
        output_lines.append("\n")
        output_lines.append(f"Mean Restorations: {np.mean(all_restorations)}\n")
        output_lines.append(f"Standard Deviation of Restorations: {np.std(all_restorations)}\n")
        output_lines.append(f"Maximum Restorations: {np.max(all_restorations)}\n")
        output_lines.append(f"Minimum Restorations: {np.min(all_restorations)}\n")
        output_lines.append("\n")
    output_lines.append(f"Time taken for 100 eval episodes: {np.round(np.mean(all_times)/episodes*100, 2)} seconds\n")

    filename = f"eval_{tag}_{episodes}_episodes.txt"
    file_path = os.path.join(dir, filename)
    with open(file_path, 'w') as file:
        file.writelines(output_lines)

    print("\n------------------------*------------------------")
    print(f"Mean reward over all agents: {np.mean([np.mean(rewards) for rewards in all_rewards])}")
    print("------------------------*------------------------")


if __name__ == '__main__':
    dir = 'final_results/mini-cage/final/mlp'  # Folder than contains the subfolders with the seeds
    episodes = 200  # Number of episodes to evaluate each seed
    tag = 'v1'  # Tag for the evaluation, can be used to differentiate between different evaluation runs

    evaluation_pipe(dir=dir, episodes=episodes, tag=tag)