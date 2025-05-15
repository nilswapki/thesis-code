from main_eval import initialize_learner_with_flags
from policies.learner import Learner
import numpy as np
import os
import re


def evaluate(learner: Learner, save_dir: str, episodes: int = 10):

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

    return returns_per_episode


if __name__ == "__main__":
    dir = 'logs/mini-cage/100/mamba/2025-03-24-11:38:38'
    episodes = 200
    all_rewards = []

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
            rewards = evaluate(learner=agent, save_dir=subfolder_path, episodes=episodes)
            all_rewards.append(rewards)

    # save mean and std of mean_rewards in a txt file
    output_lines = []
    output_lines.append(f"Mean reward over all agents: {np.mean([np.mean(rewards) for rewards in all_rewards])}\n")
    output_lines.append(f"Standard Deviation of mean reward: {np.std([np.mean(rewards) for rewards in all_rewards])}\n")
    output_lines.append(f"Maximum Reward: {np.max([np.max(rewards) for rewards in all_rewards])}\n")
    output_lines.append(f"Minimum Reward: {np.min([np.min(rewards) for rewards in all_rewards])}\n")
    filename = f"overall_rewards_adversarial_{episodes}_episodes.txt"
    file_path = os.path.join(dir, filename)
    with open(file_path, 'w') as file:
        file.writelines(output_lines)

    print("\n------------------------*------------------------")
    print(f"Mean reward over all agents: {np.mean([np.mean(rewards) for rewards in all_rewards])}")
    print("------------------------*------------------------")