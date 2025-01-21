import gym
import numpy as np

# Create the Pendulum environment
env = gym.make('Pendulum-v1', render_mode="human")  # Set render_mode="human" to visualize

# Set the seed for reproducibility
env.seed(42)

# Number of episodes
num_episodes = 5

for episode in range(num_episodes):
    # Reset the environment at the start of each episode
    state = env.reset()
    print(f"Episode {episode + 1} started. Initial state: {state}")

    # Run one episode
    done = False
    while not done:
        # Take a random action (from the action space)
        action = env.action_space.sample()

        # Step through the environment
        next_state, reward, terminated, info = env.step(action)

        # Display information
        print(f"Action: {action}, Reward: {reward:.2f}, Next State: {next_state}")

        # Termination condition
        if terminated:
            print("Episode finished.")
            done = True

# Close the environment after all episodes
env.close()
