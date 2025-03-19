from main_eval import initialize_learner_with_flags
from policies.learner import Learner
import os

os.chdir('/mnt/thesis-code/TFPORL-main/pomdp-discrete')


def evaluate(learner: Learner):
    # Evaluate the model (or perform inference)
    returns_per_episode, success_rate, total_steps, trajs, infos = learner.evaluate(episodes=10)
    print('Evaluation Completed')
    print(returns_per_episode)

    """
    # Perform inference
    learner.agent.eval()  # Set the agent to evaluation mode

    # Reset the environment to get the initial observation
    action, reward, internal_state = learner.agent.get_initial_info(learner.config_seq.sampled_seq_len)
    obs, _ = learner.eval_env.reset()
    obs = ptu.from_numpy(obs)

    done = False
    while not done:
        # Select an action based on the current observation
        action, internal_state = learner.agent.act(
            prev_internal_state=internal_state,
            prev_action=action,
            reward=reward,
            obs=obs,
            deterministic=True,
        )

        # Step the environment with the selected action
        next_obs, reward, done, info = utl.env_step(learner.eval_env, action.squeeze(dim=0))

        action_index = torch.argmax(action, dim=-1).item()
        # Print the results
        print(f"Action: {action_index}, Reward: {reward}, Done: {done}")

        # Update the observation
        obs = next_obs.clone()
    """

if __name__ == "__main__":
    agent = initialize_learner_with_flags(save_dir='logs_results/mini-cage/final/standard/lru/seed-1')
    print('Loaded Agent')

    evaluate(agent)