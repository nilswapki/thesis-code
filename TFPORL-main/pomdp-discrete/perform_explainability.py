from main_eval import initialize_learner_with_flags
from policies.learner import Learner

import shap
import timeshap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from timeshap.explainer.local_methods import local_report
from timeshap.utils import calc_avg_event

matplotlib.use('TkAgg')  # Use the standard interactive backend


def explain(learner: Learner):
    # Evaluate the model
    learner.agent.eval()  # Set the agent to evaluation mode

    # Reset the environment
    action, reward, internal_state = learner.agent.get_initial_info(learner.config_seq.sampled_seq_len)
    obs, _ = learner.eval_env.reset()
    obs = ptu.from_numpy(obs)
    trajectory = []  # Store trajectory for TimeSHAP in numpy array

    done = False
    while not done:
        # Store the current observation for TimeSHAP
        trajectory.append(obs.clone().detach())

        # Select action
        action, internal_state = learner.agent.act(
            prev_internal_state=internal_state,
            prev_action=action,
            reward=reward,
            obs=obs,
            deterministic=True,
        )

        action_index = torch.argmax(action, dim=-1).item()
        print(f"Action: {action_index}, Reward: {reward}, Done: {done}")

        # Step the environment
        next_obs, reward, done, info = utl.env_step(learner.eval_env, action.squeeze(dim=0))
        obs = next_obs.clone()

    # Convert trajectory to numpy array
    trajectory = np.stack([obs.numpy() for obs in trajectory])

    # === TimeSHAP Explainability === #
    print("\nCalculating TimeSHAP explanations...")

    # Define the model wrapper that works with TimeSHAP
    def model_wrapper(obs_batch):
        with torch.no_grad():
            actions = []
            action, reward, internal_state = learner.agent.get_initial_info(learner.config_seq.sampled_seq_len)

            # Convert obs_batch to tensor (ensure it's the correct shape)
            obs_batch = torch.tensor(obs_batch, dtype=torch.float32)

            # Iterate over batch dimension (#samples)
            for sample in range(obs_batch.shape[0]):
                internal_state = learner.agent.get_initial_info(learner.config_seq.sampled_seq_len)[
                    2]  # Reset internal state for each sample
                sample_actions = []

                # Iterate over sequence length
                for i in range(obs_batch.shape[1]):
                    obs_adjusted = obs_batch[sample, i, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, #features)
                    action, internal_state = learner.agent.act(
                        prev_internal_state=internal_state,  # shape (1,1,1,64)
                        prev_action=action,  # shape (1,1,1,56)
                        reward=reward,  # shape (1,1)
                        obs=obs_adjusted,  # shape (1,1,40)
                        deterministic=True,
                    )

                    # Select last time step from the sequence (T dimension)
                    last_action = action[:, :, -1, :]  # Shape: (1, 1, 56)

                    # Squeeze out batch and sample dimensions -> (56,)
                    last_action = last_action.squeeze(0).squeeze(0)
                    sample_actions.append(torch.argmax(last_action, dim=-1).item())

                # Append the final action of each sample
                actions.append(sample_actions[-1])

            return np.array(actions).reshape(-1, 1)  # Ensure shape (#samples, 1)

    model_features = [i for i in range(obs.shape[0])]
    # The plotting dictionary should map features to themselves if there are no custom labels
    plot_features = {f: f for f in model_features}
    avg_event = calc_avg_event(data=pd.DataFrame(trajectory.squeeze(2)), numerical_feats=model_features, categorical_feats=[])
    # Prepare TimeSHAP input dictionaries (these would need to be tailored to your environment and model)
    pruning_dict = {'tol': 0.025}
    event_dict = {'rs': 42, 'nsamples': 100}
    feature_dict = {'rs': 42, 'nsamples': 100, 'feature_names': model_features, 'plot_features': plot_features}
    cell_dict = {'rs': 42, 'nsamples': 100, 'top_x_feats': 2, 'top_x_events': 2}

    data = np.transpose(trajectory, (2, 0, 1))

    # Generate local report and plot using TimeSHAP
    plot = local_report(
        f=model_wrapper,
        data=data,
        pruning_dict=pruning_dict,
        event_dict=event_dict,
        feature_dict=feature_dict,
        cell_dict=cell_dict,
        baseline=avg_event
    )

    # Show the plot
    plot.show()


if __name__ == 'main':
    agent = initialize_learner_with_flags(save_dir='logs_results/mini-cage/final/standard/lru/seed-1/save/agent_0010010_perf-0.083.pt')

    explain(agent)