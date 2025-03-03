import os
import pickle
from absl import app, flags
from ml_collections import config_flags
from policies.learner import Learner
from envs.make_env import make_env

import torch
from torchkit import pytorch_utils as ptu
from utils import helpers as utl
import shap
import timeshap
from timeshap.explainer.local_methods import local_report
from timeshap.utils import calc_avg_event
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Use the standard interactive backend
import seaborn as sns
import pandas as pd

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config_env",
    "configs/envs/mini-cage-red.py",
    "File path to the environment configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_rl",
    "configs/rl/dqn_default.py",
    "File path to the RL algorithm configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_seq",
    "configs/seq_models/mlp_default.py",
    "File path to the seq model configuration.",
    lock_config=False,
)

flags.mark_flags_as_required(["config_rl", "config_env"])

# shared encoder settings
flags.DEFINE_boolean("shared_encoder", False, "share encoder in actor-critic or not")
flags.DEFINE_boolean(
    "freeze_critic", False, "in shared encoder, freeze critic params in actor loss"
)

# training settings
flags.DEFINE_integer("seed", 1, "Random seed.")
flags.DEFINE_integer("batch_size", 64, "Mini batch size.")
flags.DEFINE_integer("train_episodes", 100, "Number of episodes during training.")
flags.DEFINE_float("updates_per_step", 0.5, "Gradient updates per step.")
flags.DEFINE_integer("start_training", 0, "Number of episodes to start training.")

# logging settings
flags.DEFINE_boolean("debug", False, "debug mode")
flags.DEFINE_string("save_dir", "logs", "logging dir.")
flags.DEFINE_string("submit_time", None, "used in sbatch")
flags.DEFINE_string("run_name", None, "used in sbatch")


# Load flags from pkl

def load_flags_from_pkl(pkl_path):
    """Load flags from a pickled file and set them in FLAGS."""
    with open(pkl_path, "rb") as f:
        flag_values = pickle.load(f)
        for key, value in flag_values.items():
            # Set the required flags only
            if key == "config_env":
                FLAGS.config_env.env_name = value.env_name
                FLAGS.config_env.env_type = value.env_type
            elif key == "config_rl":
                FLAGS.config_rl = value
            elif key == "config_seq":
                FLAGS.config_seq = value
            elif key == "shared_encoder":
                FLAGS.shared_encoder = value
            elif key == "freeze_critic":
                FLAGS.freeze_critic = value
            elif key == "start_training":
                FLAGS.start_training = value
            elif key == "seed":
                FLAGS.seed = value
            elif key == "train_episodes":
                FLAGS.train_episodes = value
            elif key == "updates_per_step":
                FLAGS.updates_per_step = value
            elif key == "batch_size":
                FLAGS.batch_size = value
            elif key == "save_dir":
                FLAGS.save_dir = value


def main(argv):
    # Set the save directory where the model, logs, and flags are saved
    save_dir = 'logs/mini-cage-red/100/mlp/2025-02-06-10:27:20+5484-5484'

    # Load the flags
    load_flags_from_pkl(os.path.join(save_dir, "flags.pkl"))

    # Load the environment and the Learner instance
    config_env = FLAGS.config_env
    config_rl = FLAGS.config_rl
    config_seq = FLAGS.config_seq

    config_env, env_name = config_env.create_fn(config_env)
    env = make_env(env_name, FLAGS.seed)
    eval_env = make_env(env_name, FLAGS.seed + 42, eval=True)

    # Create the Learner instance and load the trained model
    learner = Learner(env, eval_env, FLAGS, config_rl, config_seq, config_env)

    # Load the model
    model_path = max([os.path.join(save_dir, 'save/', f) for f in os.listdir(os.path.join(save_dir, "save"))
                      if f.startswith("agent_") and f.endswith(".pt")], key=lambda x: int(x.split('_')[1].split('.')[0]))
    learner.load_model(model_path)
    print("Model loaded successfully.")

    #evaluate(learner)
    explain(learner)


def evaluate(learner):
    # Evaluate the model (or perform inference)
    #learner.evaluate()

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


def explain(learner):
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

if __name__ == "__main__":
    app.run(main)
