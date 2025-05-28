"""
Helper script to load a second existing agent (for adversarial evaluation)
with flags from a pkl file and initialize the learner with those flags.
"""

import os
import pickle
from absl import flags
from ml_collections import config_flags
from policies.learner import Learner
from envs.make_env import make_env
from torchkit.pytorch_utils import set_gpu_mode
import torch
import sys
from utils import system

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config_env_red",
    "configs/envs/mini-cage-red.py",
    "File path to the environment configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_rl_red",
    "configs/rl/dqn_default.py",
    "File path to the RL algorithm configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_seq_red",
    "configs/seq_models/lstm_default.py",
    "File path to the seq model configuration.",
    lock_config=False,
)

flags.mark_flags_as_required(["config_rl_red", "config_env_red"])

# shared encoder settings
flags.DEFINE_boolean("shared_encoder_red", False, "share encoder in actor-critic or not")
flags.DEFINE_boolean(
    "freeze_critic_red", False, "in shared encoder, freeze critic params in actor loss"
)

# training settings
flags.DEFINE_integer("seed_red", 1, "Random seed.")
flags.DEFINE_integer("batch_size_red", 64, "Mini batch size.")
flags.DEFINE_integer("train_episodes_red", 100, "Number of episodes during training.")
flags.DEFINE_float("updates_per_step_red", 0.5, "Gradient updates per step.")
flags.DEFINE_integer("start_training_red", 0, "Number of episodes to start training.")

# logging settings
flags.DEFINE_boolean("debug_red", False, "debug mode")
flags.DEFINE_string("save_dir_red", "logs", "logging dir.")
flags.DEFINE_string("submit_time_red", None, "used in sbatch")
flags.DEFINE_string("run_name_red", None, "used in sbatch")


# Load flags from pkl
def load_flags_from_pkl(pkl_path):
    """Load flags from a pickled file and set them in FLAGS."""
    with open(pkl_path, "rb") as f:
        flag_values = pickle.load(f)
        for key, value in flag_values.items():
            # Set the required flags only
            if key == "config_env":
                FLAGS.config_env_red.env_name = value.env_name
                FLAGS.config_env_red.env_type = value.env_type
            elif key == "config_rl":
                FLAGS.config_rl_red = value
            elif key == "config_seq":
                FLAGS.config_seq_red = value
            elif key == "shared_encoder":
                FLAGS.shared_encoder_red = value
            elif key == "freeze_critic":
                FLAGS.freeze_critic_red = value
            elif key == "start_training":
                FLAGS.start_training_red = value
            elif key == "seed":
                FLAGS.seed_red = value
            elif key == "train_episodes":
                FLAGS.train_episodes_red = value
            elif key == "updates_per_step":
                FLAGS.updates_per_step_red = value
            elif key == "batch_size":
                FLAGS.batch_size_red = value
            elif key == "save_dir":
                FLAGS.save_dir_red = value


def initialize_learner_with_flags(save_dir=None):
    """Manually initializes FLAGS and returns the learner."""
    if not FLAGS.is_parsed():  # Ensure flags are parsed
        FLAGS(sys.argv)

    if save_dir is None:
        raise ValueError('Please specify a directory')

    # Load the flags
    load_flags_from_pkl(os.path.join(save_dir, "flags.pkl"))

    gpu_mode_bool = ((torch.cuda.is_available() or torch.backends.mps.is_available())
                     and not FLAGS.config_seq_red.model.seq_model_config.name == 'mlp')
    set_gpu_mode(mode=gpu_mode_bool)

    # Load the environment and the Learner instance
    config_env_red = FLAGS.config_env_red
    config_rl_red = FLAGS.config_rl_red
    config_seq_red = FLAGS.config_seq_red

    config_env_red, env_name = config_env_red.create_fn(config_env_red)
    env = make_env(env_name, FLAGS.seed_red)
    eval_env = make_env(env_name, FLAGS.seed_red + 42, eval=True)
    system.reproduce(FLAGS.seed_red)

    # Create the Learner instance and load the trained model
    learner = Learner(env, eval_env, FLAGS, config_rl_red, config_seq_red, config_env_red)

    save_path = os.path.join(save_dir, "save")
    agent_files = [f for f in os.listdir(save_path)]
    if agent_files:
        file_paths = [os.path.join(save_path, f) for f in agent_files]
        model_path = max(file_paths, key=os.path.getctime)
    else:
        raise ValueError("No valid agent_*.pt files found in the save directory.")

    learner.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")

    return learner


if __name__ == "__main__":
    initialize_learner_with_flags('x')
