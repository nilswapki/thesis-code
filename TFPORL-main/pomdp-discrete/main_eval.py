import os
import pickle
import torch
from absl import app, flags
from ml_collections import config_flags
from policies.learner import Learner
from envs.make_env import make_env

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config_env",
    "configs/envs/mini-cage.py",
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
    save_dir = 'logs/mini-cage/100/mlp/2025-03-01-12:17:52/'

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
    model_path = os.path.join(save_dir, "save/agent_00500_perf0.000.pt")
    learner.load_model(model_path)
    print("Model loaded successfully.")

    # Evaluate the model (or perform inference)
    learner.evaluate()


if __name__ == "__main__":
    app.run(main)
