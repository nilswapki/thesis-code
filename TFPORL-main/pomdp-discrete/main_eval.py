import os
import pickle
from absl import app, flags
from ml_collections import config_flags
from policies.learner import Learner
from envs.make_env import make_env
import re
from torchkit.pytorch_utils import set_gpu_mode
import torch
import sys
from utils import system

if torch.cuda.is_available():  # if running on work computer
    os.chdir('/mnt/thesis-code/TFPORL-main/pomdp-discrete')
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config_env",
    "configs/envs/network-defender.py",  # Change env here!
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
        print("Cursor before load:", f.tell())
        #first_bytes = f.read(1000)
        print("Cursor before load:", f.tell())
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


def initialize_learner_with_flags(save_dir=None):

    """Manually initializes FLAGS and returns the learner."""
    if not FLAGS.is_parsed():  # Ensure flags are parsed
        FLAGS(sys.argv)

    if save_dir is None:
        raise ValueError('Please specify a directory')

    # Load the flags
    pkl_path = 'logs_results/mini-cage/final/standard/lstm/seed-1/flags.pkl'  #logs_results/mini-cage/final/preliminary/mamba/seed-1/flags.pkl'
    load_flags_from_pkl(os.path.join(save_dir, "flags.pkl"))

    gpu_mode_bool = ((torch.cuda.is_available() or torch.backends.mps.is_available())
                and not FLAGS.config_seq.model.seq_model_config.name == 'mlpx')
    set_gpu_mode(mode=gpu_mode_bool)

    # Load the environment and the Learner instance
    config_env = FLAGS.config_env
    config_rl = FLAGS.config_rl
    config_seq = FLAGS.config_seq

    config_env, env_name = config_env.create_fn(config_env)
    #env = make_env(env_name, FLAGS.seed)
    eval_env = make_env(env_name, FLAGS.seed + 42, eval=True)
    system.reproduce(FLAGS.seed)

    # Create the Learner instance and load the trained model
    learner = Learner(eval_env, eval_env, FLAGS, config_rl, config_seq, config_env)

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
    initialize_learner_with_flags('logs_results/network-defender/final/mamba/seed-1')
