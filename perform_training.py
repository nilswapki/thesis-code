import os, time

t0 = time.time()
pid = str(os.getpid())
if "SLURM_JOB_ID" in os.environ:
    jobid = str(os.environ["SLURM_JOB_ID"])
else:
    jobid = pid

import numpy as np
import torch
from absl import app, flags
from ml_collections import config_flags
import pickle
from utils import system, logger

from torchkit.pytorch_utils import set_gpu_mode
from policies.learner import Learner
from envs.make_env import make_env
from plotting import plot_feature, aggregate_main_metrics

mig_devices = [
    "MIG-e5bb0672-753a-5707-b8e3-afbf24f6375a",
    "MIG-d221f560-4f89-5ae7-ba27-719dcf6f0bfb",
    "MIG-40e6fded-589b-59d4-9a87-f12feff6b50b",
    "MIG-b2ca0a8b-c8bc-5072-bbd3-ed23965d4c7b"
]  # export CUDA_VISIBLE_DEVICES=MIG-d221f560-4f89-5ae7-ba27-719dcf6f0bfb

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config_env",
    "configs/envs/network-defender.py",
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
    "configs/seq_models/lstm_default.py",
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
flags.DEFINE_list("seeds", [1], "Random seed.")
flags.DEFINE_integer("batch_size", 64, "Mini batch size.")
flags.DEFINE_integer("train_episodes", 12, "Number of episodes during training.")
flags.DEFINE_float("updates_per_step", 0.5, "Gradient updates per step.")
flags.DEFINE_integer("start_training", 0, "Number of episodes to start training.")

# logging settings
flags.DEFINE_boolean("debug", False, "debug mode")
flags.DEFINE_string("save_dir", "logs", "logging dir.")
flags.DEFINE_string("submit_time", None, "used in sbatch")
flags.DEFINE_string("run_name", None, "used in sbatch")


def main(argv):

    config_env = FLAGS.config_env
    config_rl = FLAGS.config_rl
    config_seq = FLAGS.config_seq
    config_env, env_name = config_env.create_fn(config_env)

    config_seq, _ = config_seq.name_fn(config_seq, config_env.env_name)
    max_training_steps = int(FLAGS.train_episodes * config_env.env_name)
    config_rl, _ = config_rl.name_fn(
        config_rl, config_env.env_name, max_training_steps
    )

    gpu_mode_bool = (torch.cuda.is_available() or torch.backends.mps.is_available())
    set_gpu_mode(mode=gpu_mode_bool)

    uid = f"{system.now_str()}"

    for seed in FLAGS.seeds:  # Loop over the list of seeds

        env = make_env(env_name, seed)
        eval_env = make_env(env_name, seed + 42, eval=True)

        system.reproduce(seed)
        torch.set_num_threads(1)
        np.set_printoptions(precision=3, suppress=True)
        torch.set_printoptions(precision=3, sci_mode=False)


        run_name = f"{config_env.env_type}/{config_env.env_name}/{config_seq.model.seq_model_config.name}/{uid}/seed-{seed}/"

        FLAGS.run_name = uid

        format_strs = ["csv"]
        if FLAGS.debug:
            FLAGS.save_dir = "debug"
            format_strs.extend(["stdout", "log"])  # logger.log

        log_path = os.path.join(FLAGS.save_dir, run_name)
        FLAGS.log_dir = log_path
        logger.configure(dir=log_path, format_strs=format_strs)

        # write flags to a txt
        key_flags = FLAGS.get_key_flags_for_module(argv[0])
        with open(os.path.join(log_path, "flags.txt"), "w") as text_file:
            text_file.write("\n".join(f.serialize() for f in key_flags) + "\n")
            text_file.write(f"\n--sequential_model={config_seq.model.seq_model_config.name}")
        # write flags to pkl
        with open(os.path.join(log_path, "flags.pkl"), "wb") as f:
            pickle.dump(FLAGS.flag_values_dict(), f)

        # start training
        learner = Learner(env, eval_env, FLAGS, config_rl, config_seq, config_env)
        learner.train()

    # after training, save the learner
    if FLAGS.train_episodes > config_env.eval_interval:  # if eval was done
        features = ['return', 'return_eval', 'critic_loss']  # plot return and critic loss
    else:
        features = ['return', 'critic_loss']
    for feature in features:  # plot each feature
        plot_feature(folder_path=f"logs/{config_env.env_type}/{config_env.env_name}/{config_seq.model.seq_model_config.name}/{uid}/", feature=feature)

    # aggregate main metrics
    aggregate_main_metrics(folder_path=f"logs/{config_env.env_type}/{config_env.env_name}/{config_seq.model.seq_model_config.name}/{uid}/")


if __name__ == "__main__":

    app.run(main)
