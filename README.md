# Sequential Neural Architectures for Reinforcement Learning in Cybersecurity

This repository contains the code for the master's thesis project on **Reinforcement Learning (RL)** with **sequential neural architectures** in **partially observable cybersecurity environments**. The focus lies on comparing the architectures MLP, LSTM, LRU, and Mamba in the environments **MiniCAGE** and **NetworkDefender**.

This code is based on the repository from https://github.com/CTP314/TFPORL belonging to the following paper:

```bibtex
@article{Lu2024Rethink,
  title={Rethinking Transformers in Solving POMDPs},
  author={Chenhao Lu and Ruizhe Shi and Yuyao Liu and Kaizhe Hu and Simon S. Du and Huazhe Xu},
  journal={International Conference on Machine Learning}, 
  year={2024}
}
```

---

## 📜 Thesis Overview

- **Title:** Sequential Neural Architectures for Reinforcement Learning in Partially Observable Cybersecurity Scenarios
- **Author:** Nils Wapki
- **Institution:** Technical University of Munich
- **Degree Program:** M.Sc. Data Engineering and Analytics

---

## 📁 Project Structure

This repository is organized as follows:

There are three main python files:
- perform_training.py: Contains the main training loop for the RL agents.
- perform_evaluation.py: Contains the evaluation logic for the trained agents.
- perform_explainability.py: Contains the logic for applying TimeSHAP to explain the learned policies.

Other notable files include:
- plotting.py: Contains functions for plotting training and evaluation results.
- helper_eval.py and helper_eval_red.py: Helper functions for to load an existing agent by creating a Learner object

The directory structure is as follows:
- buffers: Contains buffer classes for storing experiences.
- configs: Contains configuration files for different environments, RL algorithms and architectures.
- envs: Contains the MiniCAGE and NetworkDefender environments.
- final_results: Contains the final results of the training, evaluation, and explainability runs.
- logs: Contains log files for training and evaluation runs. This directory was not added to git
- policies: Contains the actual code for the RL algorithms and sequential architectures.
- torchkit: Contains utility functions for PyTorch.
- utils: Contains utility functions for the project.

---

## Detailed Explanations for Running Specific Tasks

### Training an RL Agent
For training, run the `perform_training.py` script.
A range of parameters can be adjusted at the top of the script. 
Alternatively, they can also be specified via command line arguments.
All training results will be saved in the logs directory.

### Evaluating an RL Agent
For evaluation, run the `perform_evaluation.py` script. 
This will load the trained agent and evaluate its performance in the specified environment.
It is necessary to adjust the environment in helper_eval l.25. 
Unfortunately, this is not configurable from outside due to the absl flags used in the code framework.

Inside the miniCAGE environment code (`envs/mini_CAGE`) you can further specify env the following details:
- for miniCAGE, in `SimplifiedCAGEWrapper.py` you can set the rule-based attackers in l.26, default: all three
- for miniCAGE, in `NetworkDefenderWrapper.py` you can toggle the evaluation against an RL attacker in l.36, default: Off

### Explainability with TimeSHAP
For explainability, run the `perform_explainability.py` script.
The directory that contains the results of one run (one seed) needs to be specified. Furthermore, the normal of 
trajectories and the number of top features and last timesteps needs to be specified. No matter the value of 
top features and last timesteps, all values will be saved for later use. The values are only needed for immediate plotting.
the model and tag parameters are only needed to correctly name the output files.



In case the network-defender agent should be explained from a fixed starting node, this has to be adjusted 
under `envs/network-defender/network-defender.py` l.164

## 🧠 Objectives

- Investigate RL performance in POMDP cybersecurity environments.
- Compare sequential and non-sequential architectures (MLP, LSTM, LRU, Mamba).
- Apply model-agnostic explainability (TimeSHAP) to interpret learned policies.

---

## ⚙️ Environments

### MiniCAGE
- Based on the CybORG Challenge.
- Agent defends a simulated IT infrastructure against various attackers.
- Partially observable: observations consist of noisy sensor readings.

### NetworkDefender (Custom)
- Random, connected network topologies.
- Adjustable sensor noise and infiltration behavior.
- Emphasizes sequential decision-making under uncertainty.

---

## 🏗️ Architectures

The following architectures are implemented and evaluated:
- `MLP`: Baseline feedforward model
- `LSTM`: Recurrent neural network with memory
- `LRU`: Linear recurrent unit for efficient long-range dependencies
- `Mamba`: State-space model with selective scan for high performance in sequence modeling

---

## Requirements

The required packages are listed in the `requirements_mac.txt` file for macOS users 
and `requirements_nvidia.txt` for Linux NVIDIA GPU users.


## Questions and Contact

For questions, reach out to:

Nils Wapki
n.wapki@freenet.de