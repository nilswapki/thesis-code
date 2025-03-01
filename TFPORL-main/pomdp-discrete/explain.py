import torch

from envs.mini_CAGE.SimplifiedCAGEWrapper import SimplifiedCAGEWrapper

if __name__ == '__main__':
    env = SimplifiedCAGEWrapper()
    env.reset()

    # If your file contains the model directly
    model = torch.load('logs/mini-cage/100/mlp/2025-03-01-12:17:52/save/agent_00500_perf0.000.pt')
    model.eval()  # Important! Set the model to evaluation mode

    hidden = model.init_hidden(1)  # Batch size 1

    done = False
    obs = env.reset()

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, hidden = model(obs_tensor, hidden)  # Recurrent model returns (action, hidden)
        action = action.detach().numpy()[0]
        obs, reward, done, trunc, info = env.step(action)

