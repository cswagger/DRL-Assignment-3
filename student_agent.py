import gym
import torch
import numpy as np
import torch.nn as nn

# Define the same Dueling Q Network architecture used in training
class DuelingQNet(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU()
        )
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x / 255.0
        x = self.feature(x)
        x = self.fc(x)
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=1, keepdim=True)



class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DuelingQNet(in_channels=4, n_actions=5).to(self.device)
        self.model.load_state_dict(torch.load("rainbow_mario_final.pth", map_location=self.device))
        self.model.eval()


    def act(self, observation):
        state = observation
        if state.ndim == 4 and state.shape[-1] == 1:
            state = state.squeeze(-1)
        state = torch.FloatTensor(state.copy()).unsqueeze(0).to(self.device)  # Safe conversion
        with torch.no_grad():
            q_values = self.model(state)
        action = torch.argmax(q_values, dim=1).item()
        return action
