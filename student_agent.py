import gym
import torch
import numpy as np
import torch.nn as nn
import cv2


class DuelingCNN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(3136, 512)
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, 12)  # Use 12 if COMPLEX_MOVEMENT (used in training) has 12 actions

    def forward(self, x):
        x = x / 255.0  # normalize if input is uint8 [0,255]
        x = self.extractor(x)
        x = torch.relu(self.fc(x))
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=1, keepdim=True)


class Agent:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)  # Match training env action size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DuelingCNN(in_channels=4, num_actions=12).to(self.device)
        self.model.load_state_dict(torch.load("checkpoint_step250000.pth", map_location=self.device))
        self.model.eval()
        self.frame_stack = []

    def act(self, observation):
        # observation shape: (240, 256, 3) — RGB frame
        obs_gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        obs_resized = cv2.resize(obs_gray, (84, 84))
        obs_normalized = obs_resized.astype(np.float32) / 255.0

        if len(self.frame_stack) < 4:
            self.frame_stack = [obs_normalized] * 4
        else:
            self.frame_stack.pop(0)
            self.frame_stack.append(obs_normalized)

        state = np.stack(self.frame_stack, axis=0)  # shape: (4, 84, 84)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 4, 84, 84)

        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values, dim=1).item()
