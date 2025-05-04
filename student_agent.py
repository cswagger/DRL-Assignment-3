import gym
import torch
import numpy as np
import torch.nn as nn
import cv2
from collections import deque


class DuelingQNet(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(3136, 512)
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, n_actions)

    def forward(self, x):
        x = self.extractor(x)
        x = nn.functional.relu(self.fc(x))
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=1, keepdim=True)


class Agent:
    def __init__(self):
        from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
        self.n_actions = len(COMPLEX_MOVEMENT)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DuelingQNet(in_channels=4, n_actions=self.n_actions).to(self.device)
        self.model.load_state_dict(torch.load("checkpoint_step250000.pth", map_location=self.device))
        self.model.eval()
        self.frame_stack = deque(maxlen=4)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def act(self, observation):
        preprocessed = self.preprocess(observation)

        if len(self.frame_stack) < 4:
            for _ in range(4):
                self.frame_stack.append(preprocessed)
        else:
            self.frame_stack.append(preprocessed)

        state = np.stack(self.frame_stack, axis=0)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        return action
