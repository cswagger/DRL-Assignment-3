import gym
import torch
import gc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from torchvision.transforms import Compose, ToPILImage, Grayscale, Resize, ToTensor
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT


class QNet(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU()
        )
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, num_actions)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        v = self.value_stream(x)
        a = self.advantage_stream(x)
        return v + (a - a.mean(dim=1, keepdim=True))



class Agent:
    """Super Mario DQN Agent with frame skipping and state stacking."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = len(COMPLEX_MOVEMENT)
        self.model = QNet(4, self.n_actions).to(self.device)
        self.model.eval()

        try:
            ckpt = torch.load("best_model.pth", map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        except Exception as e:
            print(f"[Agent] Model load failed: {e}")

        self.processor = Compose([
            ToPILImage(),
            Grayscale(),
            Resize((84, 84)),
            ToTensor()
        ])
        self.frames = deque(maxlen=4)
        self.skip_interval = 3
        self.remaining_skips = 0
        self.cached_action = 0
        self.warmup_needed = True
        self.gc_tick = 0

    def act(self, observation):
        """Select action from current raw RGB frame (240x256x3)."""
        if observation.shape != (240, 256, 3):
            raise ValueError(f"Invalid input shape: {observation.shape}")

        self.gc_tick += 1
        if self.gc_tick % 60 == 0:
            gc.collect()

        frame = self.processor(np.ascontiguousarray(observation)).squeeze(0).numpy()

        if self.warmup_needed:
            self.frames.extend([frame] * 4)
            self.warmup_needed = False

        if self.remaining_skips > 0:
            self.remaining_skips -= 1
            return self.cached_action

        self.frames.append(frame)
        stacked = np.stack(self.frames, axis=0)
        input_tensor = torch.from_numpy(stacked).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(input_tensor)
            action = int(q_values.argmax(dim=1).item())

        self.cached_action = action
        self.remaining_skips = self.skip_interval
        return action
