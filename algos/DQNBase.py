import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import copy


class DQNBase(nn.Module):
    def __init__(self, input_dims, n_actions, learning_rate=3e-4, checkpoint_dir="/model/dqn/", name="dqn"):
        super().__init__()

        self.fc1 = nn.Linear(input_dims, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, n_actions)

        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.constant_(layer.bias, 0)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(checkpoint_dir, name + ".pth")

    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))


class DQNCNN(nn.Module):
    def __init__(
        self, input_channels, grid_size, n_actions, learning_rate=3e-4, checkpoint_dir="/model/dqn/", name="dqn_cnn"
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        conv_output_size = 64 * (grid_size // 2) * (grid_size // 2)

        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(checkpoint_dir, name + ".pth")

    def forward(self, state):

        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        print(f"Model saved to {self.checkpoint_file}")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
