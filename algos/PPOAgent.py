import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class PPO(nn.Module):
    def __init__(
        self,
        n_actions,
        state_shape,
        device,
        cnn=False,
        grid_size=84,
        gamma=0.99,
        lam=0.95,
        clip_coef=0.2,
        ppo_epochs=4,
        batch_size=64,
        lr=3e-4,
        ent_coef=0.01,
        vf_coef=1,
    ):
        super().__init__()
        self.device = device
        self.cnn = cnn
        self.n_actions = n_actions
        self.gamma = gamma
        self.lam = lam
        self.clip_coef = clip_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        if cnn:
            C, H, W = state_shape
            self.conv1 = nn.Conv2d(C, 16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            conv_output_size = 64 * (grid_size // 2) * (grid_size // 2)
            self.fc = nn.Linear(conv_output_size, 256)
        else:
            input_dim = state_shape if isinstance(state_shape, int) else state_shape[0]
            self.fc1 = nn.Linear(input_dim, 256)
            self.fc2 = nn.Linear(256, 256)

        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.to(self.device)

    def parameters(self):
        if self.cnn:
            return (
                list(self.conv1.parameters())
                + list(self.conv2.parameters())
                + list(self.conv3.parameters())
                + list(self.fc.parameters())
                + list(self.actor.parameters())
                + list(self.critic.parameters())
            )
        else:
            return (
                list(self.fc1.parameters())
                + list(self.fc2.parameters())
                + list(self.actor.parameters())
                + list(self.critic.parameters())
            )

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        if self.cnn:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc(x))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

        value = self.critic(x)
        logits = self.actor(x)
        return value, logits

    def select_action(self, state):
        state_t = torch.tensor(np.asarray(state), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            value, logits = self.forward(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            value.squeeze(-1).cpu().numpy(),
        )

    def update(self, rollouts):
        device = self.device
        gamma = self.gamma
        lam = self.lam

        states = torch.tensor(rollouts["states"], dtype=torch.float32, device=device)
        actions = torch.tensor(rollouts["actions"], dtype=torch.int64, device=device)
        rewards = torch.tensor(rollouts["rewards"], dtype=torch.float32, device=device)
        dones = torch.tensor(rollouts["dones"], dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(rollouts["log_probs"], dtype=torch.float32, device=device)
        values = torch.tensor(rollouts["values"], dtype=torch.float32, device=device)
        last_values = torch.tensor(rollouts["last_values"], dtype=torch.float32, device=device)

        T, N = rewards.shape

        values = torch.cat([values, last_values.unsqueeze(0)], dim=0)

        advantages = torch.zeros(T, N, device=device)
        gae = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_flat = states.reshape(T * N, *states.shape[2:])
        actions_flat = actions.reshape(T * N)
        returns_flat = returns.reshape(T * N)
        advantages_flat = advantages.reshape(T * N)
        old_log_probs_flat = old_log_probs.reshape(T * N)

        actor_loss_epoch = 0
        critic_loss_epoch = 0
        entropy_epoch = 0

        total_size = T * N
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(total_size, device=device)
            for start in range(0, total_size, self.batch_size):
                idx = perm[start : start + self.batch_size]

                batch_states = states_flat[idx]
                batch_actions = actions_flat[idx]
                batch_returns = returns_flat[idx]
                batch_adv = advantages_flat[idx]
                batch_old_log_probs = old_log_probs_flat[idx]

                new_values, new_logits = self.forward(batch_states)
                dist = torch.distributions.Categorical(logits=new_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = torch.nn.functional.mse_loss(new_values.squeeze(-1), batch_returns)

                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                actor_loss_epoch += actor_loss.item()
                critic_loss_epoch += critic_loss.item()
                entropy_epoch += entropy.item()

        num_updates = (total_size // self.batch_size) * self.ppo_epochs
        return {
            "actor_loss": actor_loss_epoch / num_updates,
            "critic_loss": critic_loss_epoch / num_updates,
            "entropy": entropy_epoch / num_updates,
        }
