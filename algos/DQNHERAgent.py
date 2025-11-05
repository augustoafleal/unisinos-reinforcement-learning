import torch
import numpy as np
import os
import copy
from algos.DQNBase import DQNBase
from algos.DQNBase import DQNCNN


class DQNHERAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        grid_size,
        state_encoding="mlp",
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        batch_size=64,
        memory_size=100000,
        replace_target=200,
        dec_epsilon=1e-5,
        min_epsilon=0.1,
        checkpoint_dir="/tmp/ddqn/",
        name="dqnh",
    ):
        self.n_actions = n_actions
        self.n_states = n_states
        self.state_encoding = state_encoding
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.replace_target_count = replace_target
        self.dec_epsilon = dec_epsilon
        self.min_epsilon = min_epsilon
        self.learn_step_counter = 0

        if state_encoding == "cnn":
            in_channels, grid_h, grid_w = n_states
            self.q_eval = DQNCNN(in_channels * 2, grid_h, n_actions, learning_rate, checkpoint_dir, name + "_eval")
            self.q_next = DQNCNN(in_channels * 2, grid_h, n_actions, learning_rate, checkpoint_dir, name + "_target")
        else:
            input_dims = n_states * 2
            self.q_eval = DQNBase(input_dims, n_actions, learning_rate, checkpoint_dir, name + "_eval")
            self.q_next = copy.deepcopy(self.q_eval)

        # Memória HER
        if state_encoding == "cnn":
            # input_dims = n_states  # n_states = (C,H,W) -> memory armazena array C*H*W?
            # self.memory = HindsightExperienceReplayMemoryCNN(
            #    memory_size, input_shape=(input_dims[0], grid_size, grid_size), n_actions=n_actions
            # )
            C, H, W = n_states
            self.memory = HindsightExperienceReplayMemoryCNN(memory_size, input_shape=(C, H, W), n_actions=n_actions)
        else:
            self.memory = HindsightExperienceReplayMemory(memory_size, n_states, n_actions)

    def store_experience(self, state, action, reward, next_state, done, goal):
        self.memory.add_experience(state, action, reward, next_state, done, goal)

    def preprocess_state(self, state, goal=None):
        if self.state_encoding == "cnn":
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.q_eval.device)
            if goal is not None:
                goal_tensor = torch.tensor(goal, dtype=torch.float32).unsqueeze(0).to(self.q_eval.device)
                return state_tensor, goal_tensor
            return state_tensor
        else:
            if goal is not None:
                concat = np.concatenate([state, goal])
            else:
                concat = state
            state_tensor = torch.tensor(concat, dtype=torch.float32).unsqueeze(0).to(self.q_eval.device)
            return state_tensor

    def choose_action(self, state, goal):
        if np.random.rand() > self.epsilon:
            if self.state_encoding == "cnn":
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.q_eval.device)
                goal_tensor = torch.tensor(goal, dtype=torch.float32).unsqueeze(0).to(self.q_eval.device)
                input_tensor = torch.cat([state_tensor, goal_tensor], dim=1)
                q_values = self.q_eval.forward(input_tensor)
                return torch.argmax(q_values).item()
            else:
                state_goal = self.preprocess_state(state, goal)
                q_values = self.q_eval(state_goal)
                return torch.argmax(q_values).item()
        else:
            return np.random.choice(self.n_actions)

    def sample_batch(self):
        return self.memory.get_random_experience(self.batch_size)

    def decrement_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.dec_epsilon)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_count == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def learn(self):
        if self.memory.counter < self.batch_size:
            return

        states, actions, rewards, next_states, dones, goals = self.sample_batch()

        if self.state_encoding == "cnn":
            states = torch.tensor(states, dtype=torch.float32).to(self.q_eval.device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(self.q_eval.device)
            goals = torch.tensor(goals, dtype=torch.float32).to(self.q_eval.device)

            state_goal = torch.cat([states, goals], dim=1)
            next_state_goal = torch.cat([next_states, goals], dim=1)
        else:
            states = torch.tensor(states, dtype=torch.float32).to(self.q_eval.device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(self.q_eval.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.q_eval.device)
            actions = torch.tensor(actions).to(self.q_eval.device)
            dones = torch.tensor(dones).to(self.q_eval.device)
            goals = torch.tensor(goals, dtype=torch.float32).to(self.q_eval.device)

            state_goal = torch.cat([states, goals], dim=1)
            next_state_goal = torch.cat([next_states, goals], dim=1)

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        q_pred = self.q_eval(state_goal)[np.arange(self.batch_size), actions]
        q_next = self.q_next(next_state_goal).max(dim=1)[0]
        q_next[dones] = 0.0

        if self.state_encoding == "cnn":
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.q_eval.device)
            q_next = torch.tensor(q_next, dtype=torch.float32).to(self.q_eval.device)

        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss_fn(q_pred, q_target)
        loss.backward()
        self.q_eval.optimizer.step()

        self.decrement_epsilon()
        self.learn_step_counter += 1


class HindsightExperienceReplayMemory:
    def __init__(self, memory_size, input_dims, n_actions, her_k=4):
        self.max_mem_size = memory_size
        self.counter = 0
        self.her_k = her_k

        self.state_memory = np.zeros((memory_size, input_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((memory_size, input_dims), dtype=np.float32)
        self.reward_memory = np.zeros(memory_size, dtype=np.float32)
        self.action_memory = np.zeros(memory_size, dtype=np.int64)
        self.terminal_memory = np.zeros(memory_size, dtype=bool)
        self.goal_memory = np.zeros((memory_size, input_dims), dtype=np.float32)

        self.episode_buffer = []

    def add_experience(self, state, action, reward, next_state, done, goal):
        self.episode_buffer.append((state, action, reward, next_state, done, goal))

        if done:
            self._store_episode_with_her()
            self.episode_buffer = []  # limpa o buffer do episódio

    def _store_episode_with_her(self):
        episode = self.episode_buffer
        ep_len = len(episode)

        for t, (state, action, reward, next_state, done, goal) in enumerate(episode):
            self._store(state, action, reward, next_state, done, goal)

            future_indexes = np.random.choice(np.arange(t, ep_len), size=self.her_k, replace=True)
            for future_t in future_indexes:
                _, _, _, future_state, _, _ = episode[future_t]

                new_goal = future_state.copy()
                new_reward = self.compute_reward(next_state, new_goal)

                self._store(state, action, new_reward, next_state, done, new_goal)

    def _store(self, state, action, reward, next_state, done, goal):
        idx = self.counter % self.max_mem_size
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.next_state_memory[idx] = next_state
        self.terminal_memory[idx] = done
        self.goal_memory[idx] = goal
        self.counter += 1

    def compute_reward(self, next_state, goal):
        return 1.0 if np.allclose(next_state, goal, atol=1e-2) else 0.0

    def get_random_experience(self, batch_size):
        max_mem = min(self.counter, self.max_mem_size)
        batch_idx = np.random.choice(max_mem, batch_size, replace=False)

        states = np.concatenate([self.state_memory[batch_idx], self.goal_memory[batch_idx]], axis=1)
        next_states = np.concatenate([self.next_state_memory[batch_idx], self.goal_memory[batch_idx]], axis=1)

        return (
            states,
            self.action_memory[batch_idx],
            self.reward_memory[batch_idx],
            next_states,
            self.terminal_memory[batch_idx],
        )


class HindsightExperienceReplayMemoryCNN:
    def __init__(self, memory_size, input_shape, n_actions, her_k=4):
        self.max_mem_size = memory_size
        self.counter = 0
        self.her_k = her_k

        C, H, W = input_shape
        self.state_memory = np.zeros((memory_size, C, H, W), dtype=np.float32)
        self.next_state_memory = np.zeros((memory_size, C, H, W), dtype=np.float32)
        self.reward_memory = np.zeros(memory_size, dtype=np.float32)
        self.action_memory = np.zeros(memory_size, dtype=np.int64)
        self.terminal_memory = np.zeros(memory_size, dtype=bool)
        self.goal_memory = np.zeros((memory_size, C, H, W), dtype=np.float32)

        self.episode_buffer = []

    def add_experience(self, state, action, reward, next_state, done, goal):
        self.episode_buffer.append((state, action, reward, next_state, done, goal))

        if done:
            self._store_episode_with_her()
            self.episode_buffer = []

    def _store_episode_with_her(self):
        episode = self.episode_buffer
        ep_len = len(episode)

        for t, (state, action, reward, next_state, done, goal) in enumerate(episode):
            self._store(state, action, reward, next_state, done, goal)

            future_indexes = np.random.choice(np.arange(t, ep_len), size=self.her_k, replace=True)
            for future_t in future_indexes:
                _, _, _, future_state, _, _ = episode[future_t]

                new_goal = future_state.copy()
                new_reward = self.compute_reward(next_state, new_goal)

                self._store(state, action, new_reward, next_state, done, new_goal)

    def _store(self, state, action, reward, next_state, done, goal):
        idx = self.counter % self.max_mem_size
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.next_state_memory[idx] = next_state
        self.terminal_memory[idx] = done
        self.goal_memory[idx] = goal
        self.counter += 1

    def compute_reward(self, next_state, goal):
        return 1.0 if np.allclose(next_state, goal, atol=1e-2) else 0.0

    def get_random_experience(self, batch_size):
        max_mem = min(self.counter, self.max_mem_size)
        batch_idx = np.random.choice(max_mem, batch_size, replace=False)
        return (
            self.state_memory[batch_idx],
            self.action_memory[batch_idx],
            self.reward_memory[batch_idx],
            self.next_state_memory[batch_idx],
            self.terminal_memory[batch_idx],
            self.goal_memory[batch_idx],
        )
