import torch
import torch.nn as nn
import numpy as np
import copy
from algos.DQNBase import DQNBase, DQNCNN


class DQNAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        grid_size=None,
        state_encoding="mlp",
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        batch_size=64,
        memory_size=100000,
        replace_target=200,
        dec_epsilon=1e-5,
        min_epsilon=0.1,
        checkpoint_dir="/tmp/dqn/",
        name="dqn",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.replace_target_count = replace_target
        self.dec_epsilon = dec_epsilon
        self.min_epsilon = min_epsilon
        self.learn_step_counter = 0
        self.state_encoding = state_encoding

        if state_encoding == "cnn":
            in_channels, grid_h, grid_w = n_states
            self.q_eval = DQNCNN(
                input_channels=in_channels,
                grid_size=grid_h,
                n_actions=n_actions,
                learning_rate=learning_rate,
                checkpoint_dir=checkpoint_dir,
                name=name + "_eval",
            )
            self.q_next = DQNCNN(
                input_channels=in_channels,
                grid_size=grid_h,
                n_actions=n_actions,
                learning_rate=learning_rate,
                checkpoint_dir=checkpoint_dir,
                name=name + "_target",
            )
            self.memory = ReplayMemory(memory_size, n_states, n_actions, is_cnn=True)
        else:
            self.q_eval = DQNBase(
                input_dims=n_states,
                n_actions=n_actions,
                learning_rate=learning_rate,
                checkpoint_dir=checkpoint_dir,
                name=name + "_eval",
            )
            self.q_next = copy.deepcopy(self.q_eval)
            self.memory = ReplayMemory(memory_size, n_states, n_actions, is_cnn=False)

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            if self.state_encoding == "cnn":
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.q_eval.device)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.q_eval.device)

            q_values = self.q_eval(state_tensor)
            action = torch.argmax(q_values).item()
        else:
            action = np.random.choice(self.n_actions)
        return action

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_count == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.dec_epsilon)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)

        if self.state_encoding == "cnn":
            states = torch.tensor(states, dtype=torch.float32).to(self.q_eval.device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(self.q_eval.device)
        else:
            states = torch.tensor(states, dtype=torch.float32).to(self.q_eval.device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(self.q_eval.device)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.q_eval.device)
        actions = torch.tensor(actions).to(self.q_eval.device)
        dones = torch.tensor(dones).to(self.q_eval.device)

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        q_pred = self.q_eval(states)[np.arange(self.batch_size), actions]
        q_next = self.q_next(next_states).max(dim=1)[0]
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss_fn(q_pred, q_target)
        loss.backward()
        self.q_eval.optimizer.step()

        self.decrement_epsilon()
        self.learn_step_counter += 1


class ReplayMemory:
    def __init__(self, memory_size, input_shape, n_actions, is_cnn=False):
        self.mem_size = memory_size
        self.mem_cntr = 0
        self.is_cnn = is_cnn
        self.n_actions = n_actions

        if is_cnn:
            C, H, W = input_shape
            self.state_memory = np.zeros((memory_size, C, H, W), dtype=np.float32)
            self.new_state_memory = np.zeros((memory_size, C, H, W), dtype=np.float32)
        else:
            self.state_memory = np.zeros((memory_size, input_shape), dtype=np.float32)
            self.new_state_memory = np.zeros((memory_size, input_shape), dtype=np.float32)

        self.action_memory = np.zeros(memory_size, dtype=np.int64)
        self.reward_memory = np.zeros(memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(memory_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
