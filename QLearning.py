import numpy as np


class QLearningAgent:
    def __init__(
        self,
        observation_space,
        action_space,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
    ):
        self.state_size = observation_space.n
        self.action_size = action_space.n
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = np.zeros((self.state_size, self.action_size))

    def choose_action(self, state, greedy=False):
        if greedy:
            return int(np.argmax(self.Q[state]))

        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            return int(np.argmax(self.Q[state]))

    def learn(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[
            next_state, best_next_action
        ] * (not done)
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.lr * td_error

        if done:
            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay
            )
