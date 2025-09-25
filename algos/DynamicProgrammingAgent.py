import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from abc import ABC, abstractmethod


class DynamicProgrammingAgent(ABC):
    def __init__(self, grid_size, islands, goal, enemies=None, gamma=0.9, epsilon=1e-4, stochastic=False):
        self.grid_size = grid_size
        self.islands = islands
        self.num_islands = len(islands)
        self.goal = goal
        self.enemies = enemies if enemies else []
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = [0, 1, 2, 3]
        self.stochastic = stochastic

        self.states = [
            (pos_index, tuple(visited))
            for pos_index in range(grid_size**2)
            for visited in product([0, 1], repeat=self.num_islands)
        ]

        self.V = {s: 0.0 for s in self.states}
        self.policy = {}
        self.P = {}
        self.iteration = 0
        self.states_iteration = 0

        if self.stochastic:
            self._build_model(wind_prob=0.2, blowing=True)
        else:
            self._build_model()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def act(self, state):
        pass

    def position_from_index(self, index):
        x = index % self.grid_size
        y = index // self.grid_size
        return x, y

    def index_from_position(self, x, y):
        return y * self.grid_size + x

    def _compute_next_state(self, state, action):
        pos_index, visited = state
        visited = list(visited)
        x, y = self.position_from_index(pos_index)

        if action == 0:
            y = max(y - 1, 0)
        elif action == 1:
            y = min(y + 1, self.grid_size - 1)
        elif action == 2:
            x = max(x - 1, 0)
        elif action == 3:
            x = min(x + 1, self.grid_size - 1)

        for idx, (ix, iy) in enumerate(self.islands):
            if (x, y) == (ix, iy):
                visited[idx] = 1

        next_pos_index = self.index_from_position(x, y)
        return (next_pos_index, tuple(visited))

    def _compute_reward(self, current_state, action, next_state):
        current_pos_index, current_visited = current_state
        next_pos_index, next_visited = next_state

        current_x, current_y = self.position_from_index(current_pos_index)
        next_x, next_y = self.position_from_index(next_pos_index)

        reward = -0.1

        if (next_x, next_y) in self.enemies:
            return -10.0

        for idx, (ix, iy) in enumerate(self.islands):
            if (next_x, next_y) == (ix, iy):
                if current_visited[idx] == 0 and next_visited[idx] == 1:
                    return 1.0
                elif current_visited[idx] == 1 and (current_x, current_y) != (next_x, next_y):
                    return -1.0

        if (next_x, next_y) == self.goal:
            if all(next_visited):
                return 10.0
            else:
                return -1.0

        return reward

    def _build_model(self, wind_prob=0.0, blowing=False):
        for state in self.states:
            for action in self.actions:
                transitions = []

                if not blowing or wind_prob == 0.0:
                    next_state = self._compute_next_state(state, action)
                    reward = self._compute_reward(state, action, next_state)
                    transitions.append((1.0, next_state, reward))
                else:
                    next_state = self._compute_next_state(state, action)
                    reward = self._compute_reward(state, action, next_state)
                    transitions.append((1 - wind_prob, next_state, reward))

                    for shift in [-1, +1]:
                        candidate_action = (action + shift) % 4
                        candidate_state = self._compute_next_state(state, candidate_action)
                        cx, cy = self.position_from_index(candidate_state[0])

                        if (cx, cy) in self.enemies:
                            continue

                        _, visited = state
                        for idx, (ix, iy) in enumerate(self.islands):
                            if (cx, cy) == (ix, iy) and visited[idx] == 1:
                                break
                        else:
                            candidate_reward = self._compute_reward(candidate_state, action, next_state)
                            transitions.append((wind_prob / 2, candidate_state, candidate_reward))

                total_prob = sum(p for p, _, _ in transitions)
                transitions = [(p / total_prob, s, r) for p, s, r in transitions]
                self.P[(state, action)] = transitions
