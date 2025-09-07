import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PirateIslandsEnv(gym.Env):
    metadata = {"render_modes": ["text", "text_emoji"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = "text",
        is_blowing_in_the_wind=False,
        wind_prob=0.30,
        map_name="4x4",
    ):
        super().__init__()
        self.render_mode = render_mode
        self.is_blowing_in_the_wind = is_blowing_in_the_wind
        self.wind_prob = wind_prob
        self.map_description = self.get_map(map_name)
        self.grid_size = len(self.map_description)

        self.islands = []
        for y, row in enumerate(self.map_description):
            for x, cell in enumerate(row):
                if cell == "I":
                    self.islands.append((x, y))
        self.num_islands = len(self.islands)

        self.observation_space = spaces.Discrete(
            self.grid_size * self.grid_size * (self.num_islands + 1)
        )
        self.action_space = spaces.Discrete(4)

        self.start = None
        self.islands = []
        self.treasure = None
        self.enemies = []

        for y, row in enumerate(self.map_description):
            for x, cell in enumerate(row):
                if cell == "S":
                    self.start = (x, y)
                elif cell == "I":
                    self.islands.append((x, y))
                elif cell == "T":
                    self.treasure = (x, y)
                elif cell == "E":
                    self.enemies.append((x, y))

        self.agent_pos = None
        self.visited = None

    def encode_state(self):
        pos_index = self.agent_pos[1] * self.grid_size + self.agent_pos[0]
        visited_count = sum(self.visited.values())
        self.num_islands = len(self.islands)
        return pos_index * (self.num_islands + 1) + visited_count

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = list(self.start)
        self.visited = {i: False for i in range(len(self.islands))}
        return self.encode_state(), {}

    def step(self, action):
        if self.is_blowing_in_the_wind and np.random.rand() < self.wind_prob:
            # print(f"Window is blowing!")
            self.agent_pos = list(self._choose_wind_position(action))
        else:
            self.agent_pos = list(self._next_position(action))

        reward = -0.1
        terminated = False
        truncated = False

        if tuple(self.agent_pos) in self.enemies:
            reward = -10
            terminated = True

        for idx, island in enumerate(self.islands):
            if tuple(self.agent_pos) == island:
                if not self.visited[idx]:
                    self.visited[idx] = True
                    reward = 1
                else:
                    reward = -1
                    terminated = True

        if tuple(self.agent_pos) == self.treasure:
            if all(self.visited.values()):
                # print(f"Visited values: {self.visited.values()}")
                reward = 10
                terminated = True
            else:
                reward = -1
                terminated = True

        return self.encode_state(), reward, terminated, truncated, {}

    def _choose_wind_position(self, action):
        possible_shifts = [-1, 1]

        for shift in np.random.permutation(possible_shifts):
            candidate_action = (action + shift) % 4
            new_pos = self._next_position(candidate_action)

            if new_pos in self.enemies:
                continue

            for idx, island in enumerate(self.islands):
                if new_pos == island and self.visited[idx]:
                    break
            else:
                return new_pos

        fallback_pos = self._next_position(action)
        return fallback_pos

    def render(self):
        if self.render_mode is None:
            print("No render mode specified.")
            return
        if self.render_mode == "text":
            return self._render_text()
        elif self.render_mode == "text_emoji":
            return self._render_text_emoji()
        else:
            raise ValueError(f"Unknown render_mode: {self.render_mode}")

    def _render_text(self):
        grid = [
            ["." for _ in range(self.grid_size)] for _ in range(self.grid_size)
        ]

        for ex, ey in self.enemies:
            grid[ey][ex] = "E"
        for idx, (ix, iy) in enumerate(self.islands):
            grid[iy][ix] = "I" if not self.visited[idx] else "i"
        tx, ty = self.treasure
        grid[ty][tx] = "T"
        ax, ay = self.agent_pos
        grid[ay][ax] = "A"

        out = "\n".join(" ".join(row) for row in grid)
        print(out + "\n")

    def _render_text_emoji(self):
        grid = [
            ["🌊" for _ in range(self.grid_size)] for _ in range(self.grid_size)
        ]

        for ex, ey in self.enemies:
            grid[ey][ex] = "☠️ "
        for idx, (ix, iy) in enumerate(self.islands):
            grid[iy][ix] = "🏝️ " if not self.visited[idx] else "🚩"

        tx, ty = self.treasure
        grid[ty][tx] = "💰"
        ax, ay = self.agent_pos
        grid[ay][ax] = "⛵"

        out = "\n".join(" ".join(row) for row in grid)
        print(out + "\n")

    def _next_position(self, action, pos=None):
        if pos is None:
            x, y = self.agent_pos
        else:
            x, y = pos

        if action == 0 and y > 0:
            return (x, y - 1)
        elif action == 1 and y < self.grid_size - 1:
            return (x, y + 1)
        elif action == 2 and x > 0:
            return (x - 1, y)
        elif action == 3 and x < self.grid_size - 1:
            return (x + 1, y)
        else:
            return (x, y)

    def get_map(self, name="4x4"):
        maps = {
            "4x4": ["SIWE", "WWWI", "WEWW", "IWWT"],
            "8x8": [
                "SWWWWWWI",
                "WWIWWWEW",
                "WWWEWWWW",
                "WWWIWWEW",
                "WWWEWWWW",
                "WIWWWWWW",
                "EWWEWIEW",
                "WEWWWWWT",
            ],
        }
        return maps[name]
