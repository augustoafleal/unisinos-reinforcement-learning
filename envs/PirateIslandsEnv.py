import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class PirateIslandsEnv(gym.Env):
    metadata = {"render_modes": ["text", "text_emoji", "human_tilemap", "rgb_array_tilemap"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = "text",
        is_blowing_in_the_wind=False,
        wind_prob=0.30,
        map_name="4x4",
        state_encoding="count",
    ):
        super().__init__()
        self.render_mode = render_mode
        self.is_blowing_in_the_wind = is_blowing_in_the_wind
        self.wind_prob = wind_prob
        self.map_description = self.get_map(map_name)
        self.grid_size = len(self.map_description)
        self.state_encoding = state_encoding

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

        self.num_islands = len(self.islands)

        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size * (self.num_islands + 1))
        self.action_space = spaces.Discrete(4)

        self.agent_pos = None
        self.visited = None
        self.agent_direction_map = {0: "A_up", 1: "A_down", 2: "A_left", 3: "A_right"}
        self.agent_direction_index = 1
        self.tileset_path = "assets/beach_tileset.png"

        if self.state_encoding == "count":
            self.observation_space = spaces.Discrete(self.grid_size * self.grid_size * (self.num_islands + 1))
        elif self.state_encoding == "bitmask":
            self.observation_space = spaces.Discrete(self.grid_size * self.grid_size * (1 << self.num_islands))
        else:
            raise ValueError(f"Unknown state_encoding: {self.state_encoding}")

    def encode_state(self):
        pos_index = self.agent_pos[1] * self.grid_size + self.agent_pos[0]

        if self.state_encoding == "count":
            visited_count = sum(self.visited.values())
            self.num_islands = len(self.islands)
            return pos_index * (self.num_islands + 1) + visited_count

        elif self.state_encoding == "bitmask":
            visited_mask = 0
            for idx, v in enumerate(self.visited.values()):
                if v:
                    visited_mask |= 1 << idx
            return pos_index * (1 << self.num_islands) + visited_mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = list(self.start)
        self.visited = {i: False for i in range(len(self.islands))}

        if hasattr(self, "render_mode") and self.render_mode in ["human_tilemap", "rgb_array_tilemap"]:
            self._render_tilemap()

        visited_tuple = tuple(int(v) for v in self.visited.values())
        info = {
            "islands": self.islands,
            "enemies": self.enemies,
            "treasure": self.treasure,
            "agent_pos": tuple(self.agent_pos),
            "visited_islands": visited_tuple,
        }
        return self.encode_state(), info

    def step(self, action):
        prev_pos = tuple(self.agent_pos)
        if self.is_blowing_in_the_wind and np.random.rand() < self.wind_prob:
            # print(f"Wind is blowing!")
            self.agent_pos = list(self._choose_wind_position(action))
        else:
            self.agent_pos = list(self._next_position(action))

        reward = -0.1
        terminated = False
        truncated = False

        if tuple(self.agent_pos) in self.enemies:
            reward -= 10
            terminated = True

        # for idx, island in enumerate(self.islands):
        #    if tuple(self.agent_pos) == island:
        #        if not self.visited[idx]:
        #            self.visited[idx] = True
        #            reward = 1
        #        else:
        #            reward = -1
        #            terminated = True
        for idx, island in enumerate(self.islands):
            if tuple(self.agent_pos) == island:
                if not self.visited[idx]:
                    self.visited[idx] = True
                    reward += 1
                elif tuple(self.agent_pos) != prev_pos:
                    reward -= 1
                    terminated = True

        if tuple(self.agent_pos) == self.treasure:
            if all(self.visited.values()):
                reward += 10
                terminated = True
            else:
                reward -= 1
                terminated = True

        visited_tuple = tuple(int(v) for v in self.visited.values())
        info = {
            "islands": self.islands,
            "enemies": self.enemies,
            "treasure": self.treasure,
            "agent_pos": tuple(self.agent_pos),
            "visited_islands": visited_tuple,
        }
        return self.encode_state(), reward, terminated, truncated, info

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
                self.agent_direction_index = candidate_action
                return new_pos

        fallback_pos = self._next_position(action)
        return fallback_pos

    def _next_position(self, action, pos=None):
        self.agent_direction_index = action
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

    def render(self):
        if self.render_mode is None:
            print("No render mode specified.")
            return
        if self.render_mode == "text":
            return self._render_text()
        elif self.render_mode == "text_emoji":
            return self._render_text_emoji()
        elif self.render_mode in ["human_tilemap", "rgb_array_tilemap"]:
            return self._render_tilemap()
        else:
            raise ValueError(
                f"Unknown render_mode: {self.render_mode}. Supported modes: {self.metadata['render_modes']}"
            )

    def _render_text(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

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
        grid = [["🌊" for _ in range(self.grid_size)] for _ in range(self.grid_size)]

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

    def _render_tilemap(self):
        tile_size = 32
        zoom = 4

        if not hasattr(self, "pygame_initialized"):
            pygame.init()

            if self.render_mode == "human_tilemap":
                self.screen = pygame.display.set_mode(
                    (self.grid_size * tile_size * zoom, self.grid_size * tile_size * zoom)
                )
            else:
                self.screen = pygame.Surface((self.grid_size * tile_size * zoom, self.grid_size * tile_size * zoom))

            self.clock = pygame.time.Clock()
            if self.render_mode == "human_tilemap":
                self.tileset_image = pygame.image.load("assets/beach_tileset.png").convert_alpha()
            else:
                self.tileset_image = pygame.image.load("assets/beach_tileset.png")

            x_start, x_end = 1, 3
            y_start, y_end = 3, 5
            width = (x_end - x_start + 1) * tile_size
            height = (y_end - y_start + 1) * tile_size
            I_base_combined = pygame.Surface((width, height), pygame.SRCALPHA)
            for y in range(y_start, y_end + 1):
                for x in range(x_start, x_end + 1):
                    rect = pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size)
                    tile = self.tileset_image.subsurface(rect)
                    I_base_combined.blit(tile, ((x - x_start) * tile_size, (y - y_start) * tile_size))
            I_base_combined = pygame.transform.scale(I_base_combined, (tile_size, tile_size))

            self.tiles = {
                "W": self.tileset_image.subsurface((1 * tile_size, 2 * tile_size, tile_size, tile_size)),
                "S": self.tileset_image.subsurface((1 * tile_size, 2 * tile_size, tile_size, tile_size)),
                "I_base": I_base_combined,
                "I_top": self.tileset_image.subsurface((5 * tile_size, 0 * tile_size, tile_size, tile_size)),
                "T_closed": self.tileset_image.subsurface((1 * tile_size, 0 * tile_size, tile_size, tile_size)),
                "T_open": self.tileset_image.subsurface((3 * tile_size, 0 * tile_size, tile_size, tile_size)),
                "E_boat": self.tileset_image.subsurface((7 * tile_size, 1 * tile_size, tile_size, tile_size)),
                "E_pirate": self.tileset_image.subsurface((10 * tile_size, 0, tile_size, tile_size)),
                "A_on_I": self.tileset_image.subsurface((14 * tile_size, 0 * tile_size, tile_size, tile_size)),
                "A_up": self.tileset_image.subsurface((8 * tile_size, 1 * tile_size, tile_size, tile_size)),
                "A_down": self.tileset_image.subsurface((9 * tile_size, 1 * tile_size, tile_size, tile_size)),
                "A_left": self.tileset_image.subsurface((9 * tile_size, 0 * tile_size, tile_size, tile_size)),
                "A_right": self.tileset_image.subsurface((8 * tile_size, 0 * tile_size, tile_size, tile_size)),
            }

            self.pygame_initialized = True

        if self.render_mode == "human_tilemap":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

        surface = pygame.Surface((self.grid_size * tile_size, self.grid_size * tile_size), pygame.SRCALPHA)

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                surface.blit(self.tiles["W"], (x * tile_size, y * tile_size))

                if (x, y) == self.start:
                    surface.blit(self.tiles["S"], (x * tile_size, y * tile_size))

                if (x, y) == self.treasure:
                    if (x, y) == tuple(self.agent_pos) and all(self.visited.values()):
                        surface.blit(self.tiles["I_base"], (x * tile_size, y * tile_size))
                        surface.blit(self.tiles["T_open"], (x * tile_size, y * tile_size))
                    else:
                        surface.blit(self.tiles["I_base"], (x * tile_size, y * tile_size))
                        surface.blit(self.tiles["T_closed"], (x * tile_size, y * tile_size))

                if (x, y) in self.islands:
                    idx = self.islands.index((x, y))
                    surface.blit(self.tiles["I_base"], (x * tile_size, y * tile_size))
                    offset_y = 5
                    if self.visited[idx]:
                        surface.blit(self.tiles["I_top"], (x * tile_size, y * tile_size - offset_y))

                if (x, y) in self.enemies:
                    surface.blit(self.tiles["E_boat"], (x * tile_size, y * tile_size))
                    offset_y = 6
                    surface.blit(self.tiles["E_pirate"], (x * tile_size, y * tile_size - offset_y))

                if (x, y) == tuple(self.agent_pos):
                    if (x, y) in self.islands:
                        offset_y = 8
                        surface.blit(self.tiles["A_on_I"], (x * tile_size, y * tile_size - offset_y))
                    elif not (x, y) in self.enemies and not (x, y) == tuple(self.treasure):
                        agent_direction = self.agent_direction_map[self.agent_direction_index]
                        surface.blit(self.tiles[agent_direction], (x * tile_size, y * tile_size))

        # grid_color = (128, 128, 128)
        grid_color = (255, 255, 255)
        line_width = 1
        for x in range(0, self.grid_size * tile_size, tile_size):
            pygame.draw.line(surface, grid_color, (x, 0), (x, self.grid_size * tile_size), line_width)
        for y in range(0, self.grid_size * tile_size, tile_size):
            pygame.draw.line(surface, grid_color, (0, y), (self.grid_size * tile_size, y), line_width)

        self.screen.blit(pygame.transform.scale(surface, self.screen.get_size()), (0, 0))

        if self.render_mode == "human_tilemap":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array_tilemap":
            rgb_array = pygame.surfarray.array3d(self.screen)
            rgb_array = np.transpose(rgb_array, (1, 0, 2))
            return rgb_array

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
