# Pirate Islands Environment

**`PirateIslandsEnv`** is a discrete grid environment inspired by pirate adventures.  
The goal of the agent is to navigate the sea, visit all clue islands (`I`) at least once, and then reach the treasure (`T`), while avoiding enemy ships (`E`).

The environment can optionally include wind, which adds stochasticity to the agent's movement.

## Key Information

| Feature            | Description                                 |
|-------------------|---------------------------------------------|
| Action Space       | Discrete(4)                                 |
| Observation Space  | Discrete(16)                                |
| Import             | `gymnasium.make("PirateIslands-v0")`       |

## Environment Description

- **Grid:** Square grid of size `4x4` or `8x8`.  
- **Grid elements:**
  - `S` – starting position of the agent
  - `W` – water (navigable)
  - `I` – clue island
  - `T` – treasure
  - `E` – enemy pirate

- **Objective:** Visit all clue islands exactly once and then reach the treasure.  
- **Episode Termination:** Occurs when:
  - The agent reaches the treasure after visiting all islands
  - The agent collides with an enemy ship
  - The agent revisits the same island

## Observation Space

- Type: `Discrete(grid_size * grid_size * (num_islands + 1))`  
- Encodes the agent's position and number of clue islands visited:
  - `pos_index = y * grid_size + x`
  - `visited_count = number of islands visited` (0 até num_islands)
  - Encoded state: `state = pos_index * (num_islands + 1) + visited_count`

## Action Space

- Type: `Discrete(4)`  
- Actions:
  - `0` – move up
  - `1` – move down
  - `2` – move left
  - `3` – move right

> If wind is active (`is_blowing_in_the_wind=True`):  
> - With probability `wind_prob`, the chosen action can be deviated.  
> - The deviation is limited to adjacent actions (`left` or `right` relative to the original action).  
> - The resulting position is validated: it cannot move the agent into an enemy or an already visited clue island.  
> - If no valid deviation exists, the original action is applied.

## Rewards

- `-0.1` – normal movement
- `+1` – visit a clue island for the first time
- `-1` – revisit the same island
- `-10` – collide with an enemy ship
- `+10` – reach the treasure after visiting all clue islands

## Rendering Modes

- `text` – simple text-based grid:
  - `A` – agent
  - `I` – unvisited clue island
  - `i` – visited clue island
  - `T` – treasure
  - `E` – enemy ship
  - `W` – water

- `text_emoji` – emoji-based grid (optional):
  - `⛵` – agent
  - `🏝️` – unvisited clue island
  - `🚩` – visited clue island
  - `💰` – treasure
  - `☠️` – enemy pirate
  - `🌊` – water

## Example Usage

```python
env = PirateIslandsEnv(map_name="4x4", render_mode="text")
obs, _ = env.reset()
env.render()

obs, reward, terminated, truncated, info = env.step(1)
env.render()