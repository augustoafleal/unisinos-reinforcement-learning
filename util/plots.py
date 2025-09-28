import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from itertools import product
from collections import defaultdict
from datetime import datetime


def plot_and_save_policy(agent, filename=None, figsize=(12, 8)):

    if agent.grid_size > 4:
        print(f"Plot skipped: grid {agent.grid_size}x{agent.grid_size} is too large.")
        return

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plots/policy_grid_{timestamp}.png"

    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    visited_combinations = list(product([0, 1], repeat=agent.num_islands))

    n_combinations = len(visited_combinations)
    cols = min(4, n_combinations)
    rows = (n_combinations + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_combinations == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = {
        "water": "#4A90E2",
        "enemy": "#E74C3C",
        "treasure": "#F1C40F",
        "island": "#2ECC71",
        "visited": "#27AE60",
        "terminal": "#95A5A6",
        "action": "#FFFFFF",
    }

    action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    for idx, visited_tuple in enumerate(visited_combinations):
        if idx >= len(axes):
            break

        ax = axes[idx]

        grid_display = np.ones((agent.grid_size, agent.grid_size, 3))
        text_annotations = []

        for y in range(agent.grid_size):
            for x in range(agent.grid_size):
                display_y = agent.grid_size - 1 - y

                pos_index = agent.index_from_position(x, y)
                state = (pos_index, visited_tuple)

                if (x, y) == (0, 0):
                    color = colors["water"]
                    base_symbol = "S"
                elif (x, y) in agent.enemies:
                    color = colors["enemy"]
                    base_symbol = "E"
                elif (x, y) == agent.goal:
                    color = colors["treasure"]
                    base_symbol = "T"
                elif (x, y) in agent.islands:
                    island_idx = agent.islands.index((x, y))
                    if visited_tuple[island_idx] == 1:
                        color = colors["visited"]
                        base_symbol = "✓"
                    else:
                        color = colors["island"]
                        base_symbol = "I"
                else:
                    color = colors["water"]
                    base_symbol = ""

                action = agent.policy.get(state)
                if action is not None:
                    action_symbol = action_symbols[action]
                    if base_symbol:
                        symbol = f"{base_symbol}\n{action_symbol}"
                    else:
                        symbol = action_symbol
                else:
                    symbol = base_symbol if base_symbol else "·"

                if isinstance(color, str):
                    color = color.lstrip("#")
                    color = tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

                grid_display[display_y, x] = color
                text_annotations.append((x, display_y, symbol))

        ax.imshow(grid_display, origin="upper")

        for x, y, symbol in text_annotations:
            if "\n" in symbol:
                ax.text(x, y, symbol, ha="center", va="center", fontsize=10, fontweight="bold", color="white")
            else:
                ax.text(x, y, symbol, ha="center", va="center", fontsize=14, fontweight="bold", color="white")

        ax.set_xlim(-0.5, agent.grid_size - 0.5)
        ax.set_ylim(-0.5, agent.grid_size - 0.5)

        ax.set_xticks(np.arange(agent.grid_size))
        ax.set_yticks(np.arange(agent.grid_size))

        ax.set_xticks(np.arange(-0.5, agent.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, agent.grid_size, 1), minor=True)

        ax.grid(which="minor", color="white", linewidth=2, alpha=0.7)

        ax.grid(which="major", visible=False)

        ax.set_xticklabels([str(i) for i in range(agent.grid_size)])
        ax.set_yticklabels([str(agent.grid_size - 1 - i) for i in range(agent.grid_size)])
        ax.tick_params(axis="both", which="both", length=0)

        visited_count = sum(visited_tuple)

        island_info = []
        for i, (ix, iy) in enumerate(agent.islands):
            status = "✓" if visited_tuple[i] == 1 else "✗"
            island_info.append(f"I{i+1}({ix},{iy}):{status}")

        islands_str = " | ".join(island_info)

        ax.set_title(f"{visited_count}/{agent.num_islands} Visited islands\n{islands_str}", fontsize=9)

    for idx in range(n_combinations, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(
        "Learned Policy\n"
        f"Grid {agent.grid_size}x{agent.grid_size}, {agent.num_islands} islands, "
        f"{len(agent.enemies)} enemies",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Policy saved: {filename}")


def save_policy_text(agent, filename="policy.txt"):
    """Saves the policy in text format"""
    action_chars = ["↑", "↓", "←", "→"]
    visited_combinations = list(product([0, 1], repeat=agent.num_islands))

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== LEARNED POLICY - VALUE ITERATION ===\n")
        f.write(f"Grid: {agent.grid_size}x{agent.grid_size}\n")
        f.write(f"Islands: {agent.num_islands}\n")
        f.write(f"Enemies: {len(agent.enemies)}\n")
        f.write(f"Iterations: {agent.iteration}\n\n")

        for visited_tuple in visited_combinations:
            visited_count = sum(visited_tuple)
            f.write(f"Policy for {visited_count}/{agent.num_islands} islands visited {visited_tuple}:\n")

            for y in range(agent.grid_size):
                row = []
                for x in range(agent.grid_size):
                    pos_index = agent.index_from_position(x, y)
                    state = (pos_index, visited_tuple)

                    if (x, y) in agent.enemies:
                        row.append("E")
                    elif (x, y) == agent.goal:
                        row.append("G")
                    elif (x, y) in agent.islands:
                        island_idx = agent.islands.index((x, y))
                        if visited_tuple[island_idx] == 1:
                            row.append("✓")
                        else:
                            row.append("I")
                    elif agent.policy.get(state) is None:
                        row.append("·")
                    else:
                        action = agent.policy.get(state)
                        if action is not None:
                            row.append(action_chars[action])
                        else:
                            row.append("·")
                f.write(" ".join(row) + "\n")
            f.write("\n")

    print(f"Policy saved as text: {filename}")


def plot_best_actions_grid(agent, filename=None, figsize=(10, 10)):

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plots/best_actions_{timestamp}.png"

    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    colors = {
        "water": "#4A90E2",
        "enemy": "#E74C3C",
        "treasure": "#F1C40F",
        "island": "#2ECC71",
    }

    action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    position_actions = defaultdict(set)

    visited_combinations = list(product([0, 1], repeat=agent.num_islands))

    for y in range(agent.grid_size):
        for x in range(agent.grid_size):
            pos_index = agent.index_from_position(x, y)

            for visited_state in visited_combinations:
                state = (pos_index, visited_state)
                action = agent.policy.get(state)

                if action is not None:
                    position_actions[(x, y)].add(action)

    grid_display = np.ones((agent.grid_size, agent.grid_size, 3))

    for y in range(agent.grid_size):
        for x in range(agent.grid_size):

            display_y = agent.grid_size - 1 - y

            if (x, y) in agent.enemies:
                color = colors["enemy"]
            elif (x, y) == agent.goal:
                color = colors["treasure"]
            elif (x, y) in agent.islands:
                color = colors["island"]
            else:
                color = colors["water"]

            if isinstance(color, str):
                color = color.lstrip("#")
                color = tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

            grid_display[display_y, x] = color

    ax.imshow(grid_display, origin="upper")

    for y in range(agent.grid_size):
        for x in range(agent.grid_size):
            display_y = agent.grid_size - 1 - y

            base_symbol = ""
            if (x, y) == (0, 0):
                base_symbol = "S"
            elif (x, y) in agent.enemies:
                base_symbol = "E"
            elif (x, y) == agent.goal:
                base_symbol = "T"
            elif (x, y) in agent.islands:
                island_idx = agent.islands.index((x, y))
                base_symbol = f"I{island_idx+1}"

            if base_symbol:
                ax.text(
                    x,
                    display_y - 0.3,
                    base_symbol,
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white",
                )

    for y in range(agent.grid_size):
        for x in range(agent.grid_size):
            display_y = agent.grid_size - 1 - y
            actions = position_actions.get((x, y), set())

            if not actions:
                continue

            arrow_positions = {
                0: (0, 0.15),
                1: (0, -0.15),
                2: (-0.15, 0),
                3: (0.15, 0),
            }

            for action in actions:
                if action in arrow_positions:
                    dx, dy = arrow_positions[action]
                    symbol = action_symbols[action]

                    ax.text(
                        x + dx,
                        display_y + dy,
                        symbol,
                        ha="center",
                        va="center",
                        fontsize=16,
                        fontweight="bold",
                        color="white",
                    )

    ax.set_xlim(-0.5, agent.grid_size - 0.5)
    ax.set_ylim(-0.5, agent.grid_size - 0.5)

    ax.set_xticks(np.arange(agent.grid_size))
    ax.set_yticks(np.arange(agent.grid_size))

    ax.set_xticks(np.arange(-0.5, agent.grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, agent.grid_size, 1), minor=True)

    ax.grid(which="minor", color="white", linewidth=2, alpha=0.7)

    ax.grid(which="major", visible=False)
    ax.tick_params(axis="both", which="both", length=0)

    ax.set_xticklabels([str(i) for i in range(agent.grid_size)])
    ax.set_yticklabels([str(agent.grid_size - 1 - i) for i in range(agent.grid_size)])

    plt.title(
        f"All Best Actions by Position\n"
        f"Grid {agent.grid_size}x{agent.grid_size} - Considering all island states\n"
        f"Yellow arrows: policy actions | E: Enemy | G: Goal | I: Island",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Best options grid saved: {filename}")


def plot_learning_curve(csv_file, window_size=25, aura_factor=0.5, aura_alpha=0.1, filename=None):
    os.makedirs("plots", exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plots/learning_curve_{timestamp}.png"
    df = pd.read_csv(csv_file)

    if not {"episode", "total_reward"}.issubset(df.columns):
        raise ValueError("CSV precisa ter colunas 'episode' e 'total_reward'.")

    rewards = df["total_reward"].values
    episodes = df["episode"].values
    rewards_series = pd.Series(rewards)
    mean_rewards_moving_avg = rewards_series.rolling(window=window_size).mean()
    std_rewards_moving = rewards_series.rolling(window=window_size).std()

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, mean_rewards_moving_avg, color="orange", label="Average reward", linewidth=2)
    plt.fill_between(
        episodes,
        mean_rewards_moving_avg - aura_factor * std_rewards_moving,
        mean_rewards_moving_avg + aura_factor * std_rewards_moving,
        color="orange",
        alpha=aura_alpha,
    )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as '{filename}'")


def extract_position_from_state(state_int, grid_size, num_islands):
    island_combinations = 2**num_islands

    pos_index = state_int // island_combinations

    y = pos_index // grid_size
    x = pos_index % grid_size

    return x, y


def plot_mc_agent_policy_grid(agent, info, grid_size, filename=None, figsize=(10, 10)):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plots/mc_policy_fixed_{timestamp}.png"

    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    colors = {
        "water": "#4A90E2",
        "enemy": "#E74C3C",
        "treasure": "#F1C40F",
        "island": "#2ECC71",
    }

    action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    islands = info["islands"]
    enemies = info["enemies"]
    treasure = info["treasure"]

    position_actions = defaultdict(set)

    num_islands = len(islands)

    print(f"Analisando {len(agent.pi)} estados da política...")

    for state_int, action in agent.pi.items():
        try:
            x, y = extract_position_from_state(state_int, grid_size, num_islands)
            if 0 <= x < grid_size and 0 <= y < grid_size:
                position_actions[(x, y)].add(action)
        except:
            pass

    if not position_actions:
        print("Política vazia, tentando Q-values...")
        for state_int in agent.Q.keys():
            try:
                action = agent.greedy_action(state_int)
                x, y = extract_position_from_state(state_int, grid_size, num_islands)
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    position_actions[(x, y)].add(action)
            except:
                pass

    if not position_actions:
        for state_int, action in agent.pi.items():
            if state_int < grid_size * grid_size:
                y = state_int // grid_size
                x = state_int % grid_size
                position_actions[(x, y)].add(action)

    grid_display = np.ones((grid_size, grid_size, 3))

    for y in range(grid_size):
        for x in range(grid_size):
            display_y = grid_size - 1 - y

            if (x, y) in enemies:
                color = colors["enemy"]
            elif (x, y) == treasure:
                color = colors["treasure"]
            elif (x, y) in islands:
                color = colors["island"]
            else:
                color = colors["water"]

            if isinstance(color, str):
                color = color.lstrip("#")
                color = tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

            grid_display[display_y, x] = color

    ax.imshow(grid_display, origin="upper")

    for y in range(grid_size):
        for x in range(grid_size):
            display_y = grid_size - 1 - y

            base_symbol = ""
            if (x, y) == (0, 0):
                base_symbol = "S"
            elif (x, y) in enemies:
                base_symbol = "E"
            elif (x, y) == treasure:
                base_symbol = "T"
            elif (x, y) in islands:
                island_idx = islands.index((x, y))
                base_symbol = f"I{island_idx+1}"

            if base_symbol:
                ax.text(
                    x,
                    display_y - 0.3,
                    base_symbol,
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white",
                )

    for y in range(grid_size):
        for x in range(grid_size):
            display_y = grid_size - 1 - y
            actions = position_actions.get((x, y), set())

            if not actions:
                continue

            arrow_positions = {
                0: (0, 0.15),
                1: (0, -0.15),
                2: (-0.15, 0),
                3: (0.15, 0),
            }

            for action in actions:
                if action in arrow_positions:
                    dx, dy = arrow_positions[action]
                    symbol = action_symbols[action]

                    ax.text(
                        x + dx,
                        display_y + dy,
                        symbol,
                        ha="center",
                        va="center",
                        fontsize=16,
                        fontweight="bold",
                        color="white",
                    )

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))

    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)

    ax.grid(which="minor", color="white", linewidth=1, alpha=0.7)
    ax.grid(which="major", visible=False)
    ax.tick_params(axis="both", which="both", length=0)

    ax.set_xticklabels([str(i) for i in range(grid_size)])
    ax.set_yticklabels([str(grid_size - 1 - i) for i in range(grid_size)])

    plt.title(
        f"Monte Carlo Agent Policy\n"
        f"Grid {grid_size}x{grid_size} - {len(position_actions)} positions with learned actions\n"
        f"Yellow arrows: learned actions | E: Enemy | T: Treasure | I: Island | S: Start",
        fontsize=max(8, 12 - grid_size // 2),
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Fixed Monte Carlo policy grid saved: {filename}")
    return position_actions
