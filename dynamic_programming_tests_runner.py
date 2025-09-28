from gymnasium.envs.registration import register
import gymnasium as gym
from datetime import datetime
from util.RenderRecorder import RenderRecorder
from util.plots import plot_and_save_policy, plot_best_actions_grid
from algos.DynamicProgrammingAgentFactory import DynamicProgrammingAgentFactory
import pandas as pd
from datetime import datetime

register(
    id="PirateIslands-v0",
    entry_point="envs.PirateIslandsEnv:PirateIslandsEnv",
)

eps = [1e-1, 1e-2, 1e-4, 1e-6]
iteration = ["value_iteration", "policy_iteration"]
winds = [True, False]
maps = [["4x4", 4], ["8x8", 8]]
gammas = [0.9, 0.95, 0.99]

df = pd.DataFrame(
    columns=["map", "wind", "iteration", "gamma", "epsilon", "steps", "reward", "iterations", "states_iteration"]
)

for m in maps:
    for wind in winds:
        for iterat in iteration:
            for epsilon_test in eps:

                env = gym.make(
                    "PirateIslands-v0", render_mode="rgb_array_tilemap", map_name=m[0], is_blowing_in_the_wind=wind
                )

                grid_size = m[1]

                obs, info = env.reset()
                print("\n\nMap:", m[0], "| Wind:", wind, "| Iteration:", iterat, "| Epsilon:", epsilon_test)

                for gamma in gammas:
                    print("Testing gamma:", gamma)

                    agent = DynamicProgrammingAgentFactory.create(
                        agent_type=iterat,
                        grid_size=grid_size,
                        islands=info["islands"],
                        goal=info["treasure"],
                        enemies=info["enemies"],
                        gamma=gamma,
                        epsilon=epsilon_test,
                        stochastic=wind,
                    )

                    agent.train()
                    print("Iterations for convergence:", agent.iteration)
                    print("States iterations for convergence:", agent.states_iteration)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    gstr = str(gamma).replace(".", "")
                    policy_filename = f"plots/policy_grid_{m[0]}_{iterat}_gamma{gstr}_{timestamp}.png"
                    best_filename = f"plots/best_actions_{m[0]}_{iterat}_gamma{gstr}_{timestamp}.png"

                    plot_and_save_policy(agent, filename=policy_filename)
                    plot_best_actions_grid(agent, filename=best_filename)

                    obs, info = env.reset()
                    recorder = RenderRecorder(fps=4)

                    pos_index = info["agent_pos"][1] * grid_size + info["agent_pos"][0]
                    visited_tuple = info["visited_islands"]
                    state = (pos_index, visited_tuple)

                    done = False
                    steps = 0
                    max_steps = 100
                    total_reward = 0

                    while not done and steps < max_steps:
                        frame = env.render()
                        recorder.capture(frame)

                        action = agent.act(state)

                        obs, reward, done, truncated, info = env.step(action)
                        total_reward += reward

                        pos_index = info["agent_pos"][1] * grid_size + info["agent_pos"][0]
                        visited_tuple = info["visited_islands"]
                        state = (pos_index, visited_tuple)

                        steps += 1

                    frame = env.render()
                    recorder.capture(frame)
                    recorder.save()

                    print("Episode finished in", steps, "steps. With Reward of", round(total_reward, 2))

                    new_line = pd.DataFrame(
                        [
                            {
                                "map": m[0],
                                "wind": wind,
                                "iteration": iterat,
                                "gamma": gamma,
                                "epsilon": epsilon_test,
                                "steps": steps,
                                "reward": round(total_reward, 2),
                                "iterations": agent.iteration,
                                "states_iteration": agent.states_iteration,
                            }
                        ]
                    )

                    df = pd.concat([df, new_line], ignore_index=True)

                env.close()

date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
df.to_csv(f"tests_results_{date_time}.csv", index=False)
