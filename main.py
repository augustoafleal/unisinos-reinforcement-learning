from gymnasium.envs.registration import register
import gymnasium as gym
from datetime import datetime
from util.RenderRecorder import RenderRecorder
from util.plots import plot_and_save_policy, plot_best_actions_grid
from algos.DynamicProgrammingAgentFactory import DynamicProgrammingAgentFactory


register(
    id="PirateIslands-v0",
    entry_point="envs.PirateIslandsEnv:PirateIslandsEnv",
)

env = gym.make("PirateIslands-v0", render_mode="rgb_array_tilemap", map_name="4x4", is_blowing_in_the_wind=False)

grid_size = 4

obs, info = env.reset()
print("Initial State:", obs)
print("Info:", info)


agent = DynamicProgrammingAgentFactory.create(
    # agent_type="value_iteration",
    agent_type="policy_iteration",
    grid_size=grid_size,
    islands=info["islands"],
    goal=info["treasure"],
    enemies=info["enemies"],
    gamma=0.9,
    epsilon=1e-1,
    stochastic=True,
)

agent.train()
print("Iterations for convergence:", agent.iteration)
print("States iterations for convergence:", agent.states_iteration)

plot_and_save_policy(agent)
plot_best_actions_grid(agent)

obs, info = env.reset()
recorder = RenderRecorder(fps=4)

pos_index = info["agent_pos"][1] * grid_size + info["agent_pos"][0]
visited_tuple = info["visited_islands"]
state = (pos_index, visited_tuple)

done = False
steps = 0
max_steps = 50

while not done and steps < max_steps:
    frame = env.render()
    recorder.capture(frame)

    action = agent.act(state)

    obs, reward, done, truncated, info = env.step(action)

    pos_index = info["agent_pos"][1] * grid_size + info["agent_pos"][0]
    visited_tuple = info["visited_islands"]
    state = (pos_index, visited_tuple)

    steps += 1

frame = env.render()
recorder.capture(frame)
recorder.save()
env.close()

print("Episode finished in", steps, "steps.")
