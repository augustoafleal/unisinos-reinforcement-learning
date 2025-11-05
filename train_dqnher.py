import gymnasium as gym
from gymnasium.envs.registration import register
import time
import torch
import numpy as np

from algos.DQNHERAgent import DQNHERAgent
from util.Logger import Logger
from util.plots import plot_learning_curve
from util.RenderRecorder import RenderRecorder
from datetime import datetime

register(
    id="PirateIslands-v1",
    entry_point="envs.PirateIslandsEnv:PirateIslandsEnv",
)

state_encoding = "cnn"

config_list = {
    # id: [map_name, wind, max_steps, gamma, epsilon, dec_epsilon, learning_rate, description]
    # 1: ["random_12x12", True, 200, 0.95, 0.1, 0.0, 0.0007, "γ=0.95 | ε fixo | lr=0.0007"],
    # 2: ["random_12x12", True, 200, 0.95, 0.1, 0.0, 0.001, "γ=0.95 | ε fixo | lr=0.001"],
    3: ["random_12x12", True, 200, 0.95, 1.0, 1e-5, 0.0007, "γ=0.95 | ε decaindo | lr=0.0007"],
    4: ["random_12x12", True, 200, 0.95, 1.0, 1e-5, 0.001, "γ=0.95 | ε decaindo | lr=0.001"],
    # 5: ["random_12x12", True, 200, 0.99, 0.1, 0.0, 0.0007, "γ=0.99 | ε fixo | lr=0.0007"],
    # 6: ["random_12x12", True, 200, 0.99, 0.1, 0.0, 0.001, "γ=0.99 | ε fixo | lr=0.001"],
    # 7: ["random_12x12", True, 200, 0.99, 1.0, 1e-5, 0.0007, "γ=0.99 | ε decaindo | lr=0.0007"],
    # 8: ["random_12x12", True, 200, 0.99, 1.0, 1e-5, 0.001, "γ=0.99 | ε decaindo | lr=0.001"],
}

for cfg_id, cfg in config_list.items():
    map_name, wind, max_steps, gamma, epsilon, dec_epsilon, lr, desc = cfg
    print("\n" + "=" * 80)
    print(f"Config {cfg_id}")
    print(f"Mapa: {map_name} | Wind: {wind}")
    print(f"Gamma: {gamma}")
    print(f"Epsilon: {epsilon} ({desc})")
    print("=" * 80)

    env = gym.make(
        "PirateIslands-v1",
        map_name=map_name,
        state_encoding=state_encoding,
        is_blowing_in_the_wind=wind,
        seed=123,
    )

    print("Observation space:", env.observation_space)
    print("Observation shape:", env.observation_space.shape)

    start = datetime.now()
    obs_shape = env.observation_space.shape
    if state_encoding == "cnn":
        n_states = obs_shape
    else:
        n_states = obs_shape[0]

    agent = DQNHERAgent(
        n_states=n_states,
        n_actions=env.action_space.n,
        grid_size=env.unwrapped.grid_size,
        state_encoding=state_encoding,
        learning_rate=lr,
        gamma=gamma,
        epsilon=epsilon,
        batch_size=64,
        memory_size=100000,
        replace_target=200,
        dec_epsilon=dec_epsilon,
        min_epsilon=0.05,
        checkpoint_dir=f"model/dqnher/",
        name=f"dqnher_config_{cfg_id}",
    )

    logger = Logger(filename=f"logs/episodes_logs_config_list_item_{map_name}_{cfg_id}.csv")

    num_episodes = 10000
    max_steps_per_episode = max_steps

    for ep in range(num_episodes):
        state, info = env.reset()
        if ep == 0:
            env.render()
        done = False
        total_reward = 0
        total_steps = 0
        goal = env.unwrapped.encode_goal()

        for step in range(max_steps_per_episode):
            action = agent.choose_action(state, goal)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_experience(state, action, reward, next_state, done, goal)
            agent.learn()

            state = next_state
            total_reward += reward
            total_steps += 1

            if done:
                break

        elapsed_time = datetime.now() - start
        print(f"Episode {ep+1}/{num_episodes} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")
        logger.log(ep + 1, total_steps, total_reward, done, elapsed_time)

    plot_learning_curve(f"logs/episodes_logs_config_list_item_{map_name}_{cfg_id}.csv")
    agent.q_eval.save_checkpoint()

    env = gym.make(
        "PirateIslands-v1",
        map_name=map_name,
        state_encoding=state_encoding,
        is_blowing_in_the_wind=wind,
        render_mode="rgb_array_tilemap",
        seed=123,
    )

    recorder = RenderRecorder(fps=4)
    state, info = env.reset()
    done = False
    goal = env.unwrapped.encode_goal()
    while not done:
        frame = env.render()
        recorder.capture(frame)

        action = agent.choose_action(state, goal)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        time.sleep(0.1)

    frame = env.render()
    recorder.capture(frame)
    recorder.save()

    env.close()
