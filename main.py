import gymnasium as gym
from gymnasium.envs.registration import register
from datetime import datetime
import numpy as np
from algos.OffPolicyMCAgent import OffPolicyMCAgent
from util.RenderRecorder import RenderRecorder
from util.Logger import Logger
from util.plots import plot_learning_curve, plot_mc_agent_policy_grid

register(
    id="PirateIslands-v0",
    entry_point="envs.PirateIslandsEnv:PirateIslandsEnv",
)

test_configs = [
    {"map": "4x4", "wind": True, "gamma": 0.95},
    {"map": "4x4", "wind": False, "gamma": 0.95},
    {"map": "8x8", "wind": True, "gamma": 0.95},
    {"map": "8x8", "wind": False, "gamma": 0.95},
]

epsilon_start = 1.0
epsilon_final = 0.01
num_episodes = 60000
decay_episodes = 40000
epsilon_decay = (epsilon_final / epsilon_start) ** (1 / decay_episodes)
max_steps = 100


def run_episode(env, agent, epsilon, max_steps=100):
    states, actions, rewards = [], [], []
    obs, info = env.reset()
    s = int(obs)
    done = False
    steps = 0
    while not done and steps < max_steps:
        if np.random.rand() < epsilon:
            a = np.random.randint(agent.nA)
        else:
            a = agent.greedy_action(s)
        next_obs, r, terminated, truncated, info = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = int(next_obs)
        done = terminated or truncated
        steps += 1
    return states, actions, rewards


for cfg in test_configs:
    print(f"\n=== Training: map={cfg['map']} wind={cfg['wind']} gamma={cfg['gamma']} ===")

    env = gym.make(
        "PirateIslands-v0",
        render_mode=None,
        map_name=cfg["map"],
        is_blowing_in_the_wind=cfg["wind"],
        state_encoding="bitmask",
    )

    nS = env.observation_space.n
    nA = env.action_space.n
    agent = OffPolicyMCAgent(nS, nA)

    logger = Logger()
    rewards_history = []
    epsilon = epsilon_start

    for ep in range(1, num_episodes + 1):
        states, actions, ep_rewards = run_episode(env, agent, epsilon, max_steps=max_steps)
        agent.update(states, actions, ep_rewards, gamma=cfg["gamma"], epsilon=epsilon)

        total_reward = sum(ep_rewards)
        rewards_history.append(total_reward)

        logger.log(
            episode=ep,
            total_steps=len(ep_rewards),
            total_reward=total_reward,
            terminated=(total_reward > 0),
        )

        if ep < decay_episodes:
            epsilon *= epsilon_decay
        else:
            epsilon = epsilon_final

        if ep % 100 == 0:
            avg_last_100 = np.mean(rewards_history[-100:])
            print(f"[Ep {ep}] Avg reward (last 100): {avg_last_100:.3f} | Epsilon: {epsilon:.3f}")

    plot_learning_curve(logger.filename, window_size=25)

    obs, info = env.reset()
    state = int(obs)
    done = False
    steps = 0
    total_reward = 0

    while not done and steps < max_steps:
        # frame = env.render()
        # recorder.capture(frame)
        action = agent.greedy_action(state)
        next_obs, reward, terminated, truncated, info = env.step(action)
        state = int(next_obs)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    frame = env.render()
    # recorder.capture(frame)
    # recorder.save()

    print(f"Eval finished in {steps} steps. Reward: {total_reward:.2f}")

    grid_size = int(cfg["map"].split("x")[0])
    plot_mc_agent_policy_grid(agent, info, grid_size)

    env.close()
