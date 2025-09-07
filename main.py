from gymnasium.envs.registration import register
import gymnasium as gym
import time
from QLearning import QLearningAgent
from Logger import Logger
from datetime import datetime

register(
    id="PirateIslands-v0",
    entry_point="PirateIslandsEnv:PirateIslandsEnv",
)

env = gym.make(
    "PirateIslands-v0",
    render_mode="text_emoji",
    map_name="4x4",
    # is_blowing_in_the_wind=True,
    # wind_prob=0.15,
    # map_name="8x8",
)

logger = Logger(
    f"logs/pirateislands_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"
)

agent = QLearningAgent(
    env.observation_space,
    env.action_space,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay=0.99,
    epsilon_min=0.01,
)

n_episodes = 500
max_steps_per_episode = 50

for episode in range(n_episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps_per_episode):
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.learn(state, action, reward, next_state, terminated)
        state = next_state
        total_reward += reward
        # env.render()

        if terminated:
            logger.log(episode, step + 1, total_reward, terminated)
            break

    print(f"Episode {episode+1}: Total Reward = {total_reward}")
# print(agent.Q)
