from gymnasium.envs.registration import register
import gymnasium as gym
import pandas as pd
import numpy as np
from datetime import datetime
from util.RenderRecorder import RenderRecorder
from util.Logger import Logger
from util.plots import plot_learning_curve
from algos.OffPolicyMCAgent import OffPolicyMCAgent


def run_episode(env, agent, epsilon=0.1, max_steps=100):
    states, actions, rewards = [], [], []
    obs, info = env.reset(seed=None)
    s = int(obs)
    done = False
    steps = 0
    while not done and steps < max_steps:
        # a = agent.act(s, epsilon)
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


register(
    id="PirateIslands-v0",
    entry_point="envs.PirateIslandsEnv:PirateIslandsEnv",
)

eps = [1.0]
winds = [True, False]
maps = [["4x4", 4], ["8x8", 8]]
gammas = [0.9, 0.95, 0.99]
num_episodes = 60000
epsilon_final = 0.01
decay_episodes = 40000

df = pd.DataFrame(columns=["map", "wind", "algo", "gamma", "epsilon_final", "steps", "reward", "avg_reward"])

for m in maps:
    for wind in winds:
        for epsilon_start in eps:
            for gamma in gammas:

                print(
                    "\n\nMap:",
                    m[0],
                    "| Wind:",
                    wind,
                    "| Algo: MC-OffPolicy",
                    "| Epsilon:",
                    epsilon_start,
                    "| Gamma:",
                    gamma,
                )

                env = gym.make(
                    "PirateIslands-v0",
                    map_name=m[0],
                    is_blowing_in_the_wind=wind,
                    # state_encoding="bitmask",
                )

                nS = env.observation_space.n
                nA = env.action_space.n
                agent = OffPolicyMCAgent(nS, nA)

                logger = Logger()

                epsilon_decay = (epsilon_final / epsilon_start) ** (1 / decay_episodes)
                epsilon = epsilon_start

                rewards = []
                for ep in range(1, num_episodes + 1):
                    states, actions, ep_rewards = run_episode(env, agent, epsilon)
                    agent.update(states, actions, ep_rewards, gamma, epsilon)

                    total_reward = sum(ep_rewards)
                    rewards.append(total_reward)

                    logger.log(
                        episode=ep,
                        total_steps=len(ep_rewards),
                        total_reward=total_reward,
                        terminated=(total_reward > 0),
                    )

                    if ep % 1000 == 0:
                        avg_last_100 = np.mean(rewards[-100:])
                        print(
                            f"[Ep {ep}] Avg reward (last 100): {avg_last_100:.3f} | "
                            f"Epsilon: {epsilon:.3f} | Last reward: {total_reward:.2f}"
                        )

                    # decaimento de epsilon
                    if ep < decay_episodes:
                        epsilon *= epsilon_decay
                    else:
                        epsilon = epsilon_final

                avg_reward = np.mean(rewards[-500:]) if len(rewards) >= 500 else np.mean(rewards)

                env = gym.make(
                    "PirateIslands-v0",
                    render_mode="rgb_array_tilemap",
                    map_name=m[0],
                    is_blowing_in_the_wind=wind,
                    # state_encoding="bitmask",
                )

                obs, info = env.reset()
                recorder = RenderRecorder(fps=4)
                state = int(obs)
                done = False
                total_reward = 0
                steps = 0
                max_steps = 100

                while not done and steps < max_steps:
                    frame = env.render()
                    recorder.capture(frame)

                    action = agent.greedy_action(state)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    state = int(next_obs)
                    total_reward += reward
                    steps += 1
                    done = terminated or truncated

                frame = env.render()
                recorder.capture(frame)
                recorder.save()

                print(
                    f"Eval finished in {steps} steps. Reward {round(total_reward, 2)} | Avg reward train: {avg_reward:.2f}"
                )

                # adicionar linha no DF geral
                new_line = pd.DataFrame(
                    [
                        {
                            "map": m[0],
                            "wind": wind,
                            "algo": "MC-OffPolicy",
                            "gamma": gamma,
                            "epsilon_final": epsilon_final,
                            "steps": steps,
                            "reward": round(total_reward, 2),
                            "avg_reward": avg_reward,
                        }
                    ]
                )
                df = pd.concat([df, new_line], ignore_index=True)

                plot_learning_curve(logger.filename, window_size=25)

                env.close()

date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
df.to_csv(f"tests_results_mc_{date_time}.csv", index=False)
