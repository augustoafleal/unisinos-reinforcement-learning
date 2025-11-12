import gymnasium as gym
from gymnasium.envs.registration import register
import time
import torch
import numpy as np
from datetime import datetime
import os
from algos.PPOAgent import PPO
from util.Logger import Logger
from util.WorkerLogger import WorkerLogger
from util.plots import plot_learning_curve_workers
from util.RenderRecorder import RenderRecorder

register(
    id="PirateIslands-v1",
    entry_point="envs.PirateIslandsEnv:PirateIslandsEnv",
)

state_encoding = "cnn"

# map, wind, steps, gamma, epsilon, dec_epsilon, epsilon_min, lr, ent_coef, batch_size, use_epsilon, desc
config_list = {
    1: [
        "random_12x12",
        True,
        128,
        0.99,
        0.00,
        0.00,
        0.00,
        0.0007,
        0.02,
        64,
        False,
        "γ=0.99 | lr=7e-4 | steps=128 | batch=64 | eps=off",
    ],
    2: [
        "random_12x12",
        True,
        256,
        0.98,
        0.00,
        0.00,
        0.00,
        0.0005,
        0.02,
        128,
        False,
        "γ=0.98 | lr=5e-4 | steps=256 | batch=128 | eps=off",
    ],
    3: [
        "random_12x12",
        True,
        512,
        0.97,
        0.00,
        0.00,
        0.00,
        0.0003,
        0.02,
        32,
        False,
        "γ=0.97 | lr=3e-4 | steps=512 | batch=32 | eps=off",
    ],
    4: [
        "random_12x12",
        True,
        128,
        0.99,
        0.10,
        4e-3,
        0.01,
        0.0007,
        0.02,
        64,
        True,
        "γ=0.99 | lr=7e-4 | steps=128 | batch=64 | eps=on(0.99→0.01)",
    ],
    5: [
        "random_12x12",
        True,
        256,
        0.98,
        0.20,
        3e-3,
        0.02,
        0.0005,
        0.05,
        128,
        True,
        "γ=0.98 | lr=5e-4 | steps=256 | batch=128 | eps=on(0.99→0.02)",
    ],
    6: [
        "random_12x12",
        True,
        128,
        0.97,
        0.30,
        2e-3,
        0.05,
        0.0003,
        0.10,
        32,
        True,
        "γ=0.97 | lr=3e-4 | steps=512 | batch=32 | eps=on(0.99→0.05)",
    ],
    7: [
        "random_12x12",
        True,
        128,
        0.99,
        0.00,
        0.00,
        0.00,
        0.0007,
        0.10,
        64,
        False,
        "γ=0.99 | lr=7e-4 | ent=0.10 | steps=128 | eps=off",
    ],
    8: [
        "random_12x12",
        True,
        128,
        0.98,
        0.00,
        0.00,
        0.00,
        0.0005,
        0.20,
        16,
        False,
        "γ=0.98 | lr=5e-4 | ent=0.20 | steps=256 | eps=off",
    ],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

n_envs = 16


def make_env():
    def _thunk():
        return gym.make(
            "PirateIslands-v1",
            map_name=map_name,
            state_encoding=state_encoding,
            is_blowing_in_the_wind=wind,
            seed=123,
        )

    return _thunk


for cfg_id, cfg in config_list.items():
    (
        map_name,
        wind,
        max_steps,
        gamma,
        epsilon,
        dec_epsilon,
        epsilon_min,
        lr,
        ent_coef,
        batch_size,
        use_epsilon,
        desc,
    ) = cfg

    print("\n" + "=" * 80)
    print(f"Config {cfg_id}")
    print(f"Mapa: {map_name} | Wind: {wind}")
    print(f"Gamma: {gamma}")
    print("=" * 80)

    tmp_env = gym.make(
        "PirateIslands-v1",
        map_name=map_name,
        state_encoding=state_encoding,
        is_blowing_in_the_wind=wind,
        seed=123,
    )
    obs_shape = tmp_env.observation_space.shape
    if state_encoding == "cnn":
        n_states = obs_shape
        grid_size = tmp_env.unwrapped.grid_size
    else:
        n_states = obs_shape[0]
        grid_size = None
    tmp_env.close()

    print("Observation space:", obs_shape)

    env_fns = [make_env() for i in range(n_envs)]
    venv = gym.vector.SyncVectorEnv(env_fns)

    ppo = PPO(
        n_actions=venv.single_action_space.n,
        state_shape=n_states,
        device=device,
        cnn=(state_encoding == "cnn"),
        grid_size=grid_size,
        gamma=gamma,
        lam=0.95,
        clip_coef=0.2,
        ppo_epochs=4,
        batch_size=batch_size,
        lr=lr,
        ent_coef=ent_coef,
        vf_coef=1.0,
    )

    logger = Logger(filename=f"logs/ppo_updates_{map_name}_{cfg_id}.csv")
    worker_logger = WorkerLogger(filename=f"logs/worker_logs_ppo_{map_name}_{cfg_id}.csv")

    max_episodes = 1  # 0_000
    max_steps_per_update = max_steps
    start = datetime.now()
    finished_rewards = np.zeros(n_envs, dtype=np.float32)
    episode_counts = np.zeros(n_envs, dtype=int)
    episode_rewards = np.zeros(n_envs, dtype=float)
    episode_steps = np.zeros(n_envs, dtype=int)
    update = 0

    while True:
        update += 1
        obs, infos = venv.reset()
        episode_rewards = np.zeros(n_envs, dtype=np.float32)
        episode_steps = np.zeros(n_envs, dtype=int)

        buf_states, buf_actions, buf_rewards, buf_dones, buf_log_probs, buf_values = ([], [], [], [], [], [])

        for t in range(max_steps_per_update):
            actions, log_probs, values = ppo.select_action(obs)

            if use_epsilon:
                random_mask = np.random.rand(n_envs) < epsilon
                if np.any(random_mask):
                    random_actions = np.array([venv.single_action_space.sample() for _ in range(n_envs)])
                    actions[random_mask] = random_actions[random_mask]

            next_obs, rewards, terminated, truncated, infos = venv.step(actions)
            dones = np.logical_or(terminated, truncated)

            buf_states.append(np.asarray(obs))
            buf_actions.append(actions)
            buf_rewards.append(np.asarray(rewards))
            buf_dones.append(dones.astype(float))
            buf_log_probs.append(log_probs)
            buf_values.append(values)

            episode_rewards += rewards
            episode_steps += 1

            for i in range(n_envs):
                if dones[i]:
                    episode_counts[i] += 1

                    finished_rewards[i] = episode_rewards[i]

                    worker_logger.log(
                        episode=episode_counts[i],
                        worker_id=i,
                        reward=float(episode_rewards[i]),
                        steps=int(episode_steps[i]),
                        done=True,
                        elapsed_time=(datetime.now() - start),
                    )

                    episode_rewards[i] = 0.0
                    episode_steps[i] = 0

            obs = next_obs

        obs_tensor = torch.tensor(np.asarray(obs), dtype=torch.float32, device=device)
        with torch.no_grad():
            last_values_tensor, _ = ppo.forward(obs_tensor)
        last_values = last_values_tensor.squeeze(-1).cpu().numpy()

        buf_states = np.asarray(buf_states)
        buf_actions = np.asarray(buf_actions)

        buf_rewards = np.asarray(buf_rewards)
        mean_r = buf_rewards.mean()
        std_r = buf_rewards.std() + 1e-8
        buf_rewards = (buf_rewards - mean_r) / std_r

        buf_dones = np.asarray(buf_dones)
        buf_log_probs = np.asarray(buf_log_probs)
        buf_values = np.asarray(buf_values)

        rollouts = {
            "states": buf_states,
            "actions": buf_actions,
            "rewards": buf_rewards,
            "dones": buf_dones,
            "log_probs": buf_log_probs,
            "values": buf_values,
            "last_values": last_values,
        }

        metrics = ppo.update(rollouts)

        elapsed_time = datetime.now() - start
        mean_reward = float(np.mean(finished_rewards[finished_rewards != 0])) if np.any(finished_rewards != 0) else 0.0
        epsilon = max(epsilon_min, epsilon - dec_epsilon)

        print("=" * 100)
        print(
            f"PPO Update {update}| Episodes: {episode_counts[1]:.2f} | Elapsed Time: {str(elapsed_time).split('.')[0]}"
        )
        print(f"Mean episodic reward (finished envs): {mean_reward:.2f}")
        formatted_rewards = [f"{r:.3f}" for r in finished_rewards]
        print(f"Completed episode rewards per thread: {formatted_rewards}")
        print(f"Epsilon: {epsilon:.2f}")
        print(f"Entropy Coef: {ppo.ent_coef:.2f}")
        print(
            f"ActorLoss: {metrics['actor_loss']:.4f} | "
            f"CriticLoss: {metrics['critic_loss']:.4f} | "
            f"Entropy: {metrics['entropy']:.4f}"
        )

        if np.all(episode_counts >= max_episodes):
            venv.close()
            break

    venv.close()
    plot_learning_curve_workers(f"logs/worker_logs_ppo_{map_name}_{cfg_id}.csv", window_size=20)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/ppo_{map_name}_{cfg_id}.pth"
    torch.save(
        {
            "model_state_dict": ppo.state_dict(),
            "optimizer_state_dict": ppo.optimizer.state_dict(),
            "config": cfg,
            "update": update,
            "timestamp": datetime.now().isoformat(),
        },
        model_path,
    )
    print(f"Model saved: {model_path}")

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
    total_reward = 0.0

    while not done:
        frame = env.render()
        recorder.capture(frame)

        actions, _, _ = ppo.select_action([state])
        action = int(actions[0])

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state
        time.sleep(0.1)

    frame = env.render()
    recorder.capture(frame)
    recorder.save()
    env.close()

    print(f"Test episode ended — total reward: {total_reward:.2f}\n")
