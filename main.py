import gymnasium as gym
from gymnasium.envs.registration import register
from datetime import datetime
import numpy as np
import pygame
import time
from util.RenderRecorder import RenderRecorder

register(
    id="PirateIslands-v1",
    entry_point="envs.PirateIslandsEnv:PirateIslandsEnv",
)

map_sizes = ["4x4", "8x8", "random_12x12", "random_16x16"]

for map_name in map_sizes:
    env = gym.make(
        "PirateIslands-v1",
        map_name=map_name,
        render_mode="rgb_array_tilemap",
        randomize_each_reset=True,
        state_encoding="positions",
        seed=None,
    )

    recorder = RenderRecorder(f"video/{map_name}.mp4", fps=4)

    obs, info = env.reset()
    done = False

    while not done:
        # ação aleatória só para gerar o vídeo
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        recorder.capture(frame)

        done = terminated or truncated
        time.sleep(0.1)  # desacelera a execução para visualização

    recorder.save()
    pygame.quit()
