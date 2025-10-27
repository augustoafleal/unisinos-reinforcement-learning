import gymnasium as gym
from gymnasium.envs.registration import register
from datetime import datetime
import numpy as np
from algos.OffPolicyMCAgent import OffPolicyMCAgent
from util.RenderRecorder import RenderRecorder
from util.Logger import Logger
from util.plots import plot_learning_curve, plot_mc_agent_policy_grid
import time
import pygame

register(
    id="PirateIslands-v1",
    entry_point="envs.PirateIslandsEnv:PirateIslandsEnv",
)

map_sizes = ["4x4", "8x8", "random_12x12", "random_16x16"]

for map_name in map_sizes:
    env = gym.make(
        "PirateIslands-v1",
        map_name=map_name,
        render_mode="human_tilemap",
        randomize_each_reset=True,
        seed=None,
    )

    state, info = env.reset()
    env.render()

    time.sleep(2)
    pygame.quit()
