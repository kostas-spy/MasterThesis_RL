import logging
from gym.envs.registration import register

register(
    id='mazeMap-v0',
    entry_point='methods.envs:MazeEnvSample',
    max_episode_steps=2000,
)
