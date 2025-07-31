import pickle
import os
import warnings

import numpy as np
import torch as th
import gym
from gym import spaces

from garage import EnvStep
from garage import StepType
from .learning.symmetry.kernel_approx import KernelFrameEstimator

class GarageToGymWrapper(gym.Env):
    """Converts a garage environment into a gym environment which can be used by Stable Baselines."""
    def __init__(self, task_env):
        self._env = task_env

        self.action_space = spaces.Box(low=self._env.action_space.low,
                                       high=self._env.action_space.high,
                                       dtype=self._env.action_space.dtype)
        self.observation_space = spaces.Box(low=self._env.observation_space.low,
                                            high=self._env.observation_space.high,
                                            dtype=self._env.observation_space.dtype)

    def reset(self):
        obs, _info = self._env.reset()
        return obs

    def step(self, action):
        step: EnvStep = self._env.step(action)
        obs = step.observation
        reward = step.reward
        done = step.step_type in (StepType.TERMINAL, StepType.TIMEOUT)
        info = step.env_info
        return obs, reward, done, info

    def render(self, mode="human"):
        return self._env.render(mode)
    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def close(self):
        self._env.close()


class Affine2D(th.nn.Module):
    """An affine neural network."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = th.nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        return self.linear(x)