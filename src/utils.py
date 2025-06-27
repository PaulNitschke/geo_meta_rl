import pickle

import numpy as np
import torch as th
import gym
from gym import spaces

from garage import EnvStep
from garage import StepType

import warnings

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


def load_replay_buffer(path: str,
                        N_steps: int) -> dict:
    """
    Load a stable-baselines3 replay buffer from a file.
    
    :param path: Path to the replay buffer file.
    :param N_steps: Number of steps to load from the replay buffer.

    :return: dict containing observations, actions, rewards, next_observations, and dones.
    """

    with open(path, 'rb') as file:
        replay_buffer = pickle.load(file)

    def clean_array(array: np.array) -> th.tensor:
        """Keeps the first N_steps, flattens the array along the number of environments dimension.
        array: np.array of shape (replay_buffer_size, num_envs, ...)."""
        tensor= th.tensor(array[:N_steps], dtype=th.float32)
        return tensor.flatten(start_dim=0, end_dim=1)

    if (replay_buffer.observations[N_steps,:,:]==0).all():
        warnings.warn("Replay buffer contains more samples than selected.")


    return {'observations': clean_array(replay_buffer.observations),
            'actions': clean_array(replay_buffer.actions),
            'rewards': clean_array(replay_buffer.rewards),
            'next_observations': clean_array(replay_buffer.next_observations),
            'dones': clean_array(replay_buffer.dones)}