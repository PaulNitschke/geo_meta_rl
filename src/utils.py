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
        array = array[:N_steps]
        return th.tensor(array.reshape(-1, *array.shape[2:]), dtype=th.float32)

    if (replay_buffer.observations[N_steps,:,:]==0).all():
        warnings.warn("Replay buffer contains more samples than selected.")


    return {'observations': clean_array(replay_buffer.observations),
            'actions': clean_array(replay_buffer.actions),
            'rewards': clean_array(replay_buffer.rewards),
            'next_observations': clean_array(replay_buffer.next_observations)}


def approx_mode(samples: th.Tensor, num_bins: int = 100):
    """
    Approximates the marginal mode of a 2D tensor (N_samples, dim)
    using histogram binning along each dimension.
    """
    assert samples.dim()==2, "Input tensor must be 2D (N_samples, dim)"
    _, D = samples.shape
    modes = []

    for d in range(D):
        col = samples[:, d]
        min_val, max_val = col.min(), col.max()
        bins = th.linspace(min_val, max_val, steps=num_bins + 1)
        bin_indices = th.bucketize(col, bins)
        counts = th.bincount(bin_indices, minlength=num_bins + 2)
        mode_bin = th.argmax(counts)
        if 0 < mode_bin < len(bins):
            mode_val = (bins[mode_bin - 1] + bins[mode_bin]) / 2
        else:
            mode_val = bins[min(mode_bin, len(bins) - 1)]

        modes.append(mode_val)
    return th.stack(modes)