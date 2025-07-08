import pickle

import numpy as np
import torch as th
import gym
from gym import spaces
import scipy.linalg

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


class matrixLogarithm:
    def __init__(self):
        """Implements a differentiable matrix logarithm in PyTorch.

        Usage:
        logm = matrixLogarithm()
        A = th.randn(3, 3, dtype=th.float32, requires_grad=True)
        log_A= logm.apply(A)
        log_A.requires_grad

        Source: https://github.com/pytorch/pytorch/issues/9983
        """
        self._logm_func = self._LogmFunction.apply

    def apply(self, A: th.Tensor) -> th.Tensor:
        return self._logm_func(A)

    @staticmethod
    def _adjoint(A, E, f):
        A_H = A.mH.to(E.dtype)
        n = A.size(0)
        M = th.zeros(2 * n, 2 * n, dtype=E.dtype, device=E.device)
        M[:n, :n] = A_H
        M[n:, n:] = A_H
        M[:n, n:] = E
        return f(M)[:n, n:].to(A.dtype)

    @staticmethod
    def _logm_scipy(A):
        return th.from_numpy(scipy.linalg.logm(A.cpu(), disp=False)[0]).to(A.device)

    class _LogmFunction(th.autograd.Function):
        @staticmethod
        def forward(ctx, A):
            assert A.ndim == 2 and A.size(0) == A.size(1), "A must be a square matrix"
            assert A.dtype in (th.float32, th.float64, th.complex64, th.complex128), "Unsupported dtype"
            ctx.save_for_backward(A)
            return matrixLogarithm._logm_scipy(A)

        @staticmethod
        def backward(ctx, G):
            A, = ctx.saved_tensors
            return matrixLogarithm._adjoint(A, G, matrixLogarithm._logm_scipy)