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



class ExponentialLinearRegressor(th.nn.Module):
    """
    # Learns a matrix W such that exp(W) \cdot X ≈ Y.
    """
    def __init__(self, input_dim: int, seed:int):
        super().__init__()
        th.manual_seed(seed)
        self.W = th.nn.Parameter(th.randn(input_dim, input_dim))

    def forward(self, X: th.Tensor) -> th.Tensor:
        """
        X: (N, n) input matrix
        Returns: (N, n) prediction
        """
        W_exp = th.matrix_exp(self.W)
        return (W_exp @ X.T).T  # Apply from left, return (N, n)

    def loss(self, X: th.Tensor, Y: th.Tensor) -> th.Tensor:
        """
        Computes MSE loss between predicted and true outputs.
        """
        Y_pred = self.forward(X)
        return th.mean((Y - Y_pred) ** 2)

    def fit(self, X: th.Tensor, Y: th.Tensor, lr: float = 1e-2,
            epochs: int = 1000, verbose: bool = False):
        """
        Fits the model parameters to minimize MSE between exp(W) * X and Y.
        """
        optimizer = th.optim.Adam([self.W], lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.loss(X, Y)
            loss.backward()
            optimizer.step()
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
        return self.W.data.clone()