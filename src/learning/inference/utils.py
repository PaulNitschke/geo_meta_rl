import numpy as np
from typing import Tuple

def sample_traj_of_env(environment, 
                       pi: callable,
                       n_steps: int=100) -> Tuple[np.array, np.array, np.array]:
    """
    Sample a trajectory from an environment using a policy pi.
    Args:
        environment: The environment to sample from.
        pi: The trained model to use for action prediction.
        n_steps (int): The number of steps to sample in the trajectory.
    Returns:
        Tuple[np.array, np.array, np.array]: A tuple containing:
            - trajectory: An array of observations.
            - actions: An array of actions taken.
            - rewards: An array of rewards received.
    """
    obs = environment.reset()
    trajectory, rewards, actions = [], [], []
    trajectory.append(obs.copy())

    for _ in range(n_steps):
        action, _ = pi.predict(obs, deterministic=True)
        obs, reward, done, _ = environment.step(action)

        actions.append(action)
        trajectory.append(obs.copy())
        rewards.append(reward)

        if done:
            break

    return np.array(trajectory), np.array(rewards), np.array(actions)


def sample_meta_traj(train_envs: list,
                     test_envs: list,
                     train_pis: list[callable],
                     test_pis: list[callable]) -> dict[str, dict[str, np.array]]:
    """
    Sample trajectories from training and testing environments using a policy.
    Args:
        train_envs (list): List of training environments.
        test_envs (list): List of testing environments.
        train_pis (list[callable]): List of policies for training environments.
        test_pis (list[callable]): List of policies for testing environments.
    Returns:
        dict[str, dict[str, np.array]]: A dictionary containing:
            - 'train': A dictionary with keys 'states', 'actions', and 'rewards' for training data.
            - 'test': A dictionary with keys 'states', 'actions', and 'rewards' for testing data.
    """
    assert len(train_envs) == len(train_pis), "Number of training environments must match number of training policies."
    assert len(test_envs) == len(test_pis), "Number of testing environments must match number of testing policies."

    states_train, actions_train, rewards_train = [], [], []
    states_test, actions_test, rewards_test = [], [], []

    for env, pi in zip(train_envs, train_pis):
        traj, actions, rewards = sample_traj_of_env(env, pi)
        states_train.append(traj)
        actions_train.append(actions)
        rewards_train.append(rewards)

    for env, pi in zip(test_envs, test_pis):
        traj, actions, rewards = sample_traj_of_env(env, pi)
        states_test.append(traj)
        actions_test.append(actions)
        rewards_test.append(rewards)

    return {
        'train': {
            'states': states_train,
            'actions': actions_train,
            'rewards': rewards_train
        },
        'test': {
            'states': states_test,
            'actions': actions_test,
            'rewards': rewards_test
        }
    }