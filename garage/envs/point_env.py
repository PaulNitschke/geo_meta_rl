"""Simple 2D environment containing a point and a goal location."""
import math

import akro
import numpy as np

from garage import Environment, EnvSpec, EnvStep, StepType


class PointEnv(Environment):
    """A simple 2D point environment.

    Args:
        goal (np.ndarray): A 2D array representing the goal position
        arena_size (float): The size of arena where the point is constrained
            within (-arena_size, arena_size) in each dimension
        done_bonus (float): A numerical bonus added to the reward
            once the point as reached the goal
        never_done (bool): Never send a `done` signal, even if the
            agent achieves the goal
        max_episode_length (int): The maximum steps allowed for an episode.

    """

    def __init__(self,
                 goal=np.array((1., 1.), dtype=np.float32),
                 arena_size=5.,
                 never_done=False,
                 max_episode_length=100,
                 sigma_noise:float=0.05):
        goal = np.array(goal, dtype=np.float32)
        self._goal = goal
        self._never_done = never_done
        self._arena_size = arena_size
        self._sigma_noise = sigma_noise

        assert ((goal >= -arena_size) & (goal <= arena_size)).all()

        self._step_cnt = 0
        self._max_episode_length = max_episode_length
        self._visualize = False

        self._point = np.zeros_like(self._goal)
        self._task = {'goal': self._goal}
        self._observation_space = akro.Box(low=-np.inf,
                                           high=np.inf,
                                           shape=(2, ),
                                           dtype=np.float32)
        self._action_space = akro.Box(low=-0.1,
                                      high=0.1,
                                      shape=(2, ),
                                      dtype=np.float32)
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space,
                             max_episode_length=max_episode_length)

    @property
    def action_space(self):
        """akro.Space: The action space specification."""
        return self._action_space

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._observation_space

    @property
    def spec(self):
        """EnvSpec: The environment specification."""
        return self._spec

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return [
            'ascii',
        ]

    def reset(self):
        """Reset the environment.

        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of the entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)

        """
        self._point = np.zeros_like(self._goal)
        self._point += np.random.normal(
            loc=0.0, scale=self._sigma_noise, size=self._point.shape)
        dist = np.linalg.norm(self._point - self._goal)

        first_obs = self._point.copy()
        # first_obs = np.concatenate([self._point, (dist, )]).astype(np.float32)
        self._step_cnt = 0

        return first_obs, dict(goal=self._goal)

    def step(self, action):
        """Step the environment.

        Args:
            action (np.ndarray): An action provided by the agent.

        Returns:
            EnvStep: The environment step resulting from the action.

        Raises:
            RuntimeError: if `step()` is called after the environment
            has been
                constructed and `reset()` has not been called.

        """
        if self._step_cnt is None:
            raise RuntimeError('reset() must be called before step()!')

        # enforce action space
        a = action.copy()  # NOTE: we MUST copy the action before modifying it
        a = np.clip(a, self.action_space.low, self.action_space.high)

        # Transition function, additive normally distributed noise
        self._point = np.clip(self._point + a, -self._arena_size,
                              self._arena_size)
        self._point += np.random.normal(
            loc=0.0, scale=self._sigma_noise, size=self._point.shape)

        if self._visualize:
            print(self.render('ascii'))

        dist = np.linalg.norm(self._point - self._goal)
        succ = dist < np.linalg.norm(self.action_space.low)

        # dense reward
        reward = -dist

        # Type conversion
        if not isinstance(reward, float):
            reward = float(reward)

        # sometimes we don't want to terminate
        done = succ and not self._never_done

        obs = self._point.copy()
        # obs = np.concatenate([self._point, (dist, )]).astype(np.float32)

        self._step_cnt += 1

        step_type = StepType.get_step_type(
            step_cnt=self._step_cnt,
            max_episode_length=self._max_episode_length,
            done=done)

        if step_type in (StepType.TERMINAL, StepType.TIMEOUT):
            self._step_cnt = None

        return EnvStep(env_spec=self.spec,
                       action=action,
                       reward=reward,
                       observation=obs,
                       env_info={
                           'task': self._task,
                           'success': succ
                       },
                       step_type=step_type)

    def render(self, mode):
        """Renders the environment.

        Args:
            mode (str): the mode to render with. The string must be present in
                `self.render_modes`.

        Returns:
            str: the point and goal of environment.

        """
        return f'Point: {self._point}, Goal: {self._goal}'

    def visualize(self):
        """Creates a visualization of the environment."""
        self._visualize = True
        print(self.render('ascii'))

    def close(self):
        """Close the env."""

    # pylint: disable=no-self-use
    def sample_tasks(self, num_tasks,
                     mode: str="uniform",
                     chart: np.array=None):
        """Sample a list of `num_tasks` tasks. Tasks are uniformly distributed on circle with radius 1.

        Args:
            num_tasks (int): Number of tasks to sample.

        Returns:
            list[dict[str, np.ndarray]]: A list of "tasks", where each task is
                a dictionary containing a single key, "goal", mapping to a
                point in 2D space.

        """
        # How to sample goals on the task distribution.
        if mode=="uniform":
            angles = np.random.uniform(0, 2 * math.pi, num_tasks)
        elif mode=="linspace":
            angles = np.linspace(0, 2 * math.pi, num_tasks, endpoint=False)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        

        # 1. Define task distribution
        chart = np.eye(2) if chart is None else chart
        coordinates = np.matmul(chart, np.vstack([np.cos(angles), np.sin(angles)]))
        x = coordinates[0, :]
        y = coordinates[1, :]


        goals = [np.array([x[i], y[i]]) for i in range(num_tasks)]
        tasks = [{'goal': goal} for goal in goals]

        # goals = np.random.uniform(-2, 2, size=(num_tasks, 2))
        # tasks = [{'goal': goal} for goal in goals]
        return tasks

    def set_task(self, task):
        """Reset with a task.

        Args:
            task (dict[str, np.ndarray]): A task (a dictionary containing a
                single key, "goal", which should be a point in 2D space).

        """
        self._task = task
        self._goal = task['goal']
