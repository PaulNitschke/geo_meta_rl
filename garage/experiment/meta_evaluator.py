"""Evaluator which tests Meta-RL algorithms on test environments."""

from dowel import logger, tabular
import wandb
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO

from garage import EpisodeBatch, log_multitask_performance
from garage.experiment.deterministic import get_seed
from garage.sampler import DefaultWorker, LocalSampler, WorkerFactory


class MetaEvaluator:
    """Evaluates Meta-RL algorithms on test environments.

    Args:
        test_task_sampler (TaskSampler): Sampler for test
            tasks. To demonstrate the effectiveness of a meta-learning method,
            these should be different from the training tasks.
        test_tasks (list[Type[gym.Env]]): List of test tasks. Can be inserted instead of test_task_sampler.
        n_test_tasks (int or None): Number of test tasks to sample each time
            evaluation is performed. Note that tasks are sampled "without
            replacement". If None, is set to `test_task_sampler.n_tasks`.
        n_exploration_eps (int): Number of episodes to gather from the
            exploration policy before requesting the meta algorithm to produce
            an adapted policy.
        n_test_episodes (int): Number of episodes to use for each adapted
            policy. The adapted policy should forget previous episodes when
            `.reset()` is called.
        prefix (str): Prefix to use when logging. Defaults to MetaTest. For
            example, this results in logging the key 'MetaTest/SuccessRate'.
            If not set to `MetaTest`, it should probably be set to `MetaTrain`.
        test_task_names (list[str]): List of task names to test. Should be in
            an order consistent with the `task_id` env_info, if that is
            present.
        worker_class (type): Type of worker the Sampler should use.
        worker_args (dict or None): Additional arguments that should be
            passed to the worker.
        return_task_embeddings (bool): If True, returns the task embeddings
            along with the adapted episodes.
        log_wandb (bool): If True, logs the rollouts and task embeddings to wandb.
        hard_coded_embeddings (np.ndarray): If not None, uses these embeddings instead of the ones from the policy.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self,
                 *,
                 test_task_sampler=None,
                 test_tasks=None,
                 n_exploration_eps=10,
                 n_test_tasks=None,
                 n_test_episodes=1,
                 prefix='MetaTest',
                 test_task_names=None,
                 worker_class=DefaultWorker,
                 worker_args=None,
                 return_task_embeddings=False,
                 log_wandb=True,
                 hard_coded_embeddings=None):
        self._test_task_sampler = test_task_sampler
        self._test_tasks= test_tasks
        self._worker_class = worker_class
        if worker_args is None:
            self._worker_args = {}
        else:
            self._worker_args = worker_args
        if n_test_tasks is None:
            n_test_tasks = test_task_sampler.n_tasks
        self._n_test_tasks = n_test_tasks
        self._n_test_episodes = n_test_episodes
        self._n_exploration_eps = n_exploration_eps
        self._eval_itr = 0
        self._prefix = prefix
        self._test_task_names = test_task_names
        self._test_sampler = None
        self._max_episode_length = None
        self._return_task_embeddings = return_task_embeddings
        self._log_wandb = log_wandb
        self._hard_coded_embeddings = hard_coded_embeddings

    def evaluate(self, algo, test_episodes_per_task=None):
        """Evaluate the Meta-RL algorithm on the test tasks.

        Args:
            algo (MetaRLAlgorithm): The algorithm to evaluate.
            test_episodes_per_task (int or None): Number of episodes per task.

        """
        if test_episodes_per_task is None:
            test_episodes_per_task = self._n_test_episodes
        adapted_episodes = []
        task_embeddings = []
        logger.log('Sampling for adapation and meta-testing...')
        if self._test_task_sampler is not None:
            env_updates = self._test_task_sampler.sample(self._n_test_tasks)
        else:
            env_updates = self._test_tasks

        if self._test_sampler is None:
            env = env_updates[0]()
            self._max_episode_length = env.spec.max_episode_length
            self._test_sampler = LocalSampler.from_worker_factory(
                WorkerFactory(seed=get_seed(),
                              max_episode_length=self._max_episode_length,
                              n_workers=1,
                              worker_class=self._worker_class,
                              worker_args=self._worker_args),
                agents=algo.get_exploration_policy(),
                envs=env)
        breakpoint=True
        for idx_env_up, env_up in enumerate(env_updates):

            policy = algo.get_exploration_policy()
            eps = EpisodeBatch.concatenate(*[
                self._test_sampler.obtain_samples(self._eval_itr, 1, policy,
                                                env_up)
                for _ in range(self._n_exploration_eps)
            ])
            adapted_policy = algo.adapt_policy(policy, eps)
            if self._hard_coded_embeddings is not None:
                adapted_policy.z = self._hard_coded_embeddings[idx_env_up]
            task_embedding = algo.policy.z.detach().cpu().numpy()
            task_embeddings.append(task_embedding)
            adapted_eps = self._test_sampler.obtain_samples(
                self._eval_itr,
                test_episodes_per_task * self._max_episode_length,
                adapted_policy)
            adapted_episodes.append(adapted_eps)
        logger.log('Finished meta-testing...')

        if self._test_task_names is not None:
            name_map = dict(enumerate(self._test_task_names))
        else:
            name_map = None

        with tabular.prefix(self._prefix + '/' if self._prefix else ''):
            log_multitask_performance(
                self._eval_itr,
                EpisodeBatch.concatenate(*adapted_episodes),
                getattr(algo, 'discount', 1.0),
                name_map=name_map)
        self._eval_itr += 1

        #wandb logging
        if self._log_wandb:
            rollouts_img = self.rollout_plotter(adapted_episodes)
            wandb.log({f'{self._prefix}/rollouts': wandb.Image(rollouts_img)})
            for idx, embedding in enumerate(task_embeddings):
                print("embedding: ", embedding)
                table_name = f'{self._prefix}/task_{idx}_embeddings'
                columns= [f'embedding_{i}' for i in range(embedding.shape[1])]
                table=wandb.Table(data=embedding, columns=columns)
                wandb.log(
                    {table_name: table}
                )
                wandb.run.summary[table_name] = table

        if self._return_task_embeddings:
            return adapted_episodes, task_embeddings
        else:
            return adapted_episodes

    def rollout_plotter(self, trajectories): #TODO, this should be an input to the class
        # Create a new figure
        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')
        
        # Plot circle
        circle = plt.Circle((0, 0), 2, color='r', fill=False, linestyle='--')
        ax.add_artist(circle)
        
        # Set plot limits
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        
        # Plot trajectories
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        for idx_task, task_traj in enumerate(trajectories):
            ax.plot(task_traj.observations[:, 0], task_traj.observations[:, 1], label=f"Task {idx_task}", color=colors[idx_task])
            ax.scatter(task_traj.env_infos["task"][0]["goal"][0], task_traj.env_infos["task"][0]["goal"][1], 
                    color=colors[idx_task], marker='x', label=f'Goal Task {idx_task}')
        
        # Plot origin
        ax.scatter(0, 0, color='black', marker='x', label='Origin')
        
        # Add legend
        ax.legend(ncol=3)
        
        # Save plot to a BytesIO buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)  # Close the figure to free memory
        buf.seek(0)  # Reset the buffer position to the start
        
        # Convert buffer to PIL Image
        image = Image.open(buf)
        return image