# #!/usr/bin/env python3
# """This is an example to train multiple tasks with TRPO algorithm."""
# from garage import wrap_experiment
from garage.envs import normalize, PointEnv
# from garage.envs.multi_env_wrapper import MultiEnvWrapper
# from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
# from garage.sampler import RaySampler
# from garage.torch.algos import TRPO
# from garage.torch.policies import GaussianMLPPolicy
# from garage.torch.policies import GaussianMLPPolicy
# from garage.trainer import TFTrainer


# @wrap_experiment
# def multi_env_trpo(ctxt=None, seed=1):
#     """Train TRPO on two different PointEnv instances.

#     Args:
#         ctxt (garage.experiment.ExperimentContext): The experiment
#             configuration used by Trainer to create the snapshotter.
#         seed (int): Used to seed the random number generator to produce
#             determinism.

#     """
#     set_seed(seed)
#     with TFTrainer(ctxt) as trainer:
#         env1 = normalize(PointEnv(goal=(-1., 0.), max_episode_length=100))
#         env2 = normalize(PointEnv(goal=(1., 0.), max_episode_length=100))
#         env = MultiEnvWrapper([env1, env2])

#         policy = GaussianMLPPolicy(env_spec=env.spec)

#         baseline = LinearFeatureBaseline(env_spec=env.spec)

#         sampler = RaySampler(agents=policy,
#                              envs=env,
#                              max_episode_length=env.spec.max_episode_length,
#                              is_tf_worker=True)

#         algo = TRPO(env_spec=env.spec,
#                     policy=policy,
#                     baseline=baseline,
#                     sampler=sampler,
#                     discount=0.99,
#                     gae_lambda=0.95,
#                     lr_clip_range=0.2,
#                     policy_ent_coeff=0.0)

#         trainer.setup(algo, env)
#         trainer.train(n_epochs=40, batch_size=2048, plot=False)


# multi_env_trpo()

#!/usr/bin/env python3
"""This is an example to train TRPO on ML1 Push environment."""
# pylint: disable=no-value-for-parameter
import click
import torch

from garage import wrap_experiment
from garage.envs import normalize
from garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import MetaWorldTaskSampler
from garage.sampler import RaySampler
from garage.torch.algos import TRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer


# @click.command()
# @click.option('--seed', default=1)
# @click.option('--epochs', default=40)
# @click.option('--batch_size', default=2048)
@wrap_experiment(snapshot_mode='all')
def multi_env_trpo(ctxt, seed, epochs, batch_size):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        batch_size (int): Number of environment steps in one batch.

    """
    set_seed(seed)
    env1 = normalize(PointEnv(goal=(-1., 0.), max_episode_length=100))
    env2 = normalize(PointEnv(goal=(1., 0.), max_episode_length=100))
    env3 = normalize(PointEnv(goal=(0, -1), max_episode_length=100))
    env = MultiEnvWrapper([env1, env2, env3])

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length)
    

    algo = TRPO(env_spec=env.spec,
                policy=policy,
                value_function=value_function,
                sampler=sampler,
                discount=0.99,
                policy_ent_coeff=0.0,
                gae_lambda=0.95)

    trainer = Trainer(ctxt)
    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs, batch_size=batch_size)


multi_env_trpo(seed=1,
               epochs=80,
               batch_size=2048)