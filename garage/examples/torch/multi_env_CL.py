#!/usr/bin/env python3
"""PEARL ML1 example."""
import click
import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))

from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv, normalize
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import CLMETA
from garage.torch.algos.pearl import PEARLWorker
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import (CLContextConditionedPolicy,
                                   TanhGaussianMLPPolicy)
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer
from garage.envs.multi_env_wrapper import MultiEnvWrapper

from garage.envs import normalize, PointEnv


# @click.command()
# @click.option('--num_epochs', default=1000)
# @click.option('--num_train_tasks', default=50)
# @click.option('--encoder_hidden_size', default=200)
# @click.option('--net_size', default=300)
# @click.option('--num_steps_per_epoch', default=4000)
# @click.option('--num_initial_steps', default=4000)
# @click.option('--num_steps_prior', default=750)
# @click.option('--num_extra_rl_steps_posterior', default=750)
# @click.option('--batch_size', default=256)
# @click.option('--embedding_batch_size', default=64)
# @click.option('--embedding_mini_batch_size', default=64)
# @click.option('--max_episode_length', default=200)
@wrap_experiment
def CL_point_env(ctxt=None,
                             seed=1,
                             num_epochs=20,
                             num_train_tasks=2,
                             num_test_tasks=20,
                             latent_size=5,
                             encoder_hidden_size=256,
                             net_size=256,
                             n_negative_samples=12,
                             meta_batch_size=32,
                             num_steps_per_epoch=15,
                             num_initial_steps=1200,
                             num_tasks_sample=5,
                             num_CL_steps_per_policy=150,
                             num_policy_steps_per_CL=25,
                             num_steps_prior=500,
                             num_extra_rl_steps_posterior=400,
                             context_lr=3e-3,
                             batch_size=256,
                             embedding_batch_size=64,
                             embedding_mini_batch_size=64,
                             reward_scale=10.,
                             max_episode_length=200,
                             use_gpu=False):
    """Train PEARL with ML1 environments.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        num_epochs (int): Number of training epochs.
        num_train_tasks (int): Number of tasks for training.
        latent_size (int): Size of latent context vector.
        encoder_hidden_size (int): Output dimension of dense layer of the
            context encoder.
        net_size (int): Output dimension of a dense layer of Q-function and
            value function.
        meta_batch_size (int): Meta batch size.
        num_steps_per_epoch (int): Number of iterations per epoch.
        num_initial_steps (int): Number of transitions obtained per task before
            training.
        num_tasks_sample (int): Number of random tasks to obtain data for each
            iteration.
        num_steps_prior (int): Number of transitions to obtain per task with
            z ~ prior.
        num_extra_rl_steps_posterior (int): Number of additional transitions
            to obtain per task with z ~ posterior that are only used to train
            the policy and NOT the encoder.
        batch_size (int): Number of transitions in RL batch.
        embedding_batch_size (int): Number of transitions in context batch.
        embedding_mini_batch_size (int): Number of transitions in mini context
            batch; should be same as embedding_batch_size for non-recurrent
            encoder.
        reward_scale (int): Reward scale.
        use_gpu (bool): Whether or not to use GPU for training.

    """
    set_seed(seed)
    encoder_hidden_sizes = (encoder_hidden_size, encoder_hidden_size,
                            encoder_hidden_size)

    env_sampler = SetTaskSampler(
        PointEnv,
        wrapper=lambda env, _: normalize(
            env))
    env = env_sampler.sample(num_train_tasks)
    test_env_sampler = SetTaskSampler(
        PointEnv,
        wrapper=lambda env, _: normalize(
            env))

    trainer = Trainer(ctxt)

    # instantiate networks
    augmented_env = CLMETA.augment_env_spec(env[0](), latent_size)
    qf = ContinuousMLPQFunction(env_spec=augmented_env,
                                hidden_sizes=[net_size, net_size, net_size])

    vf_env = CLMETA.get_env_spec(env[0](), latent_size, 'vf')
    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    inner_policy = TanhGaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size])

    sampler = LocalSampler(agents=None,
                           envs=env[0](),
                           max_episode_length=env[0]().spec.max_episode_length,
                           n_workers=1,
                           worker_class=PEARLWorker)

    clmeta = CLMETA(
        env=env,
        policy_class=CLContextConditionedPolicy,
        encoder_class=MLPEncoder,
        inner_policy=inner_policy,
        qf=qf,
        vf=vf,
        sampler=sampler,
        context_lr=context_lr,
        use_information_bottleneck=False,
        num_train_tasks=num_train_tasks,
        num_test_tasks=num_test_tasks,
        latent_dim=latent_size,
        n_negative_samples=n_negative_samples,
        encoder_hidden_sizes=encoder_hidden_sizes,
        test_env_sampler=test_env_sampler,
        meta_batch_size=meta_batch_size,
        num_steps_per_epoch=num_steps_per_epoch,
        num_initial_steps=num_initial_steps,
        num_tasks_sample=num_tasks_sample,
        num_steps_prior=num_steps_prior,
        num_policy_steps_per_CL=num_policy_steps_per_CL,
        num_CL_steps_per_policy=num_CL_steps_per_policy,
        num_extra_rl_steps_posterior=num_extra_rl_steps_posterior,
        batch_size=batch_size,
        embedding_batch_size=embedding_batch_size,
        embedding_mini_batch_size=embedding_mini_batch_size,
        reward_scale=reward_scale,
    )

    set_gpu_mode(use_gpu, gpu_id=0)
    if use_gpu:
        clmeta.to()

    trainer.setup(algo=clmeta, env=env[0]())

    trainer.train(n_epochs=num_epochs, batch_size=batch_size)


CL_point_env()
