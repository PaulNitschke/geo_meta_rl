"""PEARL and PEARLWorker in Pytorch.

Code is adapted from https://github.com/katerakelly/oyster.
"""

import copy

import akro
from dowel import logger, tabular
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from garage import EnvSpec, InOutSpec, StepType, TimeStep
from garage.experiment import MetaEvaluator
from garage.np.algos import MetaRLAlgorithm
from garage.replay_buffer import PathBuffer
from garage.sampler import DefaultWorker
from garage.torch import global_device
from garage.torch._functions import np_to_torch, zero_optim_grads
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import CLContextConditionedPolicy


class GEOMeta(MetaRLAlgorithm):
    r"""A Meta-RL algorithm that trains the task embedding via Contrastive Learning, otherwise adapted from Pearl.

    PEARL, which stands for Probablistic Embeddings for Actor-Critic
    Reinforcement Learning, is an off-policy meta-RL algorithm. It is built
    on top of SAC using two Q-functions and a value function with an addition
    of an inference network that estimates the posterior :math:`q(z \| c)`.
    The policy is conditioned on the latent variable Z in order to adpat its
    behavior to specific tasks.

    Args:
        env (list[Environment]): Batch of sampled environment updates(
            EnvUpdate), which, when invoked on environments, will configure
            them with new tasks.
        policy_class (type): Class implementing
            :pyclass:`~ContextConditionedPolicy`
        encoder_class (garage.torch.embeddings.ContextEncoder): Encoder class
            for the encoder in context-conditioned policy.
        inner_policy (garage.torch.policies.Policy): Policy.
        qf (torch.nn.Module): Q-function.
        vf (torch.nn.Module): Value function.
        sampler (garage.sampler.Sampler): Sampler.
        num_train_tasks (int): Number of tasks for training.
        num_test_tasks (int or None): Number of tasks for testing.
        latent_dim (int): Size of latent context vector.
        encoder_hidden_sizes (list[int]): Output dimension of dense layer(s) of
            the context encoder.
        test_env_sampler (garage.experiment.SetTaskSampler): Sampler for test
            tasks.
        policy_lr (float): Policy learning rate.
        qf_lr (float): Q-function learning rate.
        vf_lr (float): Value function learning rate.
        context_lr (float): Inference network learning rate.
        policy_mean_reg_coeff (float): Policy mean regulation weight.
        policy_std_reg_coeff (float): Policy std regulation weight.
        policy_pre_activation_coeff (float): Policy pre-activation weight.
        soft_target_tau (float): Interpolation parameter for doing the
            soft target update.
        kl_lambda (float): KL lambda value.
        optimizer_class (type): Type of optimizer for training networks.
        use_information_bottleneck (bool): False means latent context is
            deterministic.
        use_next_obs_in_context (bool): Whether or not to use next observation
            in distinguishing between tasks.
        meta_batch_size (int): Meta batch size.
        num_steps_per_epoch (int): Number of iterations per epoch.
        num_initial_steps (int): Number of transitions obtained per task before
            training.
        num_tasks_sample (int): Number of random tasks to obtain data for each
            iteration.
        num_steps_prior (int): Number of transitions to obtain per task with
            z ~ prior.
        num_steps_posterior (int): Number of transitions to obtain per task
            with z ~ posterior.
        num_extra_rl_steps_posterior (int): Number of additional transitions
            to obtain per task with z ~ posterior that are only used to train
            the policy and NOT the encoder.
        batch_size (int): Number of transitions in RL batch.
        embedding_batch_size (int): Number of transitions in context batch.
        embedding_mini_batch_size (int): Number of transitions in mini context
            batch; should be same as embedding_batch_size for non-recurrent
            encoder.
        discount (float): RL discount factor.
        replay_buffer_size (int): Maximum samples in replay buffer.
        reward_scale (int): Reward scale.
        update_post_train (int): How often to resample context when obtaining
            data during training (in episodes).

    """

    # pylint: disable=too-many-statements
    def __init__(
            self,
            env,
            inner_policy,
            qf,
            vf,
            sampler,
            *,  # Mostly numbers after here.
            latent_dim,
            num_train_tasks,
            num_test_tasks=None,
            encoder_hidden_sizes,
            test_env_sampler,
            n_negative_samples,
            weight_embedding_loss_continuity = None,
            policy_class=CLContextConditionedPolicy,
            encoder_class=MLPEncoder,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            context_lr=3E-4,
            policy_mean_reg_coeff=1E-3,
            policy_std_reg_coeff=1E-3,
            policy_pre_activation_coeff=0.,
            soft_target_tau=0.005,
            kl_lambda=.1,
            optimizer_class=torch.optim.Adam,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            meta_batch_size=64,
            num_steps_per_epoch=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=0,
            num_extra_rl_steps_posterior=100,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            update_post_train=1):

        self._env = env
        self._qf1 = qf
        self._qf2 = copy.deepcopy(qf)
        self._vf = vf
        self._num_train_tasks = num_train_tasks
        self._latent_dim = latent_dim

        self._policy_mean_reg_coeff = policy_mean_reg_coeff
        self._policy_std_reg_coeff = policy_std_reg_coeff
        self._policy_pre_activation_coeff = policy_pre_activation_coeff
        self._soft_target_tau = soft_target_tau
        self._kl_lambda = kl_lambda
        self._use_information_bottleneck = use_information_bottleneck
        self._use_next_obs_in_context = use_next_obs_in_context
        self._n_negative_samples = n_negative_samples
        self._weight_embedding_loss_continuity = weight_embedding_loss_continuity

        self._meta_batch_size = meta_batch_size
        self._num_steps_per_epoch = num_steps_per_epoch
        self._num_initial_steps = num_initial_steps
        self._num_tasks_sample = num_tasks_sample
        self._num_steps_prior = num_steps_prior
        self._num_steps_posterior = num_steps_posterior
        self._num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self._batch_size = batch_size
        self._embedding_batch_size = embedding_batch_size
        self._embedding_mini_batch_size = embedding_mini_batch_size
        self._discount = discount
        self._replay_buffer_size = replay_buffer_size
        self._reward_scale = reward_scale
        self._update_post_train = update_post_train
        self._task_idx = None
        self._single_env = env[0]()
        self.max_episode_length = self._single_env.spec.max_episode_length

        # Hard-code tasks to their ground-truth embedding. Only valid for point environment.
        self._env_copy = copy.deepcopy(self._env)
        self._envidx_to_embeddings = {}
        for idx_env, _ in enumerate(self._env):
            self._envidx_to_embeddings[idx_env] = self._map_task_index_to_embedding(idx_env)


        self._sampler = sampler

        self._is_resuming = False

        if num_test_tasks is None:
            num_test_tasks = test_env_sampler.n_tasks
        if num_test_tasks is None:
            raise ValueError('num_test_tasks must be provided if '
                             'test_env_sampler.n_tasks is None')

        worker_args = dict(deterministic=True, accum_context=True)

        hard_coded_embeddings=[emb.unsqueeze(0) for emb in list(self._envidx_to_embeddings.values())]
        self._train_evaluator = MetaEvaluator(test_tasks=env, 
                                              worker_class=PEARLWorker, 
                                              worker_args=worker_args, 
                                              n_test_tasks=num_train_tasks,
                                              prefix="EvaluationTrain",
                                            #   hard_coded_embeddings=hard_coded_embeddings
                                              )
        self._train_evaluator2 = MetaEvaluator(test_tasks=env, 
                                              worker_class=PEARLWorker, 
                                              worker_args=worker_args, 
                                              n_test_tasks=num_train_tasks,
                                              prefix="EvaluationTrain2",
                                              hard_coded_embeddings=hard_coded_embeddings
                                              )
        # self._test_evaluator = MetaEvaluator(test_task_sampler=test_env_sampler,
        #                                 worker_class=PEARLWorker,
        #                                 worker_args=worker_args,
        #                                 n_test_tasks=num_test_tasks,
        #                                 prefix="EvaluationTest")

        encoder_spec = self.get_env_spec(self._single_env, latent_dim,
                                         'encoder')
        encoder_in_dim = int(np.prod(encoder_spec.input_space.shape))
        encoder_out_dim = int(np.prod(encoder_spec.output_space.shape))
        # encoder_out_dim = encoder_in_dim #TODO, hard-coded for point environment
        context_encoder = encoder_class(input_dim=encoder_in_dim,
                                        output_dim=encoder_out_dim,
                                        hidden_sizes=encoder_hidden_sizes)
        
        class IdentityEncoder():
            """Identity encoder for debugging purposes."""

            def __init__(self, latent_dim):
                self.output_dim=latent_dim

            def forward(self, x):
                return x

            def reset(self):
                """Resets the encoder. This method is a placeholder for compatibility."""
                pass
            
        context_encoder = IdentityEncoder(latent_dim=self._latent_dim)

        self._policy = policy_class(
            latent_dim=latent_dim,
            context_encoder=context_encoder,
            policy=inner_policy,
            use_information_bottleneck=use_information_bottleneck,
            use_next_obs=use_next_obs_in_context)

        # buffer for training RL update
        self._replay_buffers = {
            i: PathBuffer(replay_buffer_size)
            for i in range(num_train_tasks)
        }

        self._context_replay_buffers = {
            i: PathBuffer(replay_buffer_size)
            for i in range(num_train_tasks)
        }

        self.target_vf = copy.deepcopy(self._vf)
        self.vf_criterion = torch.nn.MSELoss()

        self._policy_optimizer = optimizer_class(
            self._policy.networks[1].parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self._qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self._qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self._vf.parameters(),
            lr=vf_lr,
        )
        # self.context_optimizer = optimizer_class(
        #     self._policy.networks[0].parameters(),
        #     lr=context_lr,
        # )

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        data = self.__dict__.copy()
        del data['_replay_buffers']
        del data['_context_replay_buffers']
        return data

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): unpickled state.

        """
        self.__dict__.update(state)
        self._replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(self._num_train_tasks)
        }

        self._context_replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(self._num_train_tasks)
        }
        self._is_resuming = True

    def train(self, trainer):
        """Obtain samples, train, and evaluate for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`Trainer..step_epochs()`, which provides services
                such as snapshotting and sampler control.

        """
        self._previous_task_embedding = None
        self._previous_z = None
        for _ in trainer.step_epochs():
            epoch = trainer.step_itr / self._num_steps_per_epoch

            # obtain initial set of samples from all train tasks
            if epoch == 0 or self._is_resuming:
                for idx in range(self._num_train_tasks):
                    self._task_idx = idx #HERE
                    self._obtain_samples(trainer, epoch,
                                         self._num_initial_steps, np.inf)
                    self._is_resuming = False

            # obtain samples from random tasks
            for _ in range(self._num_tasks_sample):
                idx = np.random.randint(self._num_train_tasks)
                self._task_idx = idx
                self._context_replay_buffers[idx].clear()
                # obtain samples with z ~ prior
                if self._num_steps_prior > 0:
                    self._obtain_samples(trainer, epoch, self._num_steps_prior,
                                         np.inf)
                # obtain samples with z ~ posterior
                if self._num_steps_posterior > 0:
                    self._obtain_samples(trainer, epoch,
                                         self._num_steps_posterior,
                                         self._update_post_train)
                # obtain extras samples for RL training but not encoder
                if self._num_extra_rl_steps_posterior > 0:
                    self._obtain_samples(trainer,
                                         epoch,
                                         self._num_extra_rl_steps_posterior,
                                         self._update_post_train,
                                         add_to_enc_buffer=False)

            # if epoch==0: #TODO, now always pre-training
            logger.log('Pre-Training Encoder...')
            if self._num_train_tasks > 1:
                indices = np.random.choice(range(self._num_train_tasks),
                            self._meta_batch_size)
                cl_loss_converged=False
                idx_cont_loss = 0
                while not cl_loss_converged and idx_cont_loss < 250:
                    self._optimize_contrastive_loss(indices)
                    if idx_cont_loss % 50 == 0: #TODO, this check is only valid if the global CL optimum (e.g. orthogonal embeddings) can be achieved. need to change this later.
                        cl_loss_converged = self._did_cl_loss_converge(indices)
                    idx_cont_loss += 1
                logger.log(f'Pre-Training Encoder Done after {idx_cont_loss} steps...')

            logger.log('Training...')
            self._train_once()
        
            trainer.step_itr += 1

            # evaluate
            logger.log('Evaluating...')
            self._policy.reset_belief()
            self._train_evaluator.evaluate(self)
            self._train_evaluator2.evaluate(self)
            # self._test_evaluator.evaluate(self)

    def _train_once(self):
        """Perform one iteration of training."""
        N_STEPS = 50
        for _ in range(self._num_steps_per_epoch):
            indices = np.random.choice(range(self._num_train_tasks),
                                       self._meta_batch_size)
            
            #Train Embedding.
            if self._num_train_tasks >1:
                prev_cont_loss = torch.inf
                idx_cont_loss = 0
                cl_loss_converged=False
                while not cl_loss_converged and idx_cont_loss<200: #TODO, changed from 200
                    cont_loss=self._optimize_contrastive_loss(indices)
                    if torch.abs(cont_loss-prev_cont_loss)<1e-3:
                        cl_loss_converged=True
                        idx_cont_loss = 0
                    prev_cont_loss = cont_loss
                    idx_cont_loss += 1


            #Train Policy.
            for _ in range(N_STEPS):
                self._optimize_policy(indices)

    def _optimize_contrastive_loss(self, indices):
        """Perform one iteration of training the embedding function via Contrastive Learning."""
        if self._num_train_tasks==1:
            return
        context = self._sample_context(indices)
        self._policy.infer_posterior(context)
        # self.current_z = F.normalize(self._policy.z, p=2, dim=1)
        self.current_z = self._policy.z
        pos_context = self._sample_context(indices)
        self._policy.infer_posterior(pos_context)
        # pos_z = F.normalize(self._policy.z, p=2, dim=1)
        pos_z = self._policy.z

        zero_optim_grads(self.context_optimizer)
        neg_z_list = []
        for _, index_task in enumerate(indices):
            _current_neg_indeces_task = [i for i in range(self._num_train_tasks) if i != index_task]
            neg_task_indices=np.random.choice(_current_neg_indeces_task, self._n_negative_samples)
            neg_context = self._sample_context(neg_task_indices)
            self._policy.infer_posterior(neg_context)
            neg_z_list.append(self._policy.z)
        neg_z = torch.stack(neg_z_list, dim=0)
        cont_loss = self._policy.compute_contrastive_loss(self.current_z, pos_z, neg_z)

        # Penalty to ensure that the embedding doesn't change too much between episodes
        if self._previous_z is not None and self._weight_embedding_loss_continuity is not None:
            breakpoint = True
            _embedding_loss_continuity = -torch.cosine_similarity(self.current_z, self._previous_z).mean()

            total_cont_loss = cont_loss + self._weight_embedding_loss_continuity * _embedding_loss_continuity
            wandb.log({
                "EmbTraining/TotalContrastiveLoss": total_cont_loss.item(),
                "EmbTraining/ContrastiveLoss": cont_loss.item(),
                "EmbTraining/EmbeddingLossContinuity": _embedding_loss_continuity.item()
            })
        else:
            total_cont_loss = cont_loss
            wandb.log({
                "EmbTraining/TotalContrastiveLoss": total_cont_loss.item(),
                "EmbTraining/MeanContrastiveLoss": cont_loss.item()
            })

        self._previous_z = self.current_z.detach()


        # Update the embedding
        total_cont_loss.backward(retain_graph=True)
        self.context_optimizer.step()

        # Logging
        mean_task_embeddings = self._compute_mean_embedding(indices)
        self._did_cl_loss_converge(indices)
        if self._previous_task_embedding is not None:

            # Logs the similarity between the current embedding and the embedding from the previous episode.
            wandb.log({
                "EmbEpisodeConsistency/Mean (higher is better)": torch.cosine_similarity(mean_task_embeddings, self._previous_task_embedding).mean().item(),
                "EmbEpisodeConsistency/Max (higher is better)": torch.cosine_similarity(mean_task_embeddings, self._previous_task_embedding).max().item(),
                "EmbEpisodeConsistency/Min (higher is better)": torch.cosine_similarity(mean_task_embeddings, self._previous_task_embedding).min().item(),
                })
            
            # Logs the mean embeddings of each task
            for idx, embedding in enumerate(mean_task_embeddings):
                table_name = f'Emb/task_{idx}_embeddings'
                columns= [f'embedding_{i}' for i in range(embedding.shape[0])]
                table=wandb.Table(data=[embedding.detach().cpu().numpy().tolist()], columns=columns)
                wandb.log(
                    {table_name: table}
                )
                wandb.run.summary[table_name] = table

        self._previous_task_embedding = mean_task_embeddings

        return cont_loss

    def _compute_mean_embedding(self, indices):
        """Computes the mean embedding of each task from context. Returns tensor of shape (num_tasks, latent_dim). Mostly used for debugging and visualization"""
        indices=torch.tensor(indices)
        _mean_task_embeddings = torch.zeros(self._num_train_tasks, self._latent_dim)
        sample_counts = torch.zeros(self._num_train_tasks, dtype=torch.float)
        _mean_task_embeddings.index_add_(0, indices, self.current_z)
        sample_counts.index_add_(0, indices, torch.ones_like(indices, dtype=torch.float))
        mean_task_embeddings = (_mean_task_embeddings.T / sample_counts).T
        self.pairwise_cosine_similarity = torch.mm(F.normalize(mean_task_embeddings, p=2, dim=1), F.normalize(mean_task_embeddings, p=2, dim=1).T)
        return mean_task_embeddings

    def _did_cl_loss_converge(self, indices):
        """Check if the contrastive loss has converged. This is true if the embedding vectors of different tasks are diagnoal, e.g. their cosine similarity is -1."""
        self._compute_mean_embedding(indices)
        diagonal_mask = torch.eye(self.pairwise_cosine_similarity.size(0), dtype=torch.bool)
        off_diagonal_elements = self.pairwise_cosine_similarity[~diagonal_mask]
        cl_loss_converged = torch.allclose(off_diagonal_elements, torch.full_like(off_diagonal_elements, -1), atol=0.1)

        wandb.log({
            "EmbSimilarity/Mean (higher is worse)": off_diagonal_elements.mean().item(),
            "EmbSimilarity/Max (higher is worse)": off_diagonal_elements.max().item(),
            "EmbSimilarity/Min (higher is worse)": off_diagonal_elements.min().item(),
            "EmbSimilarity/Std (higher is worse)": off_diagonal_elements.std().item(),
        })
        
        return cl_loss_converged
    
    def _map_task_index_to_embedding(self, idx):
        """Map task index to ground-truth embeddings. Only valid for Point Environment."""
        return torch.tensor(self._env_copy[idx]._make_env().reset()[1]["goal"], dtype=torch.float32)

    def _optimize_policy(self, indices):
        """Perform algorithm optimizing.

        Args:
            indices (list): Tasks used for training.

        """
        num_tasks = len(indices)
        context = self._sample_context(indices)
        num_tasks = len(indices)
        # context = self._sample_context(indices)
        # Hard coded task indices instead of sampling
        context = np.array(indices).reshape(num_tasks, 1)
        context = np.repeat(context, self._embedding_batch_size, axis=1)
        context = context[:,:, np.newaxis]
        context = np_to_torch(context)
        X, N, _ = context.shape
        flattened_input = context.view(-1)
        mapped_values = torch.stack([self._envidx_to_embeddings[int(key.item())] for key in flattened_input])
        context = mapped_values.view(X, N, 2)
        context = F.normalize(context, p=2, dim=2) #embeddings are normalized to the unit circle first before we feed them into the policy...

        # clear context and reset belief of policy
        self._policy.reset_belief(num_tasks=num_tasks)

        # data shape is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self._sample_data(indices)
        policy_outputs, task_z = self._policy(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        # assert np.isclose(task_z.detach().cpu().numpy(), np.array([-0.8671, 0.4981]), atol=1e-3).all() or np.isclose(task_z.detach().cpu().numpy(), np.array([0, 0]), atol=1e-3).all(), "self.z: {}".format(self.z.detach().cpu().numpy())

        # flatten out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # optimize qf and encoder networks
        q1_pred = self._qf1(torch.cat([obs, actions], dim=1), task_z)
        q2_pred = self._qf2(torch.cat([obs, actions], dim=1), task_z)
        v_pred = self._vf(obs, task_z.detach())

        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)



        zero_optim_grads(self.qf1_optimizer)
        zero_optim_grads(self.qf2_optimizer)

        rewards_flat = rewards.view(self._batch_size * num_tasks, -1)
        rewards_flat = rewards_flat * self._reward_scale
        terms_flat = terms.view(self._batch_size * num_tasks, -1)
        q_target = rewards_flat + (
            1. - terms_flat) * self._discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target)**2) + torch.mean(
            (q2_pred - q_target)**2)
        qf_loss.backward()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        # compute min Q on the new actions
        q1 = self._qf1(torch.cat([obs, new_actions], dim=1), task_z.detach())
        q2 = self._qf2(torch.cat([obs, new_actions], dim=1), task_z.detach())
        min_q = torch.min(q1, q2)

        # optimize vf
        v_target = min_q - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        zero_optim_grads(self.vf_optimizer)
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # optimize policy
        log_policy_target = min_q
        policy_loss = (log_pi - log_policy_target).mean()

        mean_reg_loss = self._policy_mean_reg_coeff * (policy_mean**2).mean()
        std_reg_loss = self._policy_std_reg_coeff * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self._policy_pre_activation_coeff * (
            (pre_tanh_value**2).sum(dim=1).mean())
        policy_reg_loss = (mean_reg_loss + std_reg_loss +
                           pre_activation_reg_loss)
        policy_loss = policy_loss + policy_reg_loss

        zero_optim_grads(self._policy_optimizer)
        policy_loss.backward()
        self._policy_optimizer.step()

        wandb.log({
            'PolicyTraining/QfLoss': qf_loss.item(),
            'PolicyTraining/VfLoss': vf_loss.item(),
            'PolicyTraining/PolicyLoss': policy_loss.item(),
            'PolicyTraining/MeanQ1Vals': q1.mean().item(),
            'PolicyTraining/MeanQ2Vals': q2.mean().item(),
            'PolicyTraining/MeanVVals': v_pred.mean().item()
        })

    def _obtain_samples(self,
                        trainer,
                        itr,
                        num_samples,
                        update_posterior_rate,
                        add_to_enc_buffer=True):
        """Obtain samples.

        Args:
            trainer (Trainer): Trainer.
            itr (int): Index of iteration (epoch).
            num_samples (int): Number of samples to obtain.
            update_posterior_rate (int): How often (in episodes) to infer
                posterior of policy.
            add_to_enc_buffer (bool): Whether or not to add samples to encoder
                buffer.

        """
        self._policy.reset_belief() #samples a new context encoding z from the prior distribution, which is standard normal
        total_samples = 0

        if update_posterior_rate != np.inf:
            num_samples_per_batch = (update_posterior_rate *
                                     self.max_episode_length)
        else:
            num_samples_per_batch = num_samples

        while total_samples < num_samples:
            paths = trainer.obtain_samples(itr, num_samples_per_batch,
                                           self._policy,
                                           self._env[self._task_idx])
            total_samples += sum([len(path['rewards']) for path in paths])

            for path in paths:
                p = {
                    'observations':
                    path['observations'],
                    'actions':
                    path['actions'],
                    'rewards':
                    path['rewards'].reshape(-1, 1),
                    'next_observations':
                    path['next_observations'],
                    'dones':
                    np.array([
                        step_type == StepType.TERMINAL
                        for step_type in path['step_types']
                    ]).reshape(-1, 1)
                }
                self._replay_buffers[self._task_idx].add_path(p)

                if add_to_enc_buffer:
                    self._context_replay_buffers[self._task_idx].add_path(p)

            # if update_posterior_rate != np.inf:
            #     context = self._sample_context(self._task_idx)
            #     self._policy.infer_posterior(context) #TODO, update
            if update_posterior_rate != np.inf: #TODO, changed encoding to use hard-coded ground-truth encoding from circle.
                context = self._sample_context(self._task_idx)
                context = np.array(self._task_idx)
                context = np.repeat(context, self._embedding_batch_size, axis=0)
                context = context[:, np.newaxis]
                context = np_to_torch(context)
                flattened_input = context.view(-1)
                mapped_values = torch.stack([self._envidx_to_embeddings[int(key.item())] for key in flattened_input])
                context = mapped_values.view(1, 1, self._embedding_batch_size, 2)
                context = F.normalize(context, p=2, dim=-1)
                self._policy.infer_posterior(context)

    def _sample_data(self, indices):
        """Sample batch of training data from a list of tasks.

        Args:
            indices (list): List of task indices to sample from.

        Returns:
            torch.Tensor: Obervations, with shape :math:`(X, N, O^*)` where X
                is the number of tasks. N is batch size.
            torch.Tensor: Actions, with shape :math:`(X, N, A^*)`.
            torch.Tensor: Rewards, with shape :math:`(X, N, 1)`.
            torch.Tensor: Next obervations, with shape :math:`(X, N, O^*)`.
            torch.Tensor: Dones, with shape :math:`(X, N, 1)`.

        """
        # transitions sampled randomly from replay buffer
        initialized = False
        for idx in indices:
            batch = self._replay_buffers[idx].sample_transitions(
                self._batch_size)
            if not initialized:
                o = batch['observations'][np.newaxis]
                a = batch['actions'][np.newaxis]
                r = batch['rewards'][np.newaxis]
                no = batch['next_observations'][np.newaxis]
                d = batch['dones'][np.newaxis]
                initialized = True
            else:
                o = np.vstack((o, batch['observations'][np.newaxis]))
                a = np.vstack((a, batch['actions'][np.newaxis]))
                r = np.vstack((r, batch['rewards'][np.newaxis]))
                no = np.vstack((no, batch['next_observations'][np.newaxis]))
                d = np.vstack((d, batch['dones'][np.newaxis]))

        o = np_to_torch(o)
        a = np_to_torch(a)
        r = np_to_torch(r)
        no = np_to_torch(no)
        d = np_to_torch(d)

        return o, a, r, no, d

    def _sample_context(self, indices): #TODO, maybe we need this
        """Sample batch of context from a list of tasks.

        Args:
            indices (list): List of task indices to sample from.

        Returns:
            torch.Tensor: Context data, with shape :math:`(X, N, C)`. X is the
                number of tasks. N is batch size. C is the combined size of
                observation, action, reward, and next observation if next
                observation is used in context. Otherwise, C is the combined
                size of observation, action, and reward.

        """
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]

        initialized = False
        for idx in indices:
            batch = self._context_replay_buffers[idx].sample_transitions(
                self._embedding_batch_size)
            o = batch['observations']
            a = batch['actions']
            r = batch['rewards']
            context = np.hstack((np.hstack((o, a)), r))
            if self._use_next_obs_in_context:
                context = np.hstack((context, batch['next_observations']))

            if not initialized:
                final_context = context[np.newaxis]
                initialized = True
            else:
                final_context = np.vstack((final_context, context[np.newaxis]))

        final_context = np_to_torch(final_context)

        if len(indices) == 1:
            final_context = final_context.unsqueeze(0)

        return final_context

    def _update_target_network(self):
        """Update parameters in the target vf network."""
        for target_param, param in zip(self.target_vf.parameters(),
                                       self._vf.parameters()):
            target_param.data.copy_(target_param.data *
                                    (1.0 - self._soft_target_tau) +
                                    param.data * self._soft_target_tau)

    @property
    def policy(self):
        """Return all the policy within the model.

        Returns:
            garage.torch.policies.Policy: Policy within the model.

        """
        return self._policy

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return self._policy.networks + [self._policy] + [
            self._qf1, self._qf2, self._vf, self.target_vf
        ]

    def get_exploration_policy(self):
        """Return a policy used before adaptation to a specific task.

        Each time it is retrieved, this policy should only be evaluated in one
        task.

        Returns:
            Policy: The policy used to obtain samples that are later used for
                meta-RL adaptation.

        """
        self._policy.reset_belief() #TODO, added this myself because the policy is not reset in the original code
        return self._policy

    def adapt_policy(self, exploration_policy, exploration_episodes):
        """Produce a policy adapted for a task.

        Args:
            exploration_policy (Policy): A policy which was returned from
                get_exploration_policy(), and which generated
                exploration_episodes by interacting with an environment.
                The caller may not use this object after passing it into this
                method.
            exploration_episodes (EpisodeBatch): Episodes to which to adapt,
                generated by exploration_policy exploring the
                environment.

        Returns:
            Policy: A policy adapted to the task represented by the
                exploration_episodes.

        """
        if hasattr(exploration_episodes, "lengths"):
            total_steps = sum(exploration_episodes.lengths)
            o = exploration_episodes.observations
            a = exploration_episodes.actions
            r = exploration_episodes.rewards.reshape(total_steps, 1)
            ctxt = np.hstack((o, a, r)).reshape(1, total_steps, -1)
            context = np_to_torch(ctxt)
        else:
            context = exploration_episodes
        self._policy.infer_posterior(context)

        return self._policy

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        device = device or global_device()
        for net in self.networks:
            net.to(device)

    @classmethod
    def augment_env_spec(cls, env_spec, latent_dim):
        """Augment environment by a size of latent dimension.

        Args:
            env_spec (EnvSpec): Environment specs to be augmented.
            latent_dim (int): Latent dimension.

        Returns:
            EnvSpec: Augmented environment specs.

        """
        obs_dim = int(np.prod(env_spec.observation_space.shape))
        action_dim = int(np.prod(env_spec.action_space.shape))
        aug_obs = akro.Box(low=-1,
                           high=1,
                           shape=(obs_dim + latent_dim, ),
                           dtype=np.float32)
        aug_act = akro.Box(low=-1,
                           high=1,
                           shape=(action_dim, ),
                           dtype=np.float32)
        return EnvSpec(aug_obs, aug_act)

    @classmethod
    def get_env_spec(cls, env_spec, latent_dim, module):
        """Get environment specs of encoder with latent dimension.

        Args:
            env_spec (EnvSpec): Environment specification.
            latent_dim (int): Latent dimension.
            module (str): Module to get environment specs for.

        Returns:
            InOutSpec: Module environment specs with latent dimension.

        """
        obs_dim = int(np.prod(env_spec.observation_space.shape))
        action_dim = int(np.prod(env_spec.action_space.shape))
        if module == 'encoder':
            in_dim = obs_dim + action_dim + 1
            out_dim = latent_dim#TODO, this needs to be updated whether we use probabilitstic or deterministic encoding
        elif module == 'vf':
            in_dim = obs_dim
            out_dim = latent_dim
        in_space = akro.Box(low=-1, high=1, shape=(in_dim, ), dtype=np.float32)
        out_space = akro.Box(low=-1,
                             high=1,
                             shape=(out_dim, ),
                             dtype=np.float32)
        if module == 'encoder':
            spec = InOutSpec(in_space, out_space)
        elif module == 'vf':
            spec = EnvSpec(in_space, out_space)

        return spec


class PEARLWorker(DefaultWorker):
    """A worker class used in sampling for PEARL.

    It stores context and resample belief in the policy every step.

    Args:
        seed (int): The seed to use to intialize random number generators.
        max_episode_length(int or float): The maximum length of episodes which
            will be sampled. Can be (floating point) infinity.
        worker_number (int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.
        deterministic (bool): If True, use the mean action returned by the
            stochastic policy instead of sampling from the returned action
            distribution.
        accum_context (bool): If True, update context of the agent.

    Attributes:
        agent (Policy or None): The worker's agent.
        env (Environment or None): The worker's environment.

    """

    def __init__(self,
                 *,
                 seed,
                 max_episode_length,
                 worker_number,
                 deterministic=False,
                 accum_context=False):
        self._deterministic = deterministic
        self._accum_context = accum_context
        self._episode_info = None
        super().__init__(seed=seed,
                         max_episode_length=max_episode_length,
                         worker_number=worker_number)

    def start_episode(self):
        """Begin a new episode."""
        self._eps_length = 0
        self._prev_obs, self._episode_info = self.env.reset()

    def step_episode(self):
        """Take a single time-step in the current episode.

        Returns:
            bool: True iff the episode is done, either due to the environment
            indicating termination of due to reaching `max_episode_length`.

        """
        if self._eps_length < self._max_episode_length:
            a, agent_info = self.agent.get_action(self._prev_obs)
            if self._deterministic:
                a = agent_info['mean']
            a, agent_info = self.agent.get_action(self._prev_obs)
            es = self.env.step(a)
            self._observations.append(self._prev_obs)
            self._env_steps.append(es)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            self._eps_length += 1

            if self._accum_context:
                s = TimeStep.from_env_step(env_step=es,
                                           last_observation=self._prev_obs,
                                           agent_info=agent_info,
                                           episode_info=self._episode_info)
                self.agent.update_context(s)
            if not es.last:
                self._prev_obs = es.observation
                return False
        self._lengths.append(self._eps_length)
        self._last_observations.append(self._prev_obs)
        return True

    def rollout(self):
        """Sample a single episode of the agent in the environment.

        Returns:
            EpisodeBatch: The collected episode.

        """
        self.agent.sample_from_belief()
        self.start_episode()
        while not self.step_episode():
            pass
        return self.collect_episode()
