import warnings
import time
import logging
from typing import Literal, List, Tuple, Optional

from tqdm import tqdm
import numpy as np
import torch
import wandb

from ..initialization import identity_init_neural_net, ExponentialLinearRegressor


class HereditaryGeometryDiscovery():

    def __init__(self,
                 tasks_ps: List[torch.tensor],
                 tasks_frameestimators: List[callable],
                 kernel_dim: int,
                 encoder: callable=None,
                 decoder: callable=None,
                 seed:int=42,
                 log_lg_inits_how:Literal['log_linreg', 'random']= 'log_linreg',
                 batch_size:int=64,
                 bandwidth:float=0.5,

                 update_chart_every_n_steps:int=1,
                 lasso_coef_lgs: Optional[float] = 0.5,
                 lasso_coef_encoder_decoder: Optional[float] = 0.005,
                 learning_rate_left_actions:float=0.0035,
                 learning_rate_generator:float=0.00035,
                 learning_rate_encoder:float=0.00035,
                 learning_rate_decoder:float=0.00035,
                 n_steps_pretrain_left_actions:int=10_000,

                 log_wandb:bool=False,
                 task_specifications:list=None,
                 use_oracle_rotation_kernel:bool=False,
                 oracle_generator: torch.tensor=None,
                 learn_left_actions:bool=True,
                 learn_generator:bool=True,
                 learn_encoder_decoder:bool=False,
                ):
        """Hereditary Geometry Discovery.
        This class implements hereditary symmetry discovery.

        Args:
        - tasks_ps: list of tensors, each of shape (n_samples, n).
        - tasks_frameestimators: list of callables, given a batch of samples from the respective task, returns the frame of the kernel at these samples.
        - kernel_dim: dimension of the kernel.
        - encoder: callable, encodes points into latent space.
        - decoder: callable, decodes points from latent space to ambient space.
        - seed: random seed for reproducibility.
        - lg_inits_how: how to initialize left actions, one of ['random', 'mode', 'zeros']. Mode computes the mode of tasks_ps and then fits a linear regression between the modes.
        - batch_size: number of samples to use for optimization.
        - bandwidth: bandwidth for the kernel density frame estimators.
        - lasso_coef_lgs: regularization weight for the lasso regularizer on the left actions, if None, no regularization is applied.

        - learn_left_actions: whether to learn left actions, only use for debugging.
        - learn_encoder_decoder: whether to learn the encoder and decoder, only use for debugging.
        - oracle_generator: tensor of shape (d, n, n), the generator to be used for symmetry discovery, only use for debugging.
        - task_specifications: list of dictionaries, each containing the goal of the task, only used for debugging.

        
        Notation:
        - d: kernel dimension.
        - n: ambient dimension.
        - b: batch size.
        - N: number of tasks.
        """

        self.tasks_ps= tasks_ps
        self.tasks_frameestimators=tasks_frameestimators
        self.kernel_dim=kernel_dim
        self.base_task_index=0
        self._log_lg_inits_how=log_lg_inits_how
        self.encoder=encoder
        self.decoder=decoder
        self.batch_size=batch_size
        self.bandwidth=bandwidth
        self.seed=seed

        self._update_chart_every_n_steps=update_chart_every_n_steps
        self._lasso_coef_lgs=lasso_coef_lgs if lasso_coef_lgs is not None else 0.0
        self._lasso_coef_encoder_decoder=lasso_coef_encoder_decoder if lasso_coef_encoder_decoder is not None else 0.0
        self._learning_rate_left_actions=learning_rate_left_actions
        self._learning_rate_generator=learning_rate_generator
        self._learning_rate_encoder=learning_rate_encoder
        self._learning_rate_decoder=learning_rate_decoder
        self._n_steps_pretrain_left_actions=n_steps_pretrain_left_actions

        self._log_wandb=log_wandb
        self._use_oracle_rotation_kernel=use_oracle_rotation_kernel
        self.task_specifications=task_specifications
        self.oracle_generator= oracle_generator
        self._learn_left_actions=learn_left_actions
        self._learn_generator=learn_generator
        self._learn_encoder_decoder=learn_encoder_decoder

        self._validate_inputs()

        logging.info(f"Fitting left-actions: {self._learn_left_actions}")
        logging.info(f"Fitting generator: {self._learn_generator}")
        logging.info(f"Fitting encoder and decoder: {self._learn_encoder_decoder}")


        self.ambient_dim=tasks_ps[0][0,:].shape[0]
        self._n_tasks=len(tasks_ps)
        self._n_samples=tasks_ps[0][:,0].shape[0]
        self.task_idxs = list(range(self._n_tasks))
        self.task_idxs.remove(self.base_task_index)
        self.frame_base_task= self.tasks_frameestimators[self.base_task_index]
        self.frame_i_tasks= [self.tasks_frameestimators[i] for i in self.task_idxs]

        self._losses={}
        self._losses["left_actions"]= [torch.tensor([0])]
        self._losses["left_actions_tasks"]= [torch.tensor([0])]
        self._losses["left_actions_tasks_reg"]= [torch.tensor([0])]

        self._losses["generator"]= [torch.tensor([0])]
        
        self._losses["symmetry"]= [torch.tensor([0])]
        self._losses["reconstruction"]= [torch.tensor([0])]
        self._losses["symmetry_reg"] = [torch.tensor([0])]

        # Optimization variables
        torch.manual_seed(seed)

        if self._learn_left_actions:
            if self._log_lg_inits_how == 'log_linreg':
                self._log_lg_inits=self._init_log_lgs_linear_reg(verbose=False, epochs=2500, log_wandb=self._log_wandb)

            elif self._log_lg_inits_how == 'random':
                self._log_lg_inits = torch.randn(size=(self._n_tasks-1, self.ambient_dim, self.ambient_dim))

            self.log_lgs= torch.nn.Parameter(self._log_lg_inits.clone())
            self.optimizer_lgs = torch.optim.Adam([self.log_lgs],lr=self._learning_rate_left_actions)
            assert self.log_lgs.shape == (self._n_tasks-1, self.ambient_dim, self.ambient_dim), "Log left actions must be of shape (N_tasks-1, n, n)."
        

        self.generator=torch.nn.Parameter(torch.randn(size=(self.kernel_dim, self.ambient_dim, self.ambient_dim))) if self.oracle_generator is None else torch.nn.Parameter(self.oracle_generator.clone())
        assert self.generator.shape == (self.kernel_dim, self.ambient_dim, self.ambient_dim), "Generator must be of shape (d, n, n)."
        self.optimizer_generator=torch.optim.Adam([self.generator], lr=self._learning_rate_generator)


        self.encoder= identity_init_neural_net(self.encoder, tasks_ps=self.tasks_ps, name="encoder", log_wandb=self._log_wandb)
        self.decoder= identity_init_neural_net(self.decoder, tasks_ps=self.tasks_ps, name="decoder", log_wandb=self._log_wandb)
        self.optimizer_encoder=torch.optim.Adam(self.encoder.parameters(), lr=self._learning_rate_encoder)
        self.optimizer_decoder=torch.optim.Adam(self.decoder.parameters(), lr=self._learning_rate_decoder)

    
    def evalute_left_actions(self, 
                             ps: torch.Tensor, 
                             log_lgs: torch.Tensor, 
                             encoder: torch.nn.Module,
                             decoder: torch.nn.Module,
                             track_loss:bool=True) -> float:
        """Computes kernel alignment loss of all left-actions."""
        # 1. Push-forward
        tilde_ps=encoder(ps)
        lgs=torch.linalg.matrix_exp(log_lgs)
        lg_tilde_ps = torch.einsum("Nmn,bn->Nbm", lgs, tilde_ps)
        lg_ps = decoder(lg_tilde_ps)

        # 2. Sample tangent vectors and push-forward tangent vectors
        if not self._use_oracle_rotation_kernel:
            frame_ps=self.frame_base_task.evaluate(ps, bandwidth=self.bandwidth).transpose(-1, -2)
            frames_i_lg_ps = torch.stack([
                self.frame_i_tasks[i].evaluate(lg_ps[i], bandwidth=self.bandwidth)
                for i in range(self._n_tasks-1)], dim=0).transpose(-1, -2)
            
        else:
            # Use oracle rotation kernel
            frame_ps = self.rotation_vector_field(ps, center=self.task_specifications[self.base_task_index]['goal'])
            goals = torch.stack([torch.tensor(self.task_specifications[i]['goal']) for i in self.task_idxs])
            frames_i_lg_ps = torch.stack([
                self.rotation_vector_field(lg_ps[i], center=goals[i])
                for i in range(lg_ps.shape[0])], dim=0)
        lgs_frame_ps = torch.einsum("Nmn,bdn->Nbdm", lgs, frame_ps)


        # 3. Compute orthogonal complement and projection loss.
        _, ortho_frame_i_lg_ps = self._project_onto_vector_subspace(frames_i_lg_ps, lgs_frame_ps)
        _, ortho_lgs_frame_ps = self._project_onto_vector_subspace(lgs_frame_ps, frames_i_lg_ps)
        mean_ortho_comp = lambda vec: torch.norm(vec, dim=(-1)).mean(-1).mean(-1)
        self.task_losses = mean_ortho_comp(ortho_frame_i_lg_ps) + mean_ortho_comp(ortho_lgs_frame_ps)
        self.task_losses_reg = self._lasso_coef_lgs*torch.norm(log_lgs, p=1, dim=(-1)).mean(-1).mean(-1)

        if track_loss:
            self._losses["left_actions_tasks"].append(self.task_losses.detach().cpu().numpy())
            self._losses["left_actions_tasks_reg"].append(self.task_losses_reg.detach().cpu().numpy())
            self._losses["left_actions"].append(self.task_losses.mean().detach().cpu().numpy()) #exclude regularization term from this loss.

        return self.task_losses.mean() + self.task_losses_reg
    

    def evaluate_generator_span(self, 
                                generator: torch.Tensor, 
                                log_lgs: torch.Tensor, 
                                track_loss:bool=True)->float:
        """Evalutes whether all left-actions are inside the span of the generator."""
        _, ortho_log_lgs_generator=self._project_onto_tensor_subspace(log_lgs, generator)
        loss_span=torch.mean(torch.norm(ortho_log_lgs_generator, p="fro",dim=(1,2)),dim=0)

        if track_loss:
            self._losses["generator"].append(loss_span.detach().cpu().numpy())
        return loss_span

    
    def evalute_symmetry(self, 
                         ps: torch.Tensor, 
                         generator:torch.Tensor,
                         encoder:torch.nn.Module,
                         decoder:torch.nn.Module, 
                         track_loss: bool=True)->float:
        """
        Evaluates whether the generator is contained within the kernel distribution of the base task (expressed in the encoder and decoder).
        For stable learning, ensure that encoder and decoder are both identity maps in the beginning.
        """

        def compute_vec_jacobian(f: callable, s: torch.tensor)->torch.tensor:
            """
            Compute a vectorized Jacobian of function f: n -> m over a batch of states s.
            Input: tensor s of shape (b,d,n) where both b and d are batch dimensions and n is the dimension of the data.
            Returns: tensor of shape (b,d,n,m)
            """
            return torch.vmap(torch.vmap(torch.func.jacrev(f)))(s)

        # Let the generator act on the points in latent space.

        tilde_ps=encoder(ps)
        gen_tilde_ps = torch.einsum("dnm,bm->bdn", generator, tilde_ps)
        jac_decoder=compute_vec_jacobian(decoder, gen_tilde_ps)
        gen_ps = torch.einsum("bdmn, bdn->bdm", jac_decoder, gen_tilde_ps)

        # Check symmetry, need to evaluate frame at base points.
        #TODO, using ground truth kernel distribution.
        goal_base = self.task_specifications[self.base_task_index]['goal']
        frame_ps= self.rotation_vector_field(ps, center=goal_base)
        frame_ps=frame_ps.unsqueeze(0)

        _, gen_into_frame = self._project_onto_vector_subspace(gen_ps, frame_ps)

        loss_symmetry= torch.norm(gen_into_frame, dim=(-1)).sum(-1).mean()
        loss_reconstruction= torch.norm(ps - decoder(encoder(ps)), dim=(-1)).mean()
        l1_penalty = lambda model: sum(p.abs().sum() for p in model.parameters())
        loss_reg = self._lasso_coef_encoder_decoder * (l1_penalty(encoder) + l1_penalty(decoder))

        if track_loss:
            self._losses["symmetry"].append(loss_symmetry.detach().cpu().numpy())
            self._losses["reconstruction"].append(loss_reconstruction.detach().cpu().numpy())
            self._losses["symmetry_reg"].append(loss_reg.detach().cpu().numpy())

        return loss_symmetry+loss_reconstruction+loss_reg
        

    def _project_onto_vector_subspace(self, vecs, basis):
        """
        Projects 1-tensors onto a d-dimensional subspace of 1-tensors.#TODO update docstring to extra batch dimension.
        vecs: tensor of shape (N, b, d, n)
        basis: tensor of shape (N, b, d, n)
        Returns:
        - proj_vecs: tensor of shape (N, b, d, n)
        """
        basis_t = basis.transpose(-2, -1)
        G = torch.matmul(basis, basis_t)
        G_inv = torch.linalg.pinv(G)
        P = torch.matmul(basis_t, torch.matmul(G_inv, basis))
        proj_vecs = torch.matmul(vecs, P)
        return proj_vecs, vecs-proj_vecs


    def _project_onto_tensor_subspace(self, tensors: torch.tensor, basis: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        Projects 2-tensors onto a d-dimensional subspace of 2-tensors.
        Args:
        - tensors: torch.tensor of shape (b,n,n), b two-tensors
        - basis: torch.tensor of shape (d,n,n), a d-dimensional vector space of two-tensors, given by its basis.

        Returns: 
        - proj: torch.tensor of shape (b,n,n), the projection of tensors onto the subspace spanned by basis
        - ortho_comp: torch.tensor of shape (b,n,n), the orthogonal complement of tensors with respect to the subspace spanned by basis.
        """
        tensors= tensors.unsqueeze(1)
        basis=basis.unsqueeze(0)
        b, d, _, _ = basis.shape

        G = torch.einsum('bdij,bdij->bd', basis, basis)
        G = G.unsqueeze(-1).expand(-1, -1, d)
        G = G * torch.eye(d, device=basis.device).unsqueeze(0) + torch.einsum('bdi,bdi->bdi', basis.view(b, d, -1), basis.view(b, d, -1)).transpose(1,2)

        G = torch.matmul(basis.view(b, d, -1), basis.view(b, d, -1).transpose(1, 2))
        G_inv = torch.linalg.pinv(G)

        v = tensors.expand(-1, d, -1, -1)
        b_proj = torch.einsum('bdij,bdij->bd', v, basis)

        alpha = torch.matmul(G_inv, b_proj.unsqueeze(-1)).squeeze(-1)
        proj = torch.sum(alpha.unsqueeze(-1).unsqueeze(-1) * basis, dim=1, keepdim=True)
        ortho_comp=tensors-proj
        return proj.squeeze(1), ortho_comp.squeeze(1)

        
    def take_grad_step(self, take_step_geometry:bool=True, take_step_symmetry:bool=True):
        """Takes one gradient step."""
        ps = self.tasks_ps[self.base_task_index][torch.randint(0, self._n_samples, (self.batch_size,))]

        if take_step_geometry:

            # Freeze encoder and decoder.
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False

            self.optimizer_lgs.zero_grad()
            loss_left_action = self.evalute_left_actions(ps=ps, log_lgs=self.log_lgs, encoder=self.encoder, decoder=self.decoder)
            loss_left_action.backward()
            self.optimizer_lgs.step()


        if take_step_symmetry:
            # Freeze follower optimization variables and unfreeze leader optimization variables.
            frozen_lgs = self.log_lgs.detach().clone()
            frozen_lgs.requires_grad_(False)
            for p in self.encoder.parameters():
                p.requires_grad = True
            for p in self.decoder.parameters():
                p.requires_grad = True

            self.optimizer_generator.zero_grad()
            self.optimizer_encoder.zero_grad()
            self.optimizer_decoder.zero_grad()

            loss_span = self.evaluate_generator_span(generator=self.generator, log_lgs=frozen_lgs)
            loss_symmetry = self.evalute_symmetry(ps=ps, generator=self.generator, encoder=self.encoder, decoder=self.decoder)
            (loss_span + loss_symmetry).backward()            
            self.optimizer_generator.step()
            self.optimizer_encoder.step()
            self.optimizer_decoder.step()


    def optimize(self, n_steps:int=1000):
        """Optimizes the left action."""
        self.progress_bar = tqdm(range(n_steps), desc="Inferring left-action")

        step_counter = 0
        for _ in range(self._n_steps_pretrain_left_actions):
            self.take_grad_step(take_step_geometry=True, take_step_symmetry=False)
            step_counter += 1
            if step_counter % 5 == 0:
                self._log_to_wandb(step_counter)
                time.sleep(0.05)

            self._set_progress_bar()

        for _ in self.progress_bar:
            for _ in range(self._update_chart_every_n_steps):
                self.take_grad_step(take_step_geometry=True, take_step_symmetry=False)
                step_counter += 1
                if step_counter % 5 == 0:
                    self._log_to_wandb(step_counter)
                    time.sleep(0.05)

            self.take_grad_step(take_step_geometry=False, take_step_symmetry=True)
            step_counter += 1
            if step_counter % 5 == 0:
                self._log_to_wandb(step_counter)
                time.sleep(0.05)

            self._set_progress_bar()

            if step_counter==n_steps:
                logging.info("Reached maximum number of steps, stopping optimization.")
                break


    def rotation_vector_field(self, p_batch: torch.tensor, center)->torch.tensor:
        """Returns kernel samples at batched points p from a task."""

        _generator=torch.tensor([[0, -1], [1,0]], requires_grad=False, dtype=torch.float32).unsqueeze(0)
        projected_state = p_batch-center
        return torch.einsum("dmn, bn->bdm", _generator, projected_state)
    

    def _validate_inputs(self):
        """Validates user inputs."""
        assert self._log_lg_inits_how in ['random', 'log_linreg'], "_log_lg_inits_how must be one of ['random', 'log_linreg']."
        assert len(self.tasks_ps) == len(self.tasks_frameestimators), "Number of tasks and frame estimators must match."
        assert self.oracle_generator is None if self._learn_generator else True, "If you want to learn the generator, do not provide an oracle generator."
        assert self.oracle_generator is not None if not self._learn_generator else True, "If you do not want to learn the generator, provide an oracle generator."

        if self._learn_encoder_decoder:
            assert self.encoder is not None, "Encoder must be provided to learn symmetry."
            assert self.decoder is not None, "Decoder must be provided to learn symmetry."


    def _init_log_lgs_linear_reg(self, verbose=False, epochs=10000, log_wandb:bool=False):
        """Fits log-linear regressors to initialize left actions."""
        logging.info("Fitting log-linear regressors to initialize left actions.")
        self._log_lg_inits = [
            ExponentialLinearRegressor(input_dim=self.ambient_dim, seed=self.seed, log_wandb=log_wandb).fit(
                X=self.tasks_ps[0], Y=self.tasks_ps[idx_task], verbose=verbose, epochs=epochs
            )
            for idx_task in self.task_idxs]
        logging.info("Finished fitting log-linear regressors to initialize left actions.")       
        return torch.stack(self._log_lg_inits, dim=0)
    

    def _log_to_wandb(self, step:int):
        """Logs losses to weights and biases."""
        if not self._log_wandb:
            return

        wandb.log({
            "train/left_actions/mean": float(self._losses['left_actions'][-1]),
            # "train/left_actions/tasks": float(self._losses['left_actions_tasks'][-1]), #TODO, log task level losses.
            "train/generator": float(self._losses['generator'][-1]),
            "train/symmetry/span": float(self._losses['symmetry'][-1]),
            "train/symmetry/reconstruction": float(self._losses['reconstruction'][-1]),
            "train/regularizers/symmetry": float(self._losses['symmetry_reg'][-1]),
            "train/regularizers/left_actions/lasso": float(self._losses['left_actions_tasks_reg'][-1]),
        }, step=step)


    @property
    def lgs(self):
        return torch.linalg.matrix_exp(self.log_lgs)
    

    @property
    def lgs_inits(self):
        return torch.linalg.matrix_exp(self._log_lg_inits)


    @property
    def losses(self):
        """Returns all losses."""
        return {
            "left_actions": np.array(self._losses["left_actions"][1:]),
            "left_actions_tasks": np.array(self._losses["left_actions_tasks"][1:]),
            "left_actions_tasks_reg": np.array(self._losses["left_actions_tasks_reg"][1:]),
            "generator": np.array(self._losses["generator"][1:]),
            "symmetry": np.array(self._losses["symmetry"][1:]),
            "reconstruction": np.array(self._losses["reconstruction"][1:]),
            "symmetry_reg": np.array(self._losses["symmetry_reg"][1:]),
        }
    

    def _set_progress_bar(self):
        """Updates the progress bar with the current losses."""
        prog_bar_description = ""

        if self._learn_left_actions:
            prog_bar_description += (
                f"Left-Action Loss: {round(self._losses['left_actions'][-1].item(), 3)} | "
                f"Task Losses: {np.round(self._losses['left_actions_tasks'][-1], 3)} | "
                f"Task Losses (reg): {np.round(self._losses['left_actions_tasks_reg'][-1], 3)} | "
            )

        if self._learn_generator:
            prog_bar_description += (
                f"Generator Span Loss: {round(self._losses['generator'][-1].item(), 3)} | "
            )

        if self._learn_encoder_decoder:
            prog_bar_description += (
                f"Symmetry Loss: {round(self._losses['symmetry'][-1].item(), 3)} | "
                f"Reconstruction Loss: {round(self._losses['reconstruction'][-1].item(), 3)} | "
            )

        self.progress_bar.set_description(prog_bar_description.strip(" | "))