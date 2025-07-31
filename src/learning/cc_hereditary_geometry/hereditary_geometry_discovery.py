import os
import logging
from typing import Literal, List, Tuple, Optional
import time
import copy
import warnings

from tqdm import tqdm
import numpy as np
import torch
import wandb

from ...utils import DenseNN
from .initialization import identity_init_neural_net, ExponentialLinearRegressor


class HereditaryGeometryDiscovery():

    def __init__(self,
                 tasks_ps: List[torch.tensor],
                 tasks_frameestimators: List[callable],
                 enc_geo_net_sizes: list[int],
                 enc_sym_net_sizes: list[int],
                 eval_sym_in_follower: bool,

                 kernel_dim: int,
                 update_chart_every_n_steps:int,
                 eval_span_how:Literal['weights', 'ortho_comp'],
                 log_lg_inits_how:Literal['log_linreg', 'random'],

                 batch_size:int,
                 lr_lgs:float,
                 lr_gen:float,
                 lr_chart:float,
                 lasso_coef_lgs: Optional[float],
                 lasso_coef_generator: Optional[float],
                 lasso_coef_encoder_decoder: Optional[float],
                 n_epochs_pretrain_log_lgs: int,
                 n_epochs_init_neural_nets: int,

                 seed:int,
                 bandwidth:float,
                 log_wandb:bool,
                 log_wandb_gradients:bool,
                 save_every:int,
                 
                 use_oracle_frames:bool,
                 oracle_frames:list[callable],
                 oracle_encoder_geo: torch.nn.Module,
                 oracle_decoder_geo: torch.nn.Module,
                 oracle_encoder_sym: torch.nn.Module,
                 oracle_decoder_sym: torch.nn.Module,
                 oracle_generator: torch.tensor,
                 save_dir:str
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
        self.oracle_generator= oracle_generator
        self.encoder_geo=DenseNN(enc_geo_net_sizes)
        self.encoder_sym=DenseNN(enc_sym_net_sizes)
        self.base_task_index=0
        self._log_wandb_every=500
        self._n_epochs_pretrain_log_lgs=n_epochs_pretrain_log_lgs
        self._n_epochs_init_neural_nets=n_epochs_init_neural_nets

        self.kernel_dim=kernel_dim
        self._update_chart_every_n_steps=update_chart_every_n_steps
        self._eval_span_how=eval_span_how
        self._log_lg_inits_how=log_lg_inits_how

        self.batch_size=batch_size
        self._lr_lgs=lr_lgs
        self._lr_gen=lr_gen
        self._lr_chart=lr_chart
        self._lasso_coef_lgs=lasso_coef_lgs
        self._lasso_coef_generator = lasso_coef_generator
        self._lasso_coef_encoder_decoder=lasso_coef_encoder_decoder
        
        self.seed=seed
        self.bandwidth=bandwidth
        self._log_wandb=log_wandb
        self._log_wandb_gradients=log_wandb_gradients
        self._save_every= save_every
        
        self._eval_sym_in_follower = eval_sym_in_follower
        self._use_oracle_frames=use_oracle_frames
        self._oracle_frames=oracle_frames
        self._save_dir=save_dir
        self._oracle_encoder_sym= oracle_encoder_sym
        self._oracle_decoder_sym= oracle_decoder_sym
        self._oracle_encoder_geo= oracle_encoder_geo
        self._oracle_decoder_geo= oracle_decoder_geo

        self._validate_inputs()

        self._global_step_wandb=0
        self.ambient_dim=tasks_ps[0][0,:].shape[0]
        self._n_tasks=len(tasks_ps)
        self._n_samples=tasks_ps[0][:,0].shape[0]
        self.task_idxs = list(range(self._n_tasks))
        self.task_idxs.remove(self.base_task_index)

        if self._use_oracle_frames:
            warnings.warn("Using oracle kernel frames.")
            self.frame_base_task= self._oracle_frames[self.base_task_index]
            self.frame_i_tasks= [self._oracle_frames[i] for i in self.task_idxs]            
        else:    
            self.frame_base_task= self.tasks_frameestimators[self.base_task_index]
            self.frame_i_tasks= [self.tasks_frameestimators[i] for i in self.task_idxs]

        # Store losses and diagnostics.
        self._losses, self._diagnostics={}, {}
        _losses_names=["left_actions","left_actions_tasks_reg",
                       "generator_span","generator_weights","reconstruction_geo","generator_reg",
                       "symmetry","reconstruction_sym","symmetry_reg"]
        _diagnostics_names= ["cond_num_generator", "frob_norm_generator",
                                "encoder_loss_oracle_geo", "decoder_loss_oracle_geo",
                                "encoder_loss_oracle_sym", "decoder_loss_oracle_sym"]
        self._losses = {name: [0.0] for name in _losses_names}
        self._diagnostics = {name: [0.0] for name in _diagnostics_names}
        self._losses["left_actions_tasks"] = [np.zeros(self._n_tasks-1, dtype=np.float32)]

        torch.manual_seed(seed)

    
    def evaluate_left_actions(self, 
                             ps: torch.Tensor, 
                             log_lgs: torch.Tensor, 
                             encoder_geo: torch.nn.Module,
                             decoder_geo: torch.nn.Module,
                             track_loss:bool=True) -> float:
        """Computes kernel alignment loss of all left-actions."""
        # 1. Push-forward
        lgs=torch.linalg.matrix_exp(log_lgs.param)

        def encoded_left_action(ps):
            #TODO, this is currently only for 1-D lie groups, the computed jacobian is of shape (b,N,m,n).
            """Helper function that lets the exponential of the log-left action act on ps, represented in the current chart.
            Used to compute Jacobians."""
            tilde_ps=encoder_geo(ps)
            tilde_ps = ps
            #batch-dimension of ps is dropped here as vmap processes the tensors one-by-one.
            lg_tilde_ps = torch.einsum("Nmn,n->Nm", lgs, tilde_ps) 
            return decoder_geo(lg_tilde_ps)

        tilde_ps=encoder_geo(ps)
        lg_tilde_ps = torch.einsum("Nmn,bn->Nbm", lgs, tilde_ps)
        lg_ps = decoder_geo(lg_tilde_ps)

        # 2. Sample tangent vectors and push-forward tangent vectors
        frame_ps= self.frame_base_task(ps)
        frames_i_lg_ps = torch.stack([self.frame_i_tasks[idx_task](lg_ps[idx_task]) for idx_task in self.task_idxs], dim=0)

        
        jac_lgs = self.compute_vec_jacobian(encoded_left_action, ps)
        lgs_frame_ps = torch.einsum("bNmn,bdn->Nbdm", jac_lgs, frame_ps)


        # 3. Compute orthogonal complement and projection loss.
        _, ortho_frame_i_lg_ps = self._project_onto_vector_subspace(frames_i_lg_ps, lgs_frame_ps)
        _, ortho_lgs_frame_ps = self._project_onto_vector_subspace(lgs_frame_ps, frames_i_lg_ps)
        mean_ortho_comp = lambda vec: torch.norm(vec, dim=(-1)).mean(-1).mean(-1)
        self.task_losses = mean_ortho_comp(ortho_frame_i_lg_ps) + mean_ortho_comp(ortho_lgs_frame_ps)
        self.task_losses_reg = self._lasso_coef_lgs*torch.norm(log_lgs.param, p=1, dim=(-1)).mean(-1).mean(-1)

        loss_reconstruction_geo= torch.linalg.vector_norm(ps - decoder_geo(encoder_geo(ps)), dim=-1).mean()
        if track_loss:
            self._losses["left_actions_tasks"].append(self.task_losses.detach().cpu().numpy())
            self._losses["left_actions_tasks_reg"].append(self.task_losses_reg.detach().cpu().numpy())
            self._losses["left_actions"].append(self.task_losses.mean().detach().cpu().numpy())
            self._losses["reconstruction_geo"].append(loss_reconstruction_geo.detach().cpu().numpy())

        # Compare the current encoder to an oracle encoder, only used for debugging.
        if self._oracle_encoder_geo is not None and self._oracle_decoder_geo is not None:
            with torch.no_grad():
                encoder_loss_oracle_geo = torch.norm(self._oracle_encoder_geo(ps) - encoder_geo(ps), dim=-1).sum(0)
                decoder_loss_oracle_geo = torch.norm(self._oracle_decoder_geo(ps) - decoder_geo(ps), dim=-1).sum(0)
            if track_loss:
                self._diagnostics["encoder_loss_oracle_geo"].append(encoder_loss_oracle_geo.detach().cpu().numpy())
                self._diagnostics["decoder_loss_oracle_geo"].append(decoder_loss_oracle_geo.detach().cpu().numpy())

        return self.task_losses.mean() + loss_reconstruction_geo + self.task_losses_reg
    

    def evaluate_generator_span(self, 
                                log_lgs: torch.Tensor, 
                                generator: torch.nn.Module, 
                                track_loss:bool=True)->float:
        """
        Evalutes whether all log-left-actions are inside the span of the generator: log_lgs \in span(generator).
        log_lgs are frozen in this loss function (and hence a detached tensor).
        """

        if self._eval_span_how == "weights":
            log_lgs_hat = torch.einsum("Nd,dmn->Nmn", self.weights_lgs_to_gen.param, generator)
            loss_weights = torch.mean((log_lgs_hat - log_lgs) ** 2)
            loss_reg=self._lasso_coef_generator*torch.sum(torch.abs(generator)) + self._lasso_coef_lgs*torch.sum(torch.abs(self.weights_lgs_to_gen.param))
            loss = loss_weights + loss_reg

            with torch.no_grad():
                _, ortho_log_lgs_generator=self._project_onto_tensor_subspace(log_lgs, generator)
                loss_span=torch.mean(torch.linalg.matrix_norm(ortho_log_lgs_generator),dim=0)

        elif self._eval_span_how == "ortho_comp":
            _, ortho_log_lgs_generator=self._project_onto_tensor_subspace(log_lgs, generator)
            loss_span=torch.mean(torch.linalg.matrix_norm(ortho_log_lgs_generator),dim=0)
            loss_reg=self._lasso_coef_generator*torch.sum(torch.abs(generator))
            loss_weights = torch.tensor(0.0)
            loss = loss_span + loss_reg

        if track_loss:
            self._losses["generator_span"].append(loss_span.detach().cpu().numpy())
            self._losses["generator_weights"].append(loss_weights.detach().cpu().numpy())
            self._losses["generator_reg"].append(loss_reg.detach().cpu().numpy())

        return loss


    def compute_vec_jacobian(self, f: callable, s: torch.tensor)->torch.tensor:
        """
        Compute a vectorized Jacobian of function f: n -> m over a batch of states s.
        Input: tensor s of shape (b,n) where b is a batch dimension and n is the dimension of the data.
        Returns: tensor of shape (b,n,n)
        """
        return torch.vmap(torch.func.jacrev(f))(s)

    
    def compute_vec_vec_jacobian(self, f: callable, s: torch.tensor)->torch.tensor:
        """
        Compute a vectorized Jacobian of function f: n -> m over a batch of states s.
        Input: tensor s of shape (b,d,n) where both b and d are batch dimensions and n is the dimension of the data.
        Returns: tensor of shape (b,d,n,m)
        """
        return torch.vmap(torch.vmap(torch.func.jacrev(f)))(s)


    def evaluate_symmetry(self, 
                         ps: torch.Tensor, 
                         generator:torch.Tensor,
                         encoder_geo:torch.nn.Module,
                         decoder_geo:torch.nn.Module,
                         encoder_sym:torch.nn.Module,
                         decoder_sym:torch.nn.Module,
                         track_loss: bool=True)->float:
        """
        Evaluates whether the generator is contained within the kernel distribution of the base task (expressed in symmetry and geometry charts).
        The following objects are frozen:
        - generator, as we are interested in finding a chart that symmetrizes f given the current geometry.
        - encoder_geo, decoder_geo, as these are used to represent the geometry.

        The following objects are trainable:
        - encoder_sym, decoder_sym, as these are used to represent the symmetry.
        """

        # Let the generator act on the points in latent space. First encode via symmetry space and then via geometry space.
        tilde_ps=encoder_geo(encoder_sym(ps))
        gen_tilde_tilde_ps = torch.einsum("dnm,bm->bdn", generator, tilde_ps)
        jac_decoder_geo=self.compute_vec_vec_jacobian(decoder_geo, gen_tilde_tilde_ps)
        gen_tilde_ps = torch.einsum("bdmn, bdn->bdm", jac_decoder_geo, gen_tilde_tilde_ps)
        jac_decoder_sym=self.compute_vec_vec_jacobian(decoder_sym, gen_tilde_ps)
        gen_ps = torch.einsum("bdmn, bdn->bdm", jac_decoder_sym, gen_tilde_ps)

        # Check symmetry, need to evaluate frame at base points.
        frame_ps=self.frame_base_task(ps)
        frame_ps=frame_ps.unsqueeze(0)

        # Evaluate how far out of the kernel distribution the generator is.
        _, gen_into_frame = self._project_onto_vector_subspace(gen_ps, frame_ps)

        loss_symmetry= torch.linalg.vector_norm(gen_into_frame, dim=-1).mean(-1).mean()
        loss_reconstruction_sym= torch.linalg.vector_norm(ps - decoder_sym(encoder_sym(ps)), dim=-1).mean()
        loss_reg = self._lasso_coef_encoder_decoder * (self._l1_penalty(encoder_sym) + self._l1_penalty(decoder_sym))


        # Compare the current encoder to an oracle encoder, only used for debugging.
        if self._oracle_encoder_sym is not None and self._oracle_decoder_sym is not None:
            with torch.no_grad():
                encoder_loss_oracle_sym = torch.norm(self._oracle_encoder_sym(ps) - encoder_sym(ps), dim=-1).sum(0)
                decoder_loss_oracle_sym = torch.norm(self._oracle_decoder_sym(ps) - decoder_sym(ps), dim=-1).sum(0)
            if track_loss:
                self._diagnostics["encoder_loss_oracle_sym"].append(encoder_loss_oracle_sym.detach().cpu().numpy())
                self._diagnostics["decoder_loss_oracle_sym"].append(decoder_loss_oracle_sym.detach().cpu().numpy())

        if track_loss:
            self._losses["symmetry"].append(loss_symmetry.detach().cpu().numpy())
            self._losses["reconstruction_sym"].append(loss_reconstruction_sym.detach().cpu().numpy())
            self._losses["symmetry_reg"].append(loss_reg.detach().cpu().numpy())


        return loss_symmetry+loss_reconstruction_sym+loss_reg
    

    def _l1_penalty(self, model):
        """L1 Penalty on model parameters."""
        return sum(p.abs().sum() for p in model.parameters())
        

    def _project_onto_vector_subspace(self, vecs, basis):
        """
        Projects 1-tensors onto a d-dimensional subspace of 1-tensors.
        vecs: tensor of shape (N, b, d, n)
        basis: tensor of shape (N, b, d, n)
        Returns:
        - proj_vecs: tensor of shape (N, b, d, n)
        - ortho_vecs: tensor of shape (N, b, d, n)
        """
        basis_t = basis.transpose(-2, -1)
        G = torch.matmul(basis, basis_t)
        G_inv = torch.linalg.pinv(G)
        P = torch.matmul(basis_t, torch.matmul(G_inv, basis))
        proj_vecs = torch.matmul(vecs, P)
        return proj_vecs, vecs-proj_vecs
    

    def _project_onto_tensor_subspace(self, tensors: torch.tensor, basis:torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        Projects 2-tensors onto a d-dimensional subspace of 2-tensors.
        Args:
        - tensors: torch.tensor of shape (b, n, n), b two-tensors
        - basis: torch.tensor of shape (d, n, n), a d-dimensional vector space of two-tensors, given by its basis.

        Returns: 
        - proj: torch.tensor of shape (b, n, n), the projection of tensors onto the subspace spanned by basis
        - ortho_comp: torch.tensor of shape (b, n, n), the orthogonal complement of tensors with respect to the subspace spanned by basis.
        """
        b,n,_= tensors.shape
        d,_,_= basis.shape
        tensors_flat=tensors.reshape(b, n*n)
        basis_flat=basis.reshape(d, n*n)


        proj_vecs_flat, ortho_vecs_flat = self._project_onto_vector_subspace(tensors_flat, basis_flat)
        proj = proj_vecs_flat.reshape(b, n, n)
        ortho_comp = ortho_vecs_flat.reshape(b, n, n)

        with torch.no_grad():
            s = torch.linalg.svdvals(basis_flat)
            self._diagnostics["cond_num_generator"].append((s.max()/s.min()).item())
            self._diagnostics["frob_norm_generator"].append(torch.mean(torch.linalg.matrix_norm(basis)).item())

        return proj, ortho_comp
    
        
    def take_step_left_actions(self):
        """Update the left actions under a frozen chart."""
        ps = self.tasks_ps[self.base_task_index][torch.randint(0, self._n_samples, (self.batch_size,))] #TODO, think about which points to use here.

        for p in self.log_lgs.parameters(): p.requires_grad = True
        for p in self.encoder_geo.parameters(): p.requires_grad = False
        for p in self.decoder_geo.parameters(): p.requires_grad = False

        # 1. Step left-actions, left-action loss is independent of generator.
        self.optimizer_lgs.zero_grad()        
        loss_left_action = self.evaluate_left_actions(ps=ps, 
                                                      log_lgs=self.log_lgs, 
                                                      encoder_geo=self.encoder_geo, 
                                                      decoder_geo=self.decoder_geo)
        loss_left_action.backward()
        self.optimizer_lgs.step()


    def take_step_chart_geo(self):
        """Update the geometry chart under frozen leftactions."""
        ps = self.tasks_ps[self.base_task_index][torch.randint(0, self._n_samples, (self.batch_size,))] #TODO, probably need to sample from all tasks for this to globally train encoder/ decoder.

        for p in self.log_lgs.parameters(): p.requires_grad = False
        for p in self.encoder_geo.parameters(): p.requires_grad = True
        for p in self.decoder_geo.parameters(): p.requires_grad = True

        self.optim_encoder_geo.zero_grad()
        self.optim_decoder_geo.zero_grad()
        loss_left_action = self.evaluate_left_actions(ps=ps,
                                                      log_lgs=self.log_lgs, 
                                                      encoder_geo=self.encoder_geo, 
                                                      decoder_geo=self.decoder_geo)
        loss_left_action.backward()
        self.optim_encoder_geo.step()
        self.optim_decoder_geo.step()


    def take_step_generator(self):
        """Steps the generator given the log left actions."""        

        self.optimizer_generator.zero_grad()
        generator_normed = self.generator.param / torch.linalg.matrix_norm(self.generator.param)

        loss_span = self.evaluate_generator_span(generator=generator_normed, log_lgs=self.log_lgs.param)
        
        loss_span.backward() 
        self.optimizer_generator.step()


    def take_step_chart_sym(self):
        """
        Learns a symmetry chart given a geometry given by the generator encoded in the geometry chart.
        The generator and the geometry chart are both frozen.
        """
        #TODO, probably need to sample from all tasks for this to globally train encoder/ decoder. What does globally mean here?
        # We generally won't expect the chart to generalize to new parts of the state/ action spaces as we can't reliably train it there.
        ps = self.tasks_ps[self.base_task_index][torch.randint(0, self._n_samples, (self.batch_size,))]

        self.optim_encoder_sym.zero_grad()
        self.optim_decoder_sym.zero_grad()
        loss_symmetry = self.evaluate_symmetry(ps=ps, 
                                               generator=self.generator.param, 
                                               encoder_geo=self.encoder_geo, 
                                               decoder_geo=self.decoder_geo,
                                               encoder_sym=self.encoder_sym,
                                               decoder_sym=self.decoder_sym)
        loss_symmetry.backward()            
        self.optim_encoder_sym.step()
        self.optim_decoder_sym.step()


    def optimize(self, 
                 n_steps_lgs:int,
                 n_steps_gen:int,
                 n_steps_sym:int):
        
        """Main optimization loop."""
        self.progress_bar_lgs = tqdm(range(n_steps_lgs), desc="Learn log left actions...")
        self.progress_bar_gen = tqdm(range(n_steps_gen), desc="Learn generator...")
        self.progress_bar_sym = tqdm(range(n_steps_sym), desc="Learn symmetry...")

        # 1. Initialize left-actions, encoder and decoder.
        self._init_optimization()

        # 2. Learn left actions and their chart.
        logging.info("Learning log left actions and chart.")
        for p in self.log_lgs.parameters(): p.requires_grad = True
        for p in self.generator.parameters(): p.requires_grad = False
        for p in self.encoder_geo.parameters(): p.requires_grad = False
        for p in self.decoder_geo.parameters(): p.requires_grad = False
        for p in self.encoder_sym.parameters(): p.requires_grad = False
        for p in self.decoder_sym.parameters(): p.requires_grad = False

        if self.oracle_generator is None:
            for idx in self.progress_bar_lgs:

                if idx % self._update_chart_every_n_steps != 0:
                    self.take_step_left_actions()
                else:
                    self.take_step_chart_geo()

                if idx % self._log_wandb_every == 0:
                    self._log_to_wandb(step=self._global_step_wandb)
                    self._global_step_wandb+=self._log_wandb_every
                    time.sleep(0.05)

                if idx%self._save_every == 0 and self._save_dir is not None:
                    os.makedirs(f"{self._save_dir}/geo/step_{idx}") if not os.path.exists(f"{self._save_dir}/geo/step_{idx}") else None
                    self.save(f"{self._save_dir}/geo/step_{idx}/results.pt")

        logging.info("Finished learning log left actions and chart.")

        # 3. Learn generator.
        logging.info("Learning generator.")
        for p in self.log_lgs.parameters(): p.requires_grad = False
        for p in self.encoder_geo.parameters(): p.requires_grad = False
        for p in self.decoder_geo.parameters(): p.requires_grad = False
        for p in self.generator.parameters(): p.requires_grad = True
        if self.oracle_generator is None:
            for idx in self.progress_bar_lgs:

                self.take_step_generator()
                if idx % self._log_wandb_every == 0:
                    self._log_to_wandb(step=self._global_step_wandb)
                    self._global_step_wandb+=self._log_wandb_every
                    time.sleep(0.05)

                if idx%self._save_every == 0 and self._save_dir is not None:
                    os.makedirs(f"{self._save_dir}/gen/step_{idx}") if not os.path.exists(f"{self._save_dir}/gen/step_{idx}") else None
                    self.save(f"{self._save_dir}/gen/step_{idx}/results.pt")


        # 4. Symmetry discovery given the hereditary geometry.
        logging.info("Learning symmetry chart.")
        for p in self.generator.parameters(): p.requires_grad = False
        for p in self.encoder_sym.parameters(): p.requires_grad = True
        for p in self.decoder_sym.parameters(): p.requires_grad = True

        for idx in self.progress_bar_sym:
            
            self.take_step_chart_sym()
            
            if idx % self._log_wandb_every == 0:
                self._log_to_wandb(step=self._global_step_wandb)
                self._global_step_wandb+=self._log_wandb_every
                time.sleep(0.05)

            if idx%self._save_every == 0 and self._save_dir is not None:
                os.makedirs(f"{self._save_dir}/sym/step_{idx}") if not os.path.exists(f"{self._save_dir}/sym/step_{idx}") else None
                self.save(f"{self._save_dir}/sym/step_{idx}/results.pt")


    def save(self, path: str):
        """Saves the model to a file."""
        torch.save({
            'log_lgs': self.log_lgs.param,
            'generator': self.generator.param,
            'lgs': self.lgs,
            'encoder_geo_state_dict': self.encoder_geo.state_dict(),
            'decoder_geo_state_dict': self.decoder_geo.state_dict(),
            'encoder_sym_state_dict': self.encoder_sym.state_dict(),
            'decoder_sym_state_dict': self.decoder_sym.state_dict(),
            'losses': self._losses,
            'task_specifications': self.task_specifications,
            'seed': self.seed
        }, path)
        logging.info(f"Model saved to {path}")


    def rotation_vector_field(self, p_batch: torch.tensor, center)->torch.tensor:
        """Returns kernel samples at batched points p from a task."""

        _generator=torch.tensor([[0, -1], [1,0]], requires_grad=False, dtype=torch.float32).unsqueeze(0)
        projected_state = p_batch-center
        gradients = torch.einsum("dmn, bn->bdm", _generator, projected_state)
        norm_gradients = gradients.norm(dim=-1, keepdim=True)
        return gradients/norm_gradients


    def _init_log_lgs_linear_reg(self, epochs, log_wandb:bool=False):
        """Fits log-linear regressors to initialize left actions."""
        logging.info("Fitting log-linear regressors to initialize left actions.")
        self._log_lg_inits = [
            ExponentialLinearRegressor(input_dim=self.ambient_dim, seed=self.seed, log_wandb=log_wandb, task_idx=idx_task).fit(
                X=self.tasks_ps[0], Y=self.tasks_ps[idx_task], epochs=epochs
            )
            for idx_task in self.task_idxs]
        logging.info("Finished fitting log-linear regressors to initialize left actions.")   
        self._global_step_wandb+=len(self.task_idxs)*self._n_epochs_pretrain_log_lgs    
        return torch.stack(self._log_lg_inits, dim=0)


    @property
    def lgs(self):
        return torch.linalg.matrix_exp(self.log_lgs.param)
    

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
            "generator": np.array(self._losses["generator_span"][1:]),
            "symmetry": np.array(self._losses["symmetry"][1:]),
            "reconstruction_sym": np.array(self._losses["reconstruction_sym"][1:]),
            "symmetry_reg": np.array(self._losses["symmetry_reg"][1:]),
        }


    def _init_optimization(self):
        """Initializes the optimization: initializes the left-actions, encoder and decoder and defines the optimizers."""


        class TensorToModule(torch.nn.Module):
            def __init__(self, tensor):
                """Converts a tensor to a PyTorch module for easier gradient tracking. Used for the log-left actions and the generator."""
                super().__init__()
                self.param=torch.nn.Parameter(tensor)
                torch.nn.utils.parametrizations.weight_norm(self, name='param', dim=0)

        # 1. Log-left actions.
        if self._log_lg_inits_how == 'log_linreg':
            self._log_lg_inits=self._init_log_lgs_linear_reg(log_wandb=self._log_wandb,epochs=self._n_epochs_pretrain_log_lgs)

        elif self._log_lg_inits_how == 'random':
            self._log_lg_inits = torch.randn(size=(self._n_tasks-1, self.ambient_dim, self.ambient_dim))
        self.log_lgs=TensorToModule(self._log_lg_inits.clone())
        self.optimizer_lgs = torch.optim.Adam(self.log_lgs.parameters(),lr=self._lr_lgs)
        
        # 2. Generator.
        _generator=torch.stack([torch.eye(self.ambient_dim) for _ in range(self.kernel_dim)]) if self.oracle_generator is None else self.oracle_generator.clone()
        assert _generator.shape == (self.kernel_dim, self.ambient_dim, self.ambient_dim), "Generator must be of shape (d, n, n)." #TODO, this should rather be called Lie group dimension.
        self.generator= TensorToModule(_generator)
        self.optimizer_generator=torch.optim.Adam(self.generator.parameters(), lr=self._lr_gen)

        if self._eval_span_how=="weights":
            self.weights_lgs_to_gen = TensorToModule(torch.randn(size=(self._n_tasks-1, self.kernel_dim), requires_grad=True))
            self.optimizer_weights_lgs_to_gen= torch.optim.Adam(self.weights_lgs_to_gen.parameters(), lr=self._lr_gen)

        # 3. Charts.
        _identity_chart= identity_init_neural_net(self.encoder_geo, tasks_ps=self.tasks_ps, name="chart placeholder", log_wandb=self._log_wandb, 
                                               n_steps=self._n_epochs_init_neural_nets)
        self._global_step_wandb+=self._n_epochs_init_neural_nets

        self.encoder_geo = copy.deepcopy(_identity_chart)
        self.decoder_geo = copy.deepcopy(_identity_chart)
        self.encoder_sym = copy.deepcopy(_identity_chart)
        self.decoder_sym = copy.deepcopy(_identity_chart)
        del _identity_chart

        self.optim_encoder_geo = torch.optim.Adam(self.encoder_geo.parameters(), lr=self._lr_chart)
        self.optim_decoder_geo = torch.optim.Adam(self.decoder_geo.parameters(), lr=self._lr_chart)
        self.optim_encoder_sym = torch.optim.Adam(self.encoder_sym.parameters(), lr=self._lr_chart)
        self.optim_decoder_sym = torch.optim.Adam(self.decoder_sym.parameters(), lr=self._lr_chart)


    def _stack_samples(self):
        """Stacks samples from all tasks into a single tensor."""
        _n_samples_per_task, ambient_dim = self.tasks_ps[0].shape
        ps = torch.empty([self._n_tasks, _n_samples_per_task, ambient_dim], dtype=torch.float32)
        for i, task_ps in enumerate(self._n_tasks):
            ps[i] = task_ps
        self.all_ps=ps.reshape([-1, ambient_dim])


    def _validate_inputs(self):
        """Validates user inputs."""
        assert self._log_lg_inits_how in ['random', 'log_linreg'], "_log_lg_inits_how must be one of ['random', 'log_linreg']."
        assert len(self.tasks_ps) == len(self.tasks_frameestimators), "Number of tasks and frame estimators must match."
        logging.info("Using oracle generator") if self.oracle_generator is not None else None

        
    def _log_to_wandb(self, step:int):
        """Logs losses to weights and biases."""
        if not self._log_wandb:
            return
        
        def _log_grad_norms(module: torch.nn.Module, prefix: str):
            """Logs L2 norms of gradients of a PyTorch module to wandb."""
            for name, param in module.named_parameters():
                if param.grad is not None:
                    metrics[f"grad_norms/{prefix}/{name}"] = param.grad.norm().item()

        metrics= {
            "train/left_actions/mean": float(self._losses['left_actions'][-1]),
            "train/regularizers/left_actions/lasso": float(self._losses['left_actions_tasks_reg'][-1]),

            "train/geometry/generator_span": float(self._losses['generator_span'][-1]),
            "train/geometry/generator_weights": float(self._losses['generator_weights'][-1]),
            "train/regularizers/generator/lasso": float(self._losses['generator_reg'][-1]),
            "train/geometry/reconstruction": float(self._losses['reconstruction_geo'][-1]),

            "train/symmetry/span": float(self._losses['symmetry'][-1]),
            "train/symmetry/reconstruction": float(self._losses['reconstruction_sym'][-1]),
            "train/regularizers/symmetry": float(self._losses['symmetry_reg'][-1]),

            "diagnostics/cond_num_generator": float(self._diagnostics['cond_num_generator'][-1]),
            "diagnostics/frob_norm_generator": float(self._diagnostics['frob_norm_generator'][-1]),
        }

        if self._diagnostics["encoder_loss_oracle_sym"] is not None and self._diagnostics["decoder_loss_oracle_sym"] is not None:
            metrics["diagnostics/encoder_loss_oracle_sym"] = float(self._diagnostics['encoder_loss_oracle_sym'][-1])
            metrics["diagnostics/decoder_loss_oracle_sym"] = float(self._diagnostics['decoder_loss_oracle_sym'][-1])

        if self._diagnostics["encoder_loss_oracle_geo"] is not None and self._diagnostics["decoder_loss_oracle_geo"] is not None:
            metrics["diagnostics/encoder_loss_oracle_geo"] = float(self._diagnostics['encoder_loss_oracle_geo'][-1])
            metrics["diagnostics/decoder_loss_oracle_geo"] = float(self._diagnostics['decoder_loss_oracle_geo'][-1])

        task_losses= self._losses['left_actions_tasks'][-1]
        for idx_task in range(self._n_tasks-1):
            metrics[f"train/left_actions/tasks/task_idx={idx_task}"] = float(task_losses[idx_task])

        if self._log_wandb_gradients:
            _log_grad_norms(self.encoder_geo, "encoder_geo")
            _log_grad_norms(self.decoder_geo, "decoder_geo")
            _log_grad_norms(self.encoder_sym, "encoder_sym")
            _log_grad_norms(self.decoder_sym, "decoder_sym")
            _log_grad_norms(self.log_lgs, "log_lgs")
            _log_grad_norms(self.generator, "generator")
            _log_grad_norms(self.weights_lgs_to_gen, "weights_lgs_to_gen") if self._eval_span_how == "weights" else None

        wandb.log(metrics, step=step)