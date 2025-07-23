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
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 update_chart_every_n_steps:int,
                 lr_left_actions:float,
                 lr_generator:float,
                 lr_encoder:float,
                 lr_decoder:float,
                 n_steps_pretrain_geometry:int,
                 hyper_grad_leader_how: Literal['unrolled', 'implicit', 'blackbox'],
                 seed:int=42,
                 log_lg_inits_how:Literal['log_linreg', 'random']= 'log_linreg',
                 batch_size:int=64,
                 bandwidth:float=0.5,

                 lasso_coef_lgs: Optional[float] = 0.5,
                 lasso_coef_encoder_decoder: Optional[float] = 0.005,
                 lasso_coef_generator: Optional[float] = 0.005,
                 log_wandb:bool=False,
                 task_specifications:list=None,
                 use_oracle_rotation_kernel:bool=False,
                 oracle_generator: torch.tensor=None,
                 learn_encoder_decoder:bool=False,
                 verbose:bool=False
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
        self._lasso_coef_generator = lasso_coef_generator
        self._lr_left_actions=lr_left_actions
        self._lr_generator=lr_generator
        self._lr_encoder=lr_encoder
        self._lr_decoder=lr_decoder
        self._n_steps_pretrain_geometry=n_steps_pretrain_geometry
        self.hyper_grad_leader_how=hyper_grad_leader_how

        self._log_wandb=log_wandb
        self._use_oracle_rotation_kernel=use_oracle_rotation_kernel
        self.task_specifications=task_specifications
        self.oracle_generator= oracle_generator
        self._learn_encoder_decoder=learn_encoder_decoder
        self._verbose=verbose

        self._validate_inputs()

        logging.info(f"Fitting encoder and decoder: {self._learn_encoder_decoder}")


        self.ambient_dim=tasks_ps[0][0,:].shape[0]
        self._n_tasks=len(tasks_ps)
        self._n_samples=tasks_ps[0][:,0].shape[0]
        self.task_idxs = list(range(self._n_tasks))
        self.task_idxs.remove(self.base_task_index)
        self.frame_base_task= self.tasks_frameestimators[self.base_task_index]
        self.frame_i_tasks= [self.tasks_frameestimators[i] for i in self.task_idxs]

        self._losses, self._diagnostics={}, {}
        self._losses["left_actions"]= [torch.tensor([0])]
        self._losses["left_actions_tasks"]= [torch.tensor([0])]
        self._losses["left_actions_tasks_reg"]= [torch.tensor([0])]

        self._losses["generator"]= [torch.tensor([0])]
        self._losses["generator_reg"] = [torch.tensor([0])]
        
        self._losses["symmetry"]= [torch.tensor([0])]
        self._losses["reconstruction"]= [torch.tensor([0])]
        self._losses["symmetry_reg"] = [torch.tensor([0])]

        self._diagnostics["cond_num_generator"] = [torch.tensor([0])]
        self._diagnostics["frob_norm_generator"] = [torch.tensor([0])]
        

        # Optimization variables
        torch.manual_seed(seed)
        self._init_optimization()
    
    def evalute_left_actions(self, 
                             ps: torch.Tensor, 
                             log_lgs: torch.Tensor, 
                             encoder: torch.nn.Module,
                             decoder: torch.nn.Module,
                             track_loss:bool=True) -> float:
        """Computes kernel alignment loss of all left-actions."""
        # 1. Push-forward
        tilde_ps=encoder(ps)
        lgs=torch.linalg.matrix_exp(log_lgs.param)
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
        self.task_losses_reg = self._lasso_coef_lgs*torch.norm(log_lgs.param, p=1, dim=(-1)).mean(-1).mean(-1)

        if track_loss:
            self._losses["left_actions_tasks"].append(self.task_losses.detach().cpu().numpy())
            self._losses["left_actions_tasks_reg"].append(self.task_losses_reg.detach().cpu().numpy())
            self._losses["left_actions"].append(self.task_losses.mean().detach().cpu().numpy()) #exclude regularization term from this loss.

        return self.task_losses.mean() + self.task_losses_reg
    

    def evaluate_generator_span(self, 
                                generator: torch.nn.Module, 
                                log_lgs: torch.Tensor, 
                                track_loss:bool=True)->float:
        """
        Evalutes whether all left-actions are inside the span of the generator.
        log_lgs are frozen in this loss function (and hence a detached tensor).
        """
        _, ortho_log_lgs_generator=self._project_onto_tensor_subspace(log_lgs, generator.param)
        loss_span=torch.mean(torch.norm(ortho_log_lgs_generator, p="fro",dim=(1,2)),dim=0)
        loss_reg=self._lasso_coef_generator*self._l1_penalty(generator)

        if track_loss:
            self._losses["generator"].append(loss_span.detach().cpu().numpy())
            self._losses["generator_reg"].append(loss_reg.detach().cpu().numpy())
        return loss_span+loss_reg

    
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
        gen_tilde_ps = torch.einsum("dnm,bm->bdn", generator.param, tilde_ps)
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
        loss_reg = self._lasso_coef_encoder_decoder * (self._l1_penalty(encoder) + self._l1_penalty(decoder))

        if track_loss:
            self._losses["symmetry"].append(loss_symmetry.detach().cpu().numpy())
            self._losses["reconstruction"].append(loss_reconstruction.detach().cpu().numpy())
            self._losses["symmetry_reg"].append(loss_reg.detach().cpu().numpy())

        return loss_symmetry+loss_reconstruction+loss_reg
    

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
            self._diagnostics["frob_norm_generator"].append(torch.mean(torch.norm(basis_flat, p='fro', dim=(-1, -2))).item())

        return proj, ortho_comp
    
        
    def take_step_geometry(self, step_counter:Optional[int]=None):
        """Update the geometry variables under a frozen chart."""
        ps = self.tasks_ps[self.base_task_index][torch.randint(0, self._n_samples, (self.batch_size,))]

        for p in self.log_lgs.parameters(): p.requires_grad = True
        for p in self.generator.parameters(): p.requires_grad = True
        for p in self.encoder.parameters(): p.requires_grad = False
        for p in self.decoder.parameters(): p.requires_grad = False

        self.optimizer_lgs.zero_grad()
        self.optimizer_generator.zero_grad()
        
        loss_left_action = self.evalute_left_actions(ps=ps, log_lgs=self.log_lgs, encoder=self.encoder, decoder=self.decoder)
        loss_symmetry = self.evalute_symmetry(ps=ps, generator=self.generator, encoder=self.encoder, decoder=self.decoder)
        log_lgs_detach_tensor = self.log_lgs.param.detach()
        loss_span = self.evaluate_generator_span(generator=self.generator, log_lgs=log_lgs_detach_tensor)
        
        (loss_left_action + loss_span + loss_symmetry).backward()

        self.optimizer_lgs.step()
        self.optimizer_generator.step()

        if step_counter is not None and step_counter % 5 == 0:
            self._log_to_wandb(step_counter)
            time.sleep(0.05)


    def take_step_chart(self, step_counter:Optional[int]=None):
        """Update the chart under frozen geometry variables."""
        ps = self.tasks_ps[self.base_task_index][torch.randint(0, self._n_samples, (self.batch_size,))]

        for p in self.log_lgs.parameters(): p.requires_grad = False
        for p in self.generator.parameters(): p.requires_grad = False
        for p in self.encoder.parameters(): p.requires_grad = True
        for p in self.decoder.parameters(): p.requires_grad = True

        self.optimizer_encoder.zero_grad()
        self.optimizer_decoder.zero_grad()

        loss_left_action = self.evalute_left_actions(ps=ps, log_lgs=self.log_lgs, encoder=self.encoder, decoder=self.decoder)
        loss_symmetry = self.evalute_symmetry(ps=ps, generator=self.generator, encoder=self.encoder, decoder=self.decoder)
        (loss_symmetry + loss_left_action).backward()            
        self.optimizer_encoder.step()
        self.optimizer_decoder.step()

        if step_counter is not None and step_counter % 5 == 0:
            self._log_to_wandb(step_counter)
            time.sleep(0.05)


    def optimize(self, n_steps:int=1000):
        """Main optimization loop."""
        self.progress_bar = tqdm(range(n_steps), desc="Hereditary Symmetry Discovery")

        step_counter = 0
        for _ in range(self._n_steps_pretrain_geometry):
            self.take_step_geometry(step_counter=step_counter)
            step_counter += 1


        for _ in self.progress_bar:
            
            for _ in range(self._update_chart_every_n_steps):
                self.take_step_geometry(step_counter=step_counter)
                step_counter += 1

            if self.hyper_grad_leader_how=="unrolled":
                self.take_step_chart_unrolled(step_counter=step_counter)
            elif self.hyper_grad_leader_how=="implicit":
                self.take_step_chart_implicit(step_counter=step_counter)
            else:
                self.take_step_chart(step_counter=step_counter)
            step_counter += 1

            if step_counter>=n_steps:
                logging.info("Reached maximum number of steps, stopping optimization.")
                break


    def save(self, path: str):
        """Saves the model to a file."""
        torch.save({
            'log_lgs': self.log_lgs.param,
            'generator': self.generator.param,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'losses': self._losses,
            'task_specifications': self.task_specifications,
            'seed': self.seed
        }, path)
        logging.info(f"Model saved to {path}")


    def rotation_vector_field(self, p_batch: torch.tensor, center)->torch.tensor:
        """Returns kernel samples at batched points p from a task."""

        _generator=torch.tensor([[0, -1], [1,0]], requires_grad=False, dtype=torch.float32).unsqueeze(0)
        projected_state = p_batch-center
        return torch.einsum("dmn, bn->bdm", _generator, projected_state)
    

    def _validate_inputs(self):
        """Validates user inputs."""
        assert self._log_lg_inits_how in ['random', 'log_linreg'], "_log_lg_inits_how must be one of ['random', 'log_linreg']."
        assert len(self.tasks_ps) == len(self.tasks_frameestimators), "Number of tasks and frame estimators must match."
        logging.info("Using oracle generator") if self.oracle_generator is not None else None

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
        
        def _log_grad_norms(module: torch.nn.Module, prefix: str):
            """Logs L2 norms of gradients of a PyTorch module to wandb."""
            for name, param in module.named_parameters():
                if param.grad is not None:
                    metrics[f"grad_norms/{prefix}/{name}"] = param.grad.norm().item()

        metrics= {
            "train/left_actions/mean": float(self._losses['left_actions'][-1]),
            # "train/left_actions/tasks": float(self._losses['left_actions_tasks'][-1]), #TODO, log task level losses.
            "train/generator": float(self._losses['generator'][-1]),
            "train/symmetry/span": float(self._losses['symmetry'][-1]),
            "train/symmetry/reconstruction": float(self._losses['reconstruction'][-1]),
            "train/regularizers/symmetry": float(self._losses['symmetry_reg'][-1]),
            "train/regularizers/left_actions/lasso": float(self._losses['left_actions_tasks_reg'][-1]),
            "train/regularizers/generator/lasso": float(self._losses['generator_reg'][-1]),

            "diagnostics/cond_num_generator": float(self._diagnostics['cond_num_generator'][-1]),
            "diagnostics/frob_norm_generator": float(self._diagnostics['frob_norm_generator'][-1]),
        }

        _log_grad_norms(self.encoder, "encoder")
        _log_grad_norms(self.decoder, "decoder")
        _log_grad_norms(self.log_lgs, "log_lgs")
        _log_grad_norms(self.generator, "generator")

        wandb.log(metrics, step=step)


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
            "generator": np.array(self._losses["generator"][1:]),
            "symmetry": np.array(self._losses["symmetry"][1:]),
            "reconstruction": np.array(self._losses["reconstruction"][1:]),
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


        if self._log_lg_inits_how == 'log_linreg':
            self._log_lg_inits=self._init_log_lgs_linear_reg(verbose=False, log_wandb=self._log_wandb, 
                                                             epochs=2_500,
                                                            #  epochs=1
                                                             )

        elif self._log_lg_inits_how == 'random':
            self._log_lg_inits = torch.randn(size=(self._n_tasks-1, self.ambient_dim, self.ambient_dim))
        self.log_lgs=TensorToModule(self._log_lg_inits.clone())
        self.optimizer_lgs = torch.optim.Adam(self.log_lgs.parameters(),lr=self._lr_left_actions)
        
        _generator=torch.stack([torch.eye(self.ambient_dim) for _ in range(self.kernel_dim)]) if self.oracle_generator is None else self.oracle_generator.clone()
        # _generator=torch.randn(size=(self.kernel_dim, self.ambient_dim, self.ambient_dim)) if self.oracle_generator is None else self.oracle_generator.clone()
        assert _generator.shape == (self.kernel_dim, self.ambient_dim, self.ambient_dim), "Generator must be of shape (d, n, n)."
        self.generator= TensorToModule(_generator)
        self.optimizer_generator=torch.optim.Adam(self.generator.parameters(), lr=self._lr_generator)


        self.encoder= identity_init_neural_net(self.encoder, tasks_ps=self.tasks_ps, name="encoder", log_wandb=self._log_wandb, 
                                            #    n_steps=1
                                               )
        self.decoder= identity_init_neural_net(self.decoder, tasks_ps=self.tasks_ps, name="decoder", log_wandb=self._log_wandb,
                                            #    n_steps=1
                                               )
        self.optimizer_encoder=torch.optim.Adam(self.encoder.parameters(), lr=self._lr_encoder)
        self.optimizer_decoder=torch.optim.Adam(self.decoder.parameters(), lr=self._lr_decoder)