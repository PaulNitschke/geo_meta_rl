import torch
import warnings
from tqdm import tqdm
from typing import Literal, List

from ....utils import approx_mode, matrixLogarithm


class HereditaryGeometryDiscovery():

    def __init__(self,
                 tasks_ps: List[torch.tensor],
                 tasks_frameestimators: List[callable],
                 kernel_dim: int,
                 encoder: callable=None,
                 decoder: callable=None,
                 seed:int=42,
                 lg_inits_how:Literal['random', 'mode', 'zeros']= 'mode',
                 batch_size:int=64,
                 bandwidth:float=0.5,

                 use_oracle_rotation_kernel:bool=False,
                 learn_left_actions:bool=True,
                 learn_encoder_decoder:bool=False,
                 oracle_generator: torch.tensor=None,
                 task_specifications:list=None,
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
        self._lg_inits_how=lg_inits_how
        self.encoder=encoder
        self.decoder=decoder
        self.batch_size=batch_size
        self.bandwidth=bandwidth


        self._use_oracle_rotation_kernel=use_oracle_rotation_kernel
        self.task_specifications=task_specifications
        self.oracle_generator= oracle_generator
        self._learn_left_actions=learn_left_actions
        self._learn_encoder_decoder=learn_encoder_decoder

        self._validate_inputs()


        self.ambient_dim=tasks_ps[0][0,:].shape[0]
        self._n_tasks=len(tasks_ps)
        self._n_samples=tasks_ps[0][:,0].shape[0]
        self.task_idxs = list(range(self._n_tasks))
        self.task_idxs.remove(self.base_task_index)
        self.frame_base_task= self.tasks_frameestimators[self.base_task_index]
        self.frame_i_tasks= [self.tasks_frameestimators[i] for i in self.task_idxs]

        self._matrix_log=matrixLogarithm()
        self._losses=[]
        self._task_losses=[]
        

        # Optimization variables
        torch.manual_seed(seed)
        # self.lgs=[torch.nn.Parameter(torch.randn(self.ambient_dim, self.ambient_dim)) for _ in range(self._n_tasks-1)]  # one left-action per task minus the base task.
        # self.optimizer_lgs=[torch.optim.Adam([lg], lr=0.00035) for lg in self.lgs]  # one optimizer per left-action.
        if self._lg_inits_how == 'mode':
            approx_modes=[approx_mode(ps) for ps in tasks_ps]
            lg_inits=[torch.linalg.lstsq(approx_modes[0].unsqueeze(0), approx_modes[i].unsqueeze(0)).solution.T for i in self.task_idxs]
            lg_inits=torch.stack(lg_inits)
            self.lgs = torch.nn.Parameter(lg_inits)
            assert self.lgs.shape == (self._n_tasks-1, self.ambient_dim, self.ambient_dim), "Left actions must be of shape (N_tasks-1, n, n)."
        elif self._lg_inits_how == 'random':
            self.lgs = torch.nn.Parameter(torch.randn(size=(self._n_tasks-1, self.ambient_dim, self.ambient_dim)))
        self._lg_inits=self.lgs.clone()

        self.generator=torch.nn.Parameter(torch.randn(size=(self.kernel_dim, self.ambient_dim, self.ambient_dim))) if self.oracle_generator is None else torch.nn.Parameter(self.oracle_generator.clone())


        self.optimizer_lgs=torch.optim.Adam([self.lgs], lr=0.00035)
        self.optimizer_generator=torch.optim.Adam([self.generator], lr=0.00035)
        self.optimizer_encoder=torch.optim.Adam(self.encoder.parameters(), lr=0.00035) if self.encoder is not None else None
        self.optimizer_decoder=torch.optim.Adam(self.decoder.parameters(), lr=0.00035) if self.decoder is not None else None

    
    def evalute_left_actions(self, lgs: torch.Tensor, track_loss:bool=True) -> float:
        """Computes kernel alignment loss of all left-actions."""

        # 1. Sample points and push-forward
        _b_idxs= torch.randint(0, self._n_samples, (self.batch_size,))
        ps = self.tasks_ps[self.base_task_index][_b_idxs]
        lg_ps = torch.einsum("Nmn,bn->Nbm", lgs, ps)


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


        # 3. Compute projection loss.
        def compute_ortho_loss(vec):
            """Computes orthogonal complement loss by averaging norm of orthogonal complement across kernel dimensions and batch size."""
            assert vec.dim()==4, "Vector field must be of shape (N, b, d, n)."
            vec=torch.norm(vec, dim=(-1))
            vec=vec.sum(-1)
            return vec.mean(-1)

        ortho_frame_i_lg_ps = self._ortho_comp(frames_i_lg_ps, lgs_frame_ps)
        ortho_lgs_frame_ps = self._ortho_comp(lgs_frame_ps, frames_i_lg_ps)
        self.task_losses = compute_ortho_loss(ortho_frame_i_lg_ps) + compute_ortho_loss(ortho_lgs_frame_ps) 

        if track_loss:
            self._losses.append(self.task_losses.detach().cpu().numpy())
        return self.task_losses.mean()
    

    def evaluate_generator_span(self, track_loss:bool=True)->float:
        """Evalutes whether all left-actions are inside the span of the generator."""
        log_lgs=torch.stack([self._matrix_log.apply(self.lgs[idx_task]) for idx_task in range(self._n_tasks-1)], dim=0)

        ortho_log_lgs_generator=self._ortho_comp(log_lgs, self.generator)
    

    def evalute_symmetry(self)->float:
        """Evaluates whether the generator is contained within the kernel distribution of the base task (expressed in the encoder and decoder)."""
        assert self.oracle_generator is not None, "Oracle generator must be provided to evaluate symmetry."
        assert self.encoder is not None, "Provide an encoder to evaluate symmetry."
        assert self.decoder is not None, "Provide a decoder to evaluate symmetry."
        warnings.warn("Using oracle generator. Only use for debugging.")
        warnings.warn("Implementation not correct yet.")

        # 1. Sample points from the base task.
        ps = self.tasks_ps[self.base_task_index][torch.randint(0, self._n_samples, (self.batch_size,))]
        
        # 2. Encode points into latent space.
        tilde_ps=self.encoder(ps)

        # 3. Let generator act on points.
        gen_tilde_ps = torch.einsum("dnm,bm->dbn", self.oracle_generator, tilde_ps)  # shape (d, b, n)

        # 4. Decode points back to ambient space.
        gen_ps = self.decoder(gen_tilde_ps).unsqueeze(0) #shape(1, b, d, n) for projection function.

        # 5. Check symmetry
        goal_base = self.task_specifications[self.base_task_index]['goal']
        frame_ps= self.rotation_vector_field(ps, center=goal_base)
        frame_ps=frame_ps.unsqueeze(0) #shape(1, b, d, n) for projection function.

        gen_into_frame = self._ortho_comp(gen_ps, frame_ps)
        self.loss_symmetry= torch.norm(gen_into_frame, dim=(-1)).sum(-1).mean()

        self.loss_reconstruction= torch.norm(ps - self.decoder(self.encoder(gen_ps)), dim=(-1)).mean()

        warnings.warn("Need to enfore that encoder and decoder are identity in the beginning. Also need to use derivative of decoder Jacobian.")

        return self.loss_symmetry+self.loss_reconstruction
        

    def _project_onto_subspace(self, vecs, basis):
        """
        Projects a set of vectors onto a subspace spanned by the basis.
        vecs: tensor of shape (N, b, d, n)
        basis: tensor of shape (N, b, d, n)
        Returns:
        - proj_vecs: tensor of shape (N, b, d, n)
        """
        basis_t = basis.transpose(-2, -1)
        G = torch.matmul(basis, basis_t)
        G_inv = torch.linalg.pinv(G)
        P = torch.matmul(basis_t, torch.matmul(G_inv, basis))
        return torch.matmul(vecs, P)


    def _ortho_comp(self, vecs, basis):
        """
        Computes the orthogonal complement of vecs with respect to basis.
        vecs: tensor of shape (N, b, d, n)
        basis: tensor of shape (N, b, d, n)
        Returns:
        - ortho_vecs: tensor of shape (N, b, d, n)
        """
        proj_vecs = self._project_onto_subspace(vecs, basis)
        return vecs - proj_vecs

        
    def take_grad_step(self):
        """Takes one gradient step on the left action."""

        if self._learn_left_actions:
            self.optimizer_lgs.zero_grad()
            loss = self.evalute_left_actions(lgs=self.lgs)
            loss.backward()
            self.optimizer_lgs.step()

        if self._learn_encoder_decoder:
            self.optimizer_encoder.zero_grad()
            self.optimizer_decoder.zero_grad()

            loss = self.evalute_symmetry()
            loss.backward()
            self.optimizer_encoder.step()
            self.optimizer_decoder.step()


        # for optimizer in self.optimizer_lgs:
        #     optimizer.zero_grad()

        # loss = compute_joint_loss(self.lgs)
        # loss.backward()

        # for optimizer in self.optimizer_lgs:
        #     optimizer.step()
        return loss
    

    def optimize(self, n_steps:int=1000):
        """Optimizes the left action."""
        progress_bar = tqdm(range(n_steps), desc="Inferring left-action")

        for n_steps in progress_bar:
            loss = self.take_grad_step()
            
            if n_steps% 50 == 0 and n_steps>0:
                if self._learn_left_actions:
                    progress_bar.set_description(
                        f"Loss: {loss.item():.2e}, Task Losses: {self.task_losses.detach().cpu().numpy().round(3)}"
                    )
                if self._learn_encoder_decoder:
                    progress_bar.set_description(
                        f"Symmetry Loss: {self.loss_symmetry.item():.2e}, Reconstruction Loss: {self.loss_reconstruction.item():.2e}"
                    )


    def rotation_vector_field(self, p_batch: torch.tensor, center)->torch.tensor:
        """Returns kernel samples at batched points p from a task."""

        _generator=torch.tensor([[0, -1], [1,0]], requires_grad=False, dtype=torch.float32).unsqueeze(0)
        projected_state = p_batch-center
        return torch.einsum("dmn, bn->bdm", _generator, projected_state)
    

    def _validate_inputs(self):
        """Validates user inputs."""
        assert self._lg_inits_how in ['random', 'mode', 'zeros'], "lg_inits_how must be one of ['random', 'mode', 'zeros']."
        assert len(self.tasks_ps) == len(self.tasks_frameestimators), "Number of tasks and frame estimators must match."
        # TODO: assert that encoder and decoder are identity map in the beginning