import torch
import warnings
from tqdm import tqdm

class HereditaryGeometryDiscovery():

    def __init__(self,
                 tasks_ps: list,
                 tasks_frameestimators: list,
                 kernel_dim: int,
                 seed:int=42,
                 lg_inits:torch.tensor=None,
                 learn_left_actions:bool=True,
                 learn_encoder_decoder:bool=False,
                 oracle_generator: torch.tensor=None,
                 encoder: callable=None,
                 decoder: callable=None,
                 base_task_idx: int=0,
                 task_specifications:list=None,
                 batch_size:int=64,
                 bandwidth:float=0.5):
        """Hereditary Geometry Discovery.
        TODO: signature.
        
        Notation:
        - d: kernel dimension.
        - n: ambient dimension.
        - b: batch size.
        - N: number of tasks.
        """
        
        self.tasks_ps= tasks_ps
        self.base_task_index=base_task_idx
        self.tasks_frameestimators=tasks_frameestimators
        self.task_specifications=task_specifications
        assert len(tasks_ps) == len(tasks_frameestimators), "Number of tasks and frame estimators must match."
        self.encoder=encoder
        self.decoder=decoder
        self.oracle_generator= oracle_generator
        self._learn_left_actions=learn_left_actions
        self._learn_encoder_decoder=learn_encoder_decoder

        self.bandwidth=bandwidth

        self.kernel_dim=kernel_dim
        self.ambient_dim=tasks_ps[0][0,:].shape[0]
        self.batch_size=batch_size
        self._n_tasks=len(tasks_ps)
        self._n_samples=tasks_ps[0][:,0].shape[0]
        self.task_idxs = list(range(self._n_tasks))
        self.task_idxs.remove(base_task_idx)

        self._losses=[]
        self._task_losses=[]
        

        # Optimization variables
        warnings.warn("Only one left base action implemented.")
        torch.manual_seed(seed)
        # self.lgs=[torch.nn.Parameter(torch.randn(self.ambient_dim, self.ambient_dim)) for _ in range(self._n_tasks-1)]  # one left-action per task minus the base task.
        # self.optimizer_lgs=[torch.optim.Adam([lg], lr=0.00035) for lg in self.lgs]  # one optimizer per left-action.
        if lg_inits is not None:
            assert lg_inits.shape == (self._n_tasks-1, self.ambient_dim, self.ambient_dim), "Left-action initializations must have shape (N-1, d, d)."
            self.lgs = torch.nn.Parameter(lg_inits)
        else:
            self.lgs = torch.nn.Parameter(torch.randn(size=(self._n_tasks-1, self.ambient_dim, self.ambient_dim)))
        self.optimizer_lgs=torch.optim.Adam([self.lgs], lr=0.00035)

        self.optimizer_encoder=torch.optim.Adam(self.encoder.parameters(), lr=0.00035) if self.encoder is not None else None
        self.optimizer_decoder=torch.optim.Adam(self.decoder.parameters(), lr=0.00035) if self.decoder is not None else None

    
    def evalute_left_actions(self, lgs: torch.Tensor, track_loss:bool=True):
        """Computes kernel alignment loss of all left-actions."""

        # 1. Sample points and push-forward
        ps = self.tasks_ps[self.base_task_index][torch.randint(0, self._n_samples, (self.batch_size,))]
        lg_ps = torch.einsum("Nmn,bn->Nbm", lgs, ps)  # shape (n_tasks-1, b, n)


        # 2. Sample tangent vectors and push-forward tangent vectors
        # TODO: this uses ground-truth frame for debugging.
        goal_base = self.task_specifications[self.base_task_index]['goal']
        frame_ps= self.rotation_vector_field(ps, center=goal_base)
        lgs_frame_ps=torch.einsum("Nmn,bdn->Nbdm", lgs, frame_ps)  #(n_tasks, kernel_dim, ambient_dim)      
        # lgs_frame_ps= torch.stack([torch.einsum("mn, bn->bm", lgs[i], frame_ps) for i in self.task_idxs], dim=0)  # shape (n_tasks-1, b, d, n)
        goals = torch.stack([torch.tensor(self.task_specifications[i]['goal']) for i in self.task_idxs])
        # frames_i_lg_ps = torch.vmap(self.rotation_vector_field,in_dims=(0, 0))(lg_ps, goals)
        frames_i_lg_ps = torch.stack([
            self.rotation_vector_field(lg_ps[i], goals[i])
            for i in range(lg_ps.shape[0])
            # for i in len(lg_ps)
        ], dim=0)
    #     # frame_base_task= self.tasks_frameestimators[idx_task_base]
    #     # frame_i_task= self.tasks_frameestimators[idx_task_i]
    #     # frame_base=frame_base_task.evaluate(ps_base_batch, bandwidth=self.bandwidth)
    #     # frame_i_push= frame_i_task.evaluate(ps_base_push, bandwidth=self.bandwidth)


        # 3. Compute projection loss.
        def compute_ortho_loss(vec):
            """Computes orthogonal complement loss by averaging norm of orthogonal complement across kernel dimensions and batch size."""
            assert vec.dim()==4, "Vector field must be of shape (N, b, d, n)."
            vec=torch.norm(vec, dim=(-1)) #norm of orthogonal complement along each vector field.
            vec=vec.sum(-1) #Sum across kernel dimensions.
            return vec.mean(-1) #mean across batch

        ortho_frame_i_lg_ps = self._ortho_comp(frames_i_lg_ps, lgs_frame_ps)
        ortho_lgs_frame_ps = self._ortho_comp(lgs_frame_ps, frames_i_lg_ps)
        self.task_losses = compute_ortho_loss(ortho_frame_i_lg_ps) + compute_ortho_loss(ortho_lgs_frame_ps) 

        if track_loss:
            self._losses.append(self.task_losses.detach().cpu().numpy())
        return self.task_losses.mean()
    

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

        return self.loss_symmetry+self.loss_reconstruction
        


    def _project_onto_subspace(self, vecs, basis):
        """
        Projects a set of vectors onto a subspace spanned by the basis.
        vecs: tensor of shape (N, b, d, n)
        basis: tensor of shape (N, b, d, n)
        Returns:
        - proj_vecs: tensor of shape (N, b, d, n)
        """
        basis_t = basis.transpose(-2, -1)  # (N, b, n, d)

        # Gram matrix: (N, b, d, d)
        G = torch.matmul(basis, basis_t)
        G_inv = torch.linalg.pinv(G)  # (N, b, d, d)

        # Projection matrix: (N, b, n, n)
        P = torch.matmul(basis_t, torch.matmul(G_inv, basis))  # (N, b, n, n)

        # Project vecs: (N, b, d, n)
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
            
            if n_steps% 1000 == 0 and n_steps>0:
                if self._learn_left_actions:
                    progress_bar.set_description(
                        f"Loss: {loss.item():.2e}, Task Losses: {self.task_losses.detach().cpu().numpy().round(3)}"
                    )
                if self._learn_encoder_decoder:
                    progress_bar.set_description(
                        f"Symmetry Loss: {self.loss_symmetry.item():.2e}, Reconstruction Loss: {self.loss_reconstruction.item():.2e}"
                    )


    # def rotation_vector_field(self, x, center):
    #     """
    #     Compute the rotational vector field directions at a batch of 2D points around a given center.

    #     Args:
    #         x (Tensor): Tensor of shape (N, 2) representing N 2D points.
    #         center (Tensor): Tensor of shape (2,) representing the center of rotation (a, b).

    #     Returns:
    #         Tensor: Tensor of shape (N, 2), each row is the direction of the vector field at the corresponding point in `x`.
    #     """
    #     warnings.warn("Using hard-coded kernel. Only use for debugging.")
    #     v = x - center  # shape (N, 2)
    #     rotated = torch.stack([-v[:, 1], v[:, 0]], dim=1)  # rotate 90° CCW
    #     return rotated.unsqueeze(1)

    def rotation_vector_field(self, p_batch: torch.tensor, center)->torch.tensor:
        """Returns kernel samples at batched points p from a task."""

        _generator=torch.tensor([[0, -1], [1,0]], requires_grad=False, dtype=torch.float32).unsqueeze(0)
        projected_state = p_batch-center
        return torch.einsum("dmn, bn->bdm", _generator, projected_state)