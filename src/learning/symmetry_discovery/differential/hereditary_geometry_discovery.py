import torch
import warnings
from tqdm import tqdm
from functools import partial

class HereditaryGeometryDiscovery():

    def __init__(self,
                 tasks_ps: list,
                 tasks_frameestimators: list,
                 kernel_dim: int,
                 seed:int,
                 task_specifications:list=None,
                 batch_size:int=64,
                 bandwidth:float=0.5):
        """Hereditary Geometry Discovery.
        TODO: signature.
        
        Notation:
        - d: kernel dimension.
        - n: ambient dimension.
        - b: batch size.
        """
        
        self.tasks_ps= tasks_ps
        self.tasks_frameestimators=tasks_frameestimators
        self.task_specifications=task_specifications
        assert len(tasks_ps) == len(tasks_frameestimators), "Number of tasks and frame estimators must match."

        self.bandwidth=bandwidth

        self.kernel_dim=kernel_dim
        self.ambient_dim=tasks_ps[0][0,:].shape[0] #ambient dimension, uses small n in einsums.
        self.batch_size=batch_size
        self._n_tasks=len(tasks_ps) # uses capital N in einsums.
        self._losses=[]
        self._n_samples=tasks_ps[0][:,0].shape[0]

        # Optimization variables
        warnings.warn("Only one left base action implemented.")
        torch.manual_seed(seed)
        self.lgs = torch.nn.Parameter(torch.randn(size=(self._n_tasks-1, self.ambient_dim, self.ambient_dim))) #one left-action per task minus the base task.
        self.optimizer_lgs=torch.optim.Adam([self.lgs], lr=1e-4)


    # def evaluate_one_left_action(self, lg_i:torch.tensor, idx_task_base:int, idx_task_i:int):
    #     """Evaluates the left action of a group element lg_i on a base task ps_base_task and a target task ps_i_task."""

    #     ps_base_task= self.tasks_ps[idx_task_base]
    #     # frame_base_task= self.tasks_frameestimators[idx_task_base]
    #     # frame_i_task= self.tasks_frameestimators[idx_task_i]

    #     # Sample points and their kernel frames.
    #     ps_base_batch = ps_base_task[torch.randint(0, self.ambient_dim, (self.b,))]
    #     ps_base_push=torch.matmul(lg_i, ps_base_batch.T).T

    #     # frame_base=frame_base_task.evaluate(ps_base_batch, bandwidth=self.bandwidth)
    #     # frame_i_push= frame_i_task.evaluate(ps_base_push, bandwidth=self.bandwidth)

    #     frame_base=self.rotation_vector_field(ps_base_batch, center=self.task_specifications[idx_task_base]['goal'])
    #     frame_i_push=self.rotation_vector_field(ps_base_push, center=self.task_specifications[idx_task_i]['goal'])

    #     # Compute the orthogonal complement of frame_base with respect to frame_i_push and vice versa
    #     ortho_frame_i_push=self._ortho_comp(frame_i_push, frame_base)
    #     ortho_frame_base= self._ortho_comp(frame_base, frame_i_push)

    #     # We minimize the two orthogonal complements. If both are zero, the frames are aligned.
    #     loss=torch.sum(torch.norm(ortho_frame_i_push, dim=(1,2)), dim=0) + torch.sum(torch.norm(ortho_frame_base, dim=(1,2)), dim=0)
    #     return loss

    
    def evalute_left_actions(self, lgs: torch.Tensor, idx_task_base: int, track_loss:bool=True):
        """Computes kernel alignment loss of all left-actions."""

        # TODO, idx_task_is must be dynamically inferred.
        idx_task_is = torch.arange(1, self._n_tasks)
        goals = torch.stack([torch.tensor(self.task_specifications[i]['goal']) for i in idx_task_is])
        
        ps_base_task = self.tasks_ps[idx_task_base]
        goal_base = self.task_specifications[idx_task_base]['goal']

        # 1. Sample points and push-forward
        ps = ps_base_task[torch.randint(0, self._n_samples, (self.batch_size,))]
        lg_ps = torch.einsum("Nnm,bm->Nbn", lgs, ps)  # shape (n_tasks-1, b, n)


        # 2. Sample tangent vectors and push-forward tangent vectors
        # TODO: this uses ground-truth frame for debugging.
        frame_ps= self.rotation_vector_field(ps, center=goal_base)
        lgs_frame_ps=torch.einsum("Nnm,bdm->Nbdn", lgs, frame_ps)  #(n_tasks, kernel_dim, ambient_dim)      
        frames_i_lg_ps=torch.vmap(self.rotation_vector_field)(lg_ps, goals)
        # frames_i_lg_ps = torch.stack([self.rotation_vector_field(lg_ps[i], center=goals[i]) for i in range(len(goals))])

    #     # frame_base_task= self.tasks_frameestimators[idx_task_base]
    #     # frame_i_task= self.tasks_frameestimators[idx_task_i]
    #     # frame_base=frame_base_task.evaluate(ps_base_batch, bandwidth=self.bandwidth)
    #     # frame_i_push= frame_i_task.evaluate(ps_base_push, bandwidth=self.bandwidth)



        # # Compute projection loss.
        # def pairwise_loss(frame_i_push, push_i_frame_base):
        #     ortho_frame_i_push = self._ortho_comp(frame_i_push, push_i_frame_base)
        #     ortho_frame_base = self._ortho_comp(push_i_frame_base, frame_i_push)
        #     loss = torch.sum(torch.norm(ortho_frame_i_push, dim=(1, 2))) + \
        #         torch.sum(torch.norm(ortho_frame_base, dim=(1, 2)))
        #     return loss

        # 3. Compute projection loss.
        # task_losses = torch.vmap(pairwise_loss)(frames_i_lg_ps, lgs_frame_ps)
        ortho_frame_i_lg_ps = self._ortho_comp(frames_i_lg_ps, lgs_frame_ps)
        ortho_lgs_frame_ps = self._ortho_comp(lgs_frame_ps, frames_i_lg_ps)
        task_losses = torch.sum(torch.norm(ortho_frame_i_lg_ps, dim=(1, 2)), dim=1) + \
            torch.sum(torch.norm(ortho_lgs_frame_ps, dim=(1, 2)), dim=1)
        

        if track_loss:
            self._losses.append(task_losses.detach().cpu().numpy())
        return task_losses.mean()
    

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



    # def _project_onto_subspace(self, vecs, basis):
    #     """Projects a set of vectors into a vector space spanned by basis.
    #     vecs: tensor of shape (b,d,n)
    #     basis: tensor of shape (b,d,n)
    #     """
    #     basis_t = basis.transpose(1, 2)  #(b,n,d)
        
    #     # Compute Gram matrix and its pseudo-inverse: (b,d,d)
    #     G = torch.matmul(basis, basis_t)  # (b,d,d)
    #     G_inv = torch.linalg.pinv(G)

    #     # Compute projection matrix: (b,n,n)
    #     P = torch.matmul(basis_t, torch.matmul(G_inv, basis))  # (b,n,n)
        
    #     # Project vecs: (b,d,n)
    #     return torch.matmul(vecs, P)  # (b,d,n)
    

    # def _ortho_comp(self, vecs, basis):
    #     """Computes the orthogonal complement of a set of vectors with respect to a set of bases.
    #     vecs: tensor of shape (b,d,n)
    #     basis: tensor of shape (b,d,n)

    #     Returns:
    #     - ortho_vecs: tensor of shape (b,d,n), the orthogonal complement of vecs with respect to basis.
    #     """
    #     proj_vecs = self._project_onto_subspace(vecs, basis)
    #     return vecs - proj_vecs
        
    def take_one_gradient_step_diff(self,
                      base_task_idx):
        
        loss = self.evalute_left_actions(lgs=self.lgs, idx_task_base=base_task_idx)

        self.optimizer_lgs.zero_grad()
        loss.backward()
        self.optimizer_lgs.step()

        return loss
    

    def optimize(self, n_steps:int=1000):
        """Optimizes the left action."""
        progress_bar = tqdm(range(n_steps), desc="Inferring left-action")

        for _ in progress_bar:
            loss = self.take_one_gradient_step_diff(
                base_task_idx=0)
            
            if n_steps% 50 == 0:
                progress_bar.set_description(f"Loss: {loss.item():.4e}")


    def rotation_vector_field(self, x, center):
        """
        Compute the rotational vector field directions at a batch of 2D points around a given center.

        Args:
            x (Tensor): Tensor of shape (N, 2) representing N 2D points.
            center (Tensor): Tensor of shape (2,) representing the center of rotation (a, b).

        Returns:
            Tensor: Tensor of shape (N, 2), each row is the direction of the vector field at the corresponding point in `x`.
        """
        warnings.warn("Using hard-coded kernel. Only use for debugging.")
        v = x - center  # shape (N, 2)
        rotated = torch.stack([-v[:, 1], v[:, 0]], dim=1)  # rotate 90° CCW
        return rotated.unsqueeze(1)