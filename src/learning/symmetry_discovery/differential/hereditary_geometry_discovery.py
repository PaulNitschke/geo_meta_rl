import torch
import warnings
from tqdm import tqdm

class HereditaryGeometryDiscovery():

    def __init__(self,
                 tasks_ps: list,
                 tasks_frameestimators: list,
                 kernel_dim: int,
                 batch_size:int=64,
                 knn:int=10,
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

        self.k= knn
        self.bandwidth=bandwidth

        self.d=kernel_dim
        self.n=tasks_ps[0][0,:].shape[0]
        self.b=batch_size

        # Optimization variables
        warnings.warn("Only one left-action implemented.")
        self.lg_i=torch.tensor(torch.randn(self.n, self.n), requires_grad=True)
        self.optimizer_lg_i=torch.optim.Adam([self.lg_i], lr=1e-3)


    def evaluate_left_action(self, lg_i:torch.tensor, idx_task_base:int, idx_task_i:int):
        """Evaluates the left action of a group element lg_i on a base task ps_base_task and a target task ps_i_task."""

        ps_base_task= self.tasks_ps[idx_task_base]
        frame_base_task= self.tasks_frameestimators[idx_task_base]
        frame_i_task= self.tasks_frameestimators[idx_task_i]

        # Sample points and their kernel frames.
        ps_base_batch = ps_base_task[torch.randint(0, self.n, (self.b,))]
        ps_base_push=torch.matmul(lg_i, ps_base_batch.T).T
        frame_base=self._sample_frame(ps_base_task, frame_base_task)
        frame_i_push= self._sample_frame(ps_base_push, frame_i_task)

        # Compute the orthogonal complement of frame_base with respect to frame_i_push and vice versa
        ortho_frame_i_push=self._ortho_comp(frame_i_push, frame_base)
        ortho_frame_base= self._ortho_comp(frame_base, frame_i_push)

        # We minimize the two orthogonal complements. If both are zero, the frames are aligned.
        loss=torch.sum(torch.norm(ortho_frame_i_push, dim=(1,2)), dim=0) + torch.sum(torch.norm(ortho_frame_base, dim=(1,2)), dim=0)

    def _sample_frame(self, ps, frame_evaluator):
        """Samples a kernel frame at a batch of points.
        Args:
        - ps, torch.tensor of shape (b,n).
        - frame_evaluator, KernelFrameEstimator instance.
        
        Returns:
        - frame_ps, torch.tensor of shape (b,d,n), the frame of the kernel evaluated at ps.
        """
        frame_ps=torch.zeros(size=(self.b, self.d, self.n))
        for i in range(self.b):
            #TODO, kernel_approx.py computes the frame of shape (n,d) while we need (d,n)
            frame_ps[i]=frame_evaluator.evaluate(ps[i], k=self.k, bandwidth=self.bandwidth).T


    def _project_onto_subspace(self, vecs, basis):
        """Projects a set of vectors into a vector space spanned by basis.
        vecs: tensor of shape (b,d,n)
        basis: tensor of shape (b,d,n)
        """
        basis_t = basis.transpose(1, 2)  #(b,n,d)
        
        # Compute Gram matrix and its pseudo-inverse: (b,d,d)
        G = torch.matmul(basis, basis_t)  # (b,d,d)
        G_inv = torch.linalg.pinv(G)

        # Compute projection matrix: (b,n,n)
        P = torch.matmul(basis_t, torch.matmul(G_inv, basis))  # (b,n,n)
        
        # Project vecs: (b,d,n)
        return torch.matmul(vecs, P)  # (b,d,n)
    

    def _ortho_comp(self, vecs, basis):
        """Computes the orthogonal complement of a set of vectors with respect to a set of bases.
        vecs: tensor of shape (b,d,n)
        basis: tensor of shape (b,d,n)

        Returns:
        - ortho_vecs: tensor of shape (b,d,n), the orthogonal complement of vecs with respect to basis.
        """
        try:
            proj_vecs = self._project_onto_subspace(vecs, basis)
        except:
            breakpoint=True
        return vecs - proj_vecs
        
    def take_one_gradient_step_diff(self,
                      base_task_idx,
                      task_i_idx):
        
        loss = self.evaluate_left_action(
            lg_i=self.lg_i,
            idx_task_base=base_task_idx,
            idx_task_i=task_i_idx
        )

        # Update generator
        self.optimizer_lg_i.zero_grad()
        loss.backward()
        self.optimizer_lg_i.step()

        return loss
    

    def optimize(self, n_steps:int=1000):
        """Optimizes the left action."""
        progress_bar = tqdm(range(n_steps), desc="Inferring left-action")

        for _ in progress_bar:
            loss = self.take_one_gradient_step_diff(
                base_task_idx=0,
                task_i_idx=1
            )
            progress_bar.set_description(f"Loss: {loss.item():.4e}")