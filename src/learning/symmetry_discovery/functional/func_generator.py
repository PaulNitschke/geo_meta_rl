import random
import warnings

from tqdm import tqdm
import torch

class FuncGenerator():

    def __init__(self,
                 g_0, 
                 p,
                 f: callable,
                 batch_size: int,
                 optimizer,
                 n_steps: int,
                 random_seed: int=None,
                 track_hessian: bool=False):
        """
        Learns a differential generator via functional symmetry discovery of f: M \rightarrow N.

        Args:
        - g_0: torch.tensor, initialization of the infinitesimal generator
        - p: torch.tensor of shape (n_samples, |M|), samples
        - f: callable, the function for which we aim to learn a symmetry.
        - batch_size: int
        - n_steps: int, number of gradient steps
        - track_hessian: bool, whether to track the hessian of the loss function
        """
        
        self.g = g_0
        self.p = p
        self.f = f

        self._optimizer_func = optimizer
        self._track_hessian = track_hessian

        self._batch_size = batch_size
        self._n_steps = n_steps
        self._group_dim = self.g.shape[0]

        self._losses = []
        if self._track_hessian:
            self.gs = []
            self.hessians = []
            
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        warnings.warn("Sampled group actions are in 2pi range.")

    def take_one_gradient_step_func(self,
                      generator,
                      p_batch,
                      group_coeffs_batch,
                      take_gradient_step: bool = True
                      ):
        
        loss=self._compute_loss_func(generator, p_batch, group_coeffs_batch)

        # Update generator, can be turned off to evaluate the current generator.
        if take_gradient_step:
            self._optimizer_func.zero_grad()
            loss.backward()
            self._optimizer_func.step()

        self._losses.append(loss.detach().numpy())

        return loss
    

    def _compute_loss_func(self, generator, p_batch, group_coeffs_batch) -> float:
        """Computes the loss of the current generator via functional symmetry."""
        # Normalize infinitesimal generator to have unit Frobenius norm in each Lie group dimension.
        # self.g_norm = self._normalize_tensor(tensor=generator, dim=(1,2))
        self.g_norm = generator
        self.g_norm_batched_ = self.g_norm.unsqueeze(0)
        self.g_norm_batched = self.g_norm_batched_.repeat(self._batch_size, 1, 1, 1) #(b, d, m, n)

        #
        group_coeffs_batch = group_coeffs_batch.view(self._batch_size, 1, 1, 1) #(b, 1, 1, 1)
    
        # Compute group actions via matrix exponential
        diff_group_action = torch.sum(group_coeffs_batch*self.g_norm_batched, dim=1) #(b,m,n) #TODO, is this correct
        group_actions_batched = torch.matrix_exp(diff_group_action)

        # Compute loss
        image_f_raw = self.f(p_batch)
        image_f_group = self.f(torch.einsum('bnm,bm->bn', group_actions_batched, p_batch))
        loss =  torch.norm(image_f_raw - image_f_group, dim=-1).sum(dim=0)
        return loss
    

    def _compute_hessian(self, generator, p_batch, group_coeffs_batch) -> torch.tensor:
        """Computes the Hessian of the loss function at the current generator.
        Returns:
        - Hessian: tensor of shape (|M|^2, |M|^2)
        """

        generator_hessian = generator.detach().clone().requires_grad_(True)
        loss_fn = lambda gen: self._compute_loss_func(gen, p_batch, group_coeffs_batch)
        H = torch.autograd.functional.hessian(loss_fn, generator_hessian)
        return H.reshape(generator.numel(), generator.numel())

    
    def _normalize_tensor(self, tensor, dim, norm='fro'):
        """Normalizes a tensor along given dimensions to have unit Frobenius norm."""
        _norm_tensor = tensor.norm(p=norm, dim=dim, keepdim=True) + 1e-8 #(d,d,n)
        return tensor/_norm_tensor
    

    def _sample_data_func(self, n_samples):
        """Uniformly samples n_samples points and group action coefficients."""
        idxs = random.sample(list(range(len(self.p))), n_samples, )
        p_batch = self.p[idxs]

        # Group actions
        # TODO, currently for symmetry group, change range from 2pi to arbitrary coefficients.
        coeffs_group_batch = torch.rand((self._batch_size, self._group_dim), )*2*torch.pi

        return p_batch, coeffs_group_batch
    

    def optimize(self):
        """Main Optimization Loop."""
        for _ in tqdm(range(self._n_steps)):

            #Sample data
            p_batch, coeffs_group_batch = self._sample_data_func(n_samples=self._batch_size)

            # Track Hessian of loss function.
            if self._track_hessian:
                self.hessians.append(self._compute_hessian(self.g, p_batch, coeffs_group_batch))
                self.gs.append(self.g.detach().clone())

            self.take_one_gradient_step_func(generator=self.g, p_batch=p_batch, group_coeffs_batch=coeffs_group_batch)


        return self.g