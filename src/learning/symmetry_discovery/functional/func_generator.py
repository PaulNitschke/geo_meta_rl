import random

from tqdm import tqdm
import torch

class FuncGenerator():

    def __init__(self,
                 g_0, 
                 p,
                 f: callable,
                 batch_size: int,
                 optimizer,
                 n_steps: int):
        """
        Learns a differential generator via functional symmetry discovery of f: M \rightarrow N.

        Args:
        - g_0: torch.tensor, initialization of the infinitesimal generator
        - p: torch.tensor of shape (n_samples, |M|), samples
        - f: callable, the function for which we aim to learn a symmetry.
        - batch_size: int
        - n_steps: int, number of gradient steps
        """
        
        self.g = g_0
        self.p = p
        self.f = f

        self._optimizer_func = optimizer

        self._batch_size = batch_size
        self._n_steps = n_steps
        self._group_dim = self.g.shape[0]

        self._losses = []


    def take_one_gradient_step_func(self,
                      generator,
                      p_batch,
                      group_coeffs_batch,
                      take_gradient_step: bool = True
                      ):
        
        # Normalize infinitesimal generator to have unit Frobenius norm in each Lie group dimension.
        self.g_norm = self._normalize_tensor(tensor=generator, dim=(1,2))
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

        # Update generator, can be turned off to evaluate the current generator.
        if take_gradient_step:
            self._optimizer_func.zero_grad()
            loss.backward()
            self._optimizer_func.step()

        self._losses.append(loss.detach().numpy())

        return loss

    
    def _normalize_tensor(self, tensor, dim, norm='fro'):
        """Normalizes a tensor along given dimensions to have unit Frobenius norm."""
        _norm_tensor = tensor.norm(p=norm, dim=dim, keepdim=True) + 1e-8 #(d,d,n)
        return tensor/_norm_tensor
    

    def optimize(self):
        """Main Optimization Loop."""
        for _ in tqdm(range(self._n_steps)):

            idxs = random.sample(list(range(len(self.p))), self._batch_size)
            p_batch = self.p[idxs]

            # Group actions
            # TODO, currently for symmetry group, change range from 2pi to arbitrary coefficients.
            coeffs_group_batch = torch.rand((self._batch_size, self._group_dim))*2*torch.pi

            self.take_one_gradient_step_func(generator=self.g, p_batch=p_batch, group_coeffs_batch=coeffs_group_batch)

        return self.g_norm