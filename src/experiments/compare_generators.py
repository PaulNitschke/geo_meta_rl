import random
import warnings

import torch
from tqdm import tqdm

from constants import DTYPE
from src.learning.symmetry_discovery.diff_generator import DiffGenerator
from src.learning.symmetry_discovery.func_generator import FuncGenerator

class DiffFuncGenerator(DiffGenerator, FuncGenerator):
    """Learns a differential generator via both differential and functional symmetry discovery."""

    def __init__(self, 
                 g_0, 
                 p, 
                 bases,
                 func,
                 batch_size: int,
                 g_oracle,
                 n_steps: int,
                 random_seed: int=None):
        
        if random_seed is None:
            warnings.warn("No random seed set.")
        
        self.g_diff = torch.nn.Parameter(g_0.clone().detach().requires_grad_(True))
        self.g_func = torch.nn.Parameter(g_0.clone().detach().requires_grad_(True))

        self.optimizer_diff = torch.optim.Adam([self.g_diff], lr=0.00045)
        self.optimizer_func = torch.optim.Adam([self.g_func], lr=0.00045)

        DiffGenerator.__init__(self, self.g_diff, p, bases, batch_size, n_steps, optimizer=self.optimizer_diff, random_seed=random_seed)
        FuncGenerator.__init__(self, self.g_func, p, func, batch_size, n_steps=n_steps, optimizer=self.optimizer_func, random_seed=random_seed)
        warnings.warn("Current evaluation only supports rotation symmetry.")

        self.p = p
        self.bases = bases
        self.func = func
        self.batch_size = batch_size,
        self.n_steps = n_steps
        self.g_oracle = g_oracle

        self.func_losses, self.diff_losses, self.oracle_losses = [], [], []
        self.func_losses_symmetry, self.diff_losses_symmetry, self.oracle_losses_symmetry = [], [], []
        self.func_losses_maximal, self.diff_losses_maximal, self.oracle_losses_maximal = [], [], []
        self.func_losses_func_space, self.diff_losses_func_space, self.oracle_losses_func_space = [], [], []


    def take_one_gradient_step(self,
                               p_batch_diff,
                               bases_batch,
                               p_batch_func,
                               group_coeffs_batch):
        """Takes a gradient step via both differental and functional symmetry discovery on the same data (but different generators)."""

        # Gradient Update.
        self.take_one_gradient_step_diff(generator=self.g_diff, p_batch=p_batch_diff, bases_batch=bases_batch)
        self.take_one_gradient_step_func(generator=self.g_func, p_batch=p_batch_func, group_coeffs_batch=group_coeffs_batch, take_gradient_step=True)

    def evaluate_generator(self, p_batch, generator, oracle_generator):
        """Evaluates a learned generator against a ground truth generator by checking whether they span the same subspaces at different points."""
        
        def project_onto_subspace(vecs, basis):
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
        
        # Step 0: Batch
        generator_batched = generator.unsqueeze(0)
        generator_batched = generator_batched.repeat(self._batch_size, 1, 1, 1)
        oracle_generator_batched = oracle_generator.unsqueeze(0)
        oracle_generator_batched = oracle_generator_batched.repeat(self._batch_size, 1, 1, 1)
        
        # Step 1: Evaluate the generators at different points:
        generator_ps = torch.einsum('bdnm,bm->bdn', generator_batched, p_batch)
        oracle_generator_ps = torch.einsum('bdnm,bm->bdn', oracle_generator_batched, p_batch)

        # Step 2: Check whether generator is a subspace of oracle generator at different points. This is the symmetry part where
        # the learned generator spans no more than the subspace of the oracle.
        proj_gen_on_oracle = project_onto_subspace(generator_ps, oracle_generator_ps)
        orth_gen_on_oracle = generator_ps - proj_gen_on_oracle

        # Step 3: Check whether oracle generator is a subspace of generator. This is the maximal part where the learned generator
        # spans at least the subspace of the oracle.
        proj_oracle_on_gen = project_onto_subspace(oracle_generator_ps, generator_ps)
        orth_oracle_on_gen = oracle_generator_ps - proj_oracle_on_gen

        loss_symmetry = torch.sum(torch.norm(orth_gen_on_oracle, dim=(1,2)), dim=0)
        loss_maximal = torch.sum(torch.norm(orth_oracle_on_gen, dim=(1,2)), dim=0)

        return loss_symmetry, loss_maximal, loss_symmetry + loss_maximal
            

    def optimize(self):

        pbar = tqdm(range(self.n_steps), desc="Learning Differential and Functional Generator") 
        for idx_step in pbar:

            #1. Sample data
            p_batch_diff, bases_batch = self._sample_data_diff(n_samples=self._batch_size)
            p_batch_func, group_coeffs_batch = self._sample_data_func(n_samples=self._batch_size)

            #2. Take Gradient step.
            self.take_one_gradient_step(p_batch_diff=p_batch_diff, bases_batch=bases_batch, p_batch_func=p_batch_func, group_coeffs_batch=group_coeffs_batch)
            del p_batch_diff, bases_batch, p_batch_func, group_coeffs_batch

            with torch.no_grad():
            #3. Evaluate on fresh data
                p_batch, _ = self._sample_data_diff(n_samples=self._batch_size)
                _, group_coeffs_batch = self._sample_data_func(n_samples=self._batch_size)

                # Generator Span Evaluation
                ## Normalize as length of orthogonal component depends on length of basis.
                self.g_diff_norm = self._normalize_tensor(self.g_diff, dim=(1,2))
                self.g_func_norm = self._normalize_tensor(self.g_func, dim=(1,2))
                loss_diff_symmetry, loss_diff_maximal, loss_diff = self.evaluate_generator(p_batch, self.g_diff_norm, self.g_oracle)
                loss_func_symmetry, loss_func_maximal, loss_func = self.evaluate_generator(p_batch, self.g_func_norm, self.g_oracle)
                loss_oracle_symmetry, loss_oracle_maximal, loss_oracle = self.evaluate_generator(p_batch, self.g_oracle, self.g_oracle)

                # Function Level Evaluation.
                loss_func_func_space = self._compute_loss_func(self.g_func, p_batch=p_batch, group_coeffs_batch=group_coeffs_batch)
                loss_diff_func_space = self._compute_loss_func(self.g_diff, p_batch=p_batch, group_coeffs_batch=group_coeffs_batch)
                loss_oracle_func_space = self._compute_loss_func(self.g_oracle, p_batch=p_batch, group_coeffs_batch=group_coeffs_batch)

                self.diff_losses.append(loss_diff.detach().numpy())
                self.func_losses.append(loss_func.detach().numpy())
                self.oracle_losses.append(loss_oracle.detach().numpy())

                self.diff_losses_symmetry.append(loss_diff_symmetry.detach().numpy())
                self.func_losses_symmetry.append(loss_func_symmetry.detach().numpy())
                self.oracle_losses_symmetry.append(loss_oracle_symmetry.detach().numpy())

                self.diff_losses_maximal.append(loss_diff_maximal.detach().numpy())
                self.func_losses_maximal.append(loss_func_maximal.detach().numpy())
                self.oracle_losses_maximal.append(loss_oracle_maximal.detach().numpy())

                self.func_losses_func_space.append(loss_func_func_space.detach().numpy())
                self.diff_losses_func_space.append(loss_diff_func_space.detach().numpy())
                self.oracle_losses_func_space.append(loss_oracle_func_space.detach().numpy())
            
            if idx_step % 100 == 0:
                pbar.set_postfix({'Generator Span Diff. Loss': f'{loss_diff:.2f}', 'Generator Span Func. Loss': f'{loss_func:.2f}'})