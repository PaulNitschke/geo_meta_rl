import random
import torch

from src.learning.symmetry_discovery.differential.diff_generator import DiffGenerator
from src.learning.symmetry_discovery.functional.func_generator import FuncGenerator

class DiffFuncGenerator(DiffGenerator, FuncGenerator):
    """Learns a differential generator via both differential and functional symmetry discovery."""

    def __init__(self, 
                 g_0, 
                 p, 
                 bases,
                 func,
                 batch_size: int,
                 g_oracle,
                 n_steps: int):
        
        self.g_0_diff = torch.nn.Parameter(g_0.clone().detach().requires_grad_(True))
        self.g_0_func = torch.nn.Parameter(g_0.clone().detach().requires_grad_(True))

        self.optimizer_diff = torch.optim.Adam([self.g_0_diff], lr=0.00045)
        self.optimizer_func = torch.optim.Adam([self.g_0_func], lr=0.00045)

        DiffGenerator.__init__(self, self.g_0_diff, p, bases, batch_size, n_steps, optimizer=self.optimizer_diff)
        FuncGenerator.__init__(self, self.g_0_func, p, func, batch_size, n_steps=n_steps, optimizer=self.optimizer_func)

        self.p = p
        self.bases = bases
        self.func = func
        self.batch_size = batch_size,
        self.n_steps = n_steps
        self.g_oracle = g_oracle

        self.func_losses = []
        self.diff_losses = []
        self.oracle_losses = []

        self.func_losses_symmetry = []
        self.diff_losses_symmetry = []
        self.oracle_losses_symmetry = []

        self.func_losses_maximal = []
        self.diff_losses_maximal = []
        self.oracle_losses_maximal = []


    def take_one_gradient_step(self,
                               p_batch,
                               bases_batch,
                               group_coeffs_batch):
        """Takes a gradient step via both differental and functional symmetry discovery on the same data (but different generators)."""

        # Gradient Update.
        self.take_one_gradient_step_diff(generator=self.g_0_diff, p_batch=p_batch, bases_batch=bases_batch)
        self.take_one_gradient_step_func(generator=self.g_0_func, p_batch=p_batch, group_coeffs_batch=group_coeffs_batch, take_gradient_step=True)


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
        
        # Step 0: Evaluate the generators at different points:
        generator_ps = torch.einsum('bdnm,bm->bdn', generator_batched, p_batch)
        oracle_generator_ps = torch.einsum('bdnm,bm->bdn', oracle_generator_batched, p_batch)

        # Step 1: Check whether generator is a subspace of oracle generator at different points
        proj_gen_on_oracle = project_onto_subspace(generator_ps, oracle_generator_ps)
        orth_gen_on_oracle = generator_ps - proj_gen_on_oracle

        # Step 2: Check whether oracle generator is a subspace of generator
        proj_oracle_on_gen = project_onto_subspace(oracle_generator_ps, generator_ps)
        orth_oracle_on_gen = oracle_generator_ps - proj_oracle_on_gen

        loss_symmetry = torch.sum(torch.norm(orth_gen_on_oracle, dim=(1,2)), dim=0)
        loss_maximal = torch.sum(torch.norm(orth_oracle_on_gen, dim=(1,2)), dim=0)

        return loss_symmetry, loss_maximal, loss_symmetry + loss_maximal


    def sample_data(self):
        """"Samples batch size points from the manifold, kernel basis vector and coeffs for group actions."""
        _bases_idxs = list(self.bases.keys()) #Only sample from those points where we estimated a basis.
        idxs = random.sample(_bases_idxs, self._batch_size)
        p_batch = self.p[idxs]

        # Bases for differential discovery
        bases_batch = torch.vstack([self.bases[i] for i in idxs]).unsqueeze(-1) #(b, n, d)
        bases_batch = self._normalize_tensor(tensor=bases_batch, dim=(1,2))

        # Group actions for functional discovery
        # TODO, currently for symmetry group, change range from 2pi to arbitrary coefficients.
        group_coeffs_batch = torch.rand((self._batch_size, self._group_dim))*2*torch.pi

        return p_batch, bases_batch, group_coeffs_batch
            

    def optimize(self):

        for _ in range(self.n_steps):

            #1. Sample data
            p_batch, bases_batch, group_coeffs_batch = self.sample_data()

            #2. Take Gradient step.
            self.take_one_gradient_step(p_batch=p_batch, bases_batch=bases_batch, group_coeffs_batch=group_coeffs_batch)
            del p_batch, bases_batch, group_coeffs_batch

            #3. Evaluate on fresh data
            p_batch, _, _ = self.sample_data()
            with torch.no_grad():
                # Normalize as length of orthogonal component depends on length of basis.
                self.g_0_diff_norm = self._normalize_tensor(self.g_0_diff, dim=(1,2))
                self.g_0_func_norm = self._normalize_tensor(self.g_0_func, dim=(1,2))
                loss_diff_symmetry, loss_diff_maximal, loss_diff = self.evaluate_generator(p_batch, self.g_0_diff_norm, self.g_oracle)
                loss_func_symmetry, loss_func_maximal, loss_func = self.evaluate_generator(p_batch, self.g_0_func_norm, self.g_oracle)
                # TODO: hard-coded oracle prime, change this to a rotation of self.g_oracle
                g_oracle_prime = torch.tensor([[0, 1], [-1, 0]], dtype=DTYPE).unsqueeze(0)
                loss_oracle_symmetry, loss_oracle_maximal, loss_oracle = self.evaluate_generator(p_batch, g_oracle_prime, self.g_oracle)


                self.diff_losses.append(loss_diff.detach().numpy())
                self.func_losses.append(loss_func.detach().numpy())
                self.oracle_losses.append(loss_oracle.detach().numpy())

                self.diff_losses_symmetry.append(loss_diff_symmetry.detach().numpy())
                self.func_losses_symmetry.append(loss_func_symmetry.detach().numpy())
                self.oracle_losses_symmetry.append(loss_oracle_symmetry.detach().numpy())

                self.diff_losses_maximal.append(loss_diff_maximal.detach().numpy())
                self.func_losses_maximal.append(loss_func_maximal.detach().numpy())
                self.oracle_losses_maximal.append(loss_oracle_maximal.detach().numpy())