import random
import warnings

from tqdm import tqdm
import torch

class DiffGenerator():
    """Learns a differential generator given pointwise bases of the Kernel."""

    def __init__(self,
                 g_0, 
                 p, 
                 bases,
                 batch_size: int,
                 n_steps: int,
                 optimizer,
                 random_seed: int=None,
                 track_hessian: bool=False):
        """
        Learns a maximal symmetry from point-wise bases.

        Args:
        - g_0: torch.tensor, initialization of the infinitesimal generator
        - p: torch.tensor, samples
        - bases: torch.tensor, point-wise bases
        - batch_size: int
        - n_steps: int, number of gradient steps
                         false may learn subsymmetry but more stable while true learns maximal symmetry but less stable)
        - track_hessian: bool, whether to track the hessian of the loss function
        """
        
        self.g = g_0
        self.p = p
        self.bases = bases
        self.optimizer_diff = optimizer

        self._batch_size = batch_size
        self._n_steps = n_steps
        self._track_hessian = track_hessian

        self._bases_idxs = list(self.bases.keys())
        self._losses = []
        self._losses_symmetry = []
        self._losses_maximal = []
        if self._track_hessian:
            self.gs = []
            self.hessians = []

        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        warnings.warn("Differential Generator is not Normalized During Training")

    def orthogonal_projec(self,
                          infinitesimal_generator, 
                          ps, 
                          kernel_bases):
        """
        Projects a batched point p via a projection and computes its orthogonal component with respect to a given subspace.
        Args:
            infinitesimal_generator: (d, n, n)
            p: (b, n)
            kernel_bases: (b, n, d)
        Returns:
            p_ortho: (b, n)

        where:
        - b: batch size
        - d, f: generator dimension, d=f
        - n, m: manifold ambient dimensions, n=m
        """
        #TODO, fix this, kernel bases should have correct shape from the beginning.
        kernel_bases = kernel_bases.transpose(1,2) #(b,d,n)


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
        

        # Batch Infinitesimal Generator
        infinitesimal_generator_batched_ = infinitesimal_generator.unsqueeze(0)
        infinitesimal_generator_batched = infinitesimal_generator_batched_.repeat(self._batch_size, 1, 1, 1)
        del infinitesimal_generator_batched_, infinitesimal_generator

        # Step 1: Use infinitesimal generator to compute differential generator bases at points p according to infinitesimal generator.
        gen_bases = torch.einsum('bdnm,bm->bdn', infinitesimal_generator_batched, ps)

        # Step 2a: Symmetry: Check whether each basis vector in kernel_bases_diff_gen is in the span of kernel_bases.
        # To this end, compute orthogonal complement:
        proj_gen_on_kernel = project_onto_subspace(gen_bases, kernel_bases)
        orth_gen_on_kernel = gen_bases - proj_gen_on_kernel


        # Step 2b: Maximal: Check whether each kernel basis vector is in the span of gen_bases.
        proj_kernel_on_gen = project_onto_subspace(kernel_bases, gen_bases)
        orth_kernel_on_gen = kernel_bases - proj_kernel_on_gen


        return orth_gen_on_kernel, orth_kernel_on_gen
    

    def take_one_gradient_step_diff(self,
                      generator,
                      p_batch,
                      bases_batch):
        
        loss = self._compute_loss(generator, p_batch, bases_batch, track_losses=True)

        # Update generator
        self.optimizer_diff.zero_grad()
        loss.backward()
        self.optimizer_diff.step()

    
    def _compute_loss(self, generator, p_batch, bases_batch, track_losses: bool=False):
        """Computes Main loss: projects generator into kernel and kernel into generator."""
        # Normalize infinitesimal generator to have unit Frobenius norm in each Lie group dimension.
        # self.g_norm = self._normalize_tensor(tensor=generator, dim=(1,2))
        self.g_norm = generator
        # Compute orthogonal complement of differential generator with respect to kernel bases and orth.complement of kernel bases with respect to diff. generator.
        orth_gen_on_kernel, orth_kernel_on_gen = self.orthogonal_projec(infinitesimal_generator=self.g_norm,
                                                                        ps = p_batch,
                                                                        kernel_bases = bases_batch)
        
        # Minimize orthogonal complements. Here, orth_gen_on_kernel and orth_kernel_on_gen are of shape (b,d,n) and we compute the norm along d and n
        # which is zero if each basis vector is contained in the span of the respective other vectors.
        loss_symmetry = torch.sum(torch.norm(orth_gen_on_kernel, dim=(1,2)), dim=0)
        loss_maximal = torch.sum(torch.norm(orth_kernel_on_gen, dim=(1,2)), dim=0)
        loss = loss_symmetry + loss_maximal

        if track_losses:
            self._losses_symmetry.append(loss_symmetry.detach().numpy())
            self._losses_maximal.append(loss_maximal.detach().numpy())
            self._losses.append(loss.detach().numpy())

        return loss        


    def _compute_hessian(self, generator, p_batch, bases_batch) -> torch.tensor:
        """Computes the Hessian of the loss function at a generator.
        Returns:
        - Hessian: tensor of shape (|M|^2, |M|^2)
        """

        generator_hessian = generator.detach().clone().requires_grad_(True)
        loss_fn = lambda gen: self._compute_loss(gen, p_batch, bases_batch)
        H = torch.autograd.functional.hessian(loss_fn, generator_hessian)
        return H.reshape(generator.numel(), generator.numel())


    def _normalize_tensor(self, tensor, dim, norm='fro'):
        """Normalizes a tensor along given dimensions to have unit Frobenius norm."""
        _norm_tensor = tensor.norm(p=norm, dim=dim, keepdim=True) + 1e-8 #(d,d,n)
        return tensor/_norm_tensor
    

    def _sample_data(self, n_samples: int):
        """Uniformly samples n_samples points and their corresponding kernel bases."""
        idxs = random.sample(self._bases_idxs, n_samples)
        p_batch = self.p[idxs]
        bases_batch = torch.vstack([self.bases[i] for i in idxs]).unsqueeze(-1) #(b, n, d)

        # Normalize bases to have unit Frobenius norm
        bases_batch = self._normalize_tensor(tensor=bases_batch, dim=(1,2))

        return p_batch, bases_batch
    

    def optimize(self):
        """Main Optimization Loop."""
        for _ in tqdm(range(self._n_steps)):

            #Sample data
            p_batch, bases_batch = self._sample_data(n_samples=self._batch_size)

            # Track Hessian of loss function.
            if self._track_hessian:
                self.hessians.append(self._compute_hessian(self.g, p_batch, bases_batch))
                self.gs.append(self.g.detach().clone())

            self.take_one_gradient_step_diff(generator=self.g, p_batch=p_batch, bases_batch=bases_batch)


        return self.g_norm