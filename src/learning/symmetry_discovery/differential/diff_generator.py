import random

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
                 optimizer):
        """
        Learns a maximal symmetry from point-wise bases.

        Args:
        - g_0: torch.tensor, initialization of the infinitesimal generator
        - p: torch.tensor, samples
        - bases: torch.tensor, point-wise bases
        - batch_size: int
        - n_steps: int, number of gradient steps
        - learn_symmetry: bool, whether to compute orthogonal complement of infinitesimal generator w.r.t. bases (default should be true)
        - learn_maximal: bool, whether to compute orthogonal complement of bases w.r.t. infinitesimal generator (default can be true or false, 
                         false may learn subsymmetry but more stable while true learns maximal symmetry but less stable)
        """
        
        self.g = g_0
        self.p = p
        self.bases = bases
        self.optimizer_diff = optimizer

        self._batch_size = batch_size
        self._n_steps = n_steps

        self._bases_idxs = list(self.bases.keys())
        self._losses = []
        self._losses_symmetry = []
        self._losses_maximal = []

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
        
        # Normalize infinitesimal generator to have unit Frobenius norm in each Lie group dimension.
        self.g_norm = self._normalize_tensor(tensor=generator, dim=(1,2))
        
        # Compute orthogonal complement of differential generator with respect to kernel bases and orth.complement of kernel bases with respect to diff. generator.
        orth_gen_on_kernel, orth_kernel_on_gen = self.orthogonal_projec(infinitesimal_generator=self.g_norm,
                                                                        ps = p_batch,
                                                                        kernel_bases = bases_batch)
        
        # Minimize orthogonal complements. Here, orth_gen_on_kernel and orth_kernel_on_gen are of shape (b,d,n) and we compute the norm along d and n
        # which is zero if each basis vector is contained in the span of the respective other vectors.
        loss_symmetry = torch.sum(torch.norm(orth_gen_on_kernel, dim=(1,2)), dim=0)
        loss_maximal = torch.sum(torch.norm(orth_kernel_on_gen, dim=(1,2)), dim=0)

        # Final loss
        loss = loss_symmetry + loss_maximal

        # Update generator
        self.optimizer_diff.zero_grad()
        loss.backward()
        self.optimizer_diff.step()

        self._losses_symmetry.append(loss_symmetry.detach().numpy())
        self._losses_maximal.append(loss_maximal.detach().numpy())
        self._losses.append(loss.detach().numpy())

    
    def _normalize_tensor(self, tensor, dim, norm='fro'):
        """Normalizes a tensor along given dimensions to have unit Frobenius norm."""
        _norm_tensor = tensor.norm(p=norm, dim=dim, keepdim=True) + 1e-8 #(d,d,n)
        return tensor/_norm_tensor
    

    def optimize(self):
        """Main Optimization Loop."""
        for _ in tqdm(range(self._n_steps)):

            idxs = random.sample(self._bases_idxs, self._batch_size)
            p_batch = self.p[idxs]
            bases_batch = torch.vstack([self.bases[i] for i in idxs]).unsqueeze(-1) #(b, n, d)

            # Normalize bases to have unit Frobenius norm
            bases_batch = self._normalize_tensor(tensor=bases_batch, dim=(1,2))

            self.take_one_gradient_step_diff(generator=self.g, p_batch=p_batch, bases_batch=bases_batch)

        return self.g_norm