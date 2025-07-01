from typing import Tuple, Dict
import warnings
import logging
logging.basicConfig(level=logging.INFO)

from tqdm import tqdm
import numpy as np
from scipy.spatial import KDTree
import torch

from constants import DTYPE

class PointwiseKernelApproximation():

    def __init__(self,
                 ps: torch.tensor,
                 ns: torch.tensor,
                 kernel_dim: int,
                 epsilon_ball: float,
                 epsilon_level_set: float):
        
        """Computes pointwise bases of a kernel of a function f: M \rightarrow N by performing a first degree Taylor expansion of f.
        
        Args:
        - p: torch.tensor of shape (n_samples, |M|).
        - n: torch.tensor of shape (n_samples, |N|).
        - kernel_dim: int, dimension of the kernel
        - epsilon_ball: float, radius of ball on which we Taylor approximate f.
        - epsilon_level_set: float, up to what tolerance are p,p' in the same level set, e.g. |f(p)-f(p')|< epsilon_level_set. Generally, epsilon_level_set should be smaller than epsilon_ball
        """

        self.ps = ps
        self.ns = ns
        self.kernel_dim = kernel_dim
        self.epsilon_ball = epsilon_ball
        self.epsilon_level_set = epsilon_level_set


    def compute(self) -> dict[int, torch.tensor]:
        """
        Computes a pointwise frame of the kernel distribution.
        Returns:
        - dict:
            - keys: indices of each data sample where the kernel frame could be computed.
            - values: basis vectors of shape (kernel_dim, |M])
        """
        warnings.warn("TODO: Dimension of kernel should be actively inferred, not passed as an argument.")
        kernel_vectors, _ = self._compute_kernel_samples(self.ps, self.ns, self.epsilon_ball, self.epsilon_level_set)
        return self._compute_pointwise_basis(kernel_vectors=kernel_vectors, kernel_dim=self.kernel_dim)


    def _compute_neighborhood(self, data, epsilon) -> list:
        """Computes the neighborhood of each point in data.
        Args:
            data: torch.Tensor of shape (n_samples, n_features)
            epsilon: float"

        Returns:
            list of lists of integers, where the ith element is the list of indices of the neighbors of the ith point in data.
        """
        tree = KDTree(data.numpy())
        return tree.query_ball_tree(tree, epsilon)


    def _compute_kernel_samples(self, ps, ns, epsilon_ball, epsilon_level_set) -> Tuple[Dict, Dict]:
        """
        Computes a pointwise approximation of samples from the kernel of $f$.

        Args:
            neighbors: list of lists of integers, where the ith element is the list of indices of the neighbors of the ith point in data.
            ns: torch.Tensor of shape (n_samples,), the readout of f at each point in data.
            epsilon_level_set: float, tolerance level for the level set.

        Returns:
            kernel_vectors: dictonary of length (n_samples,), each key is the index of a sample and the value is a tensor of shape (n_kernel_vectors, n_features) containing a sample from the kernel distribution
            self.local_level_set: TODO
        """
        warnings.warn("Kernel Approximation currently only supports real-valued functions.")
        _, n_features = ps.shape

        logging.info("Computing neighborhood of samples via kdtree...")
        neighbors = self._compute_neighborhood(ps, epsilon_ball)

        # a) Approximate which samples are (i) close to a given sample and (ii) belong to the same level.
        self.local_level_set = {}
        for idx, y in tqdm(enumerate(ns), desc="Locate samples in neighborhood...", total=len(ns)):

            x_neighbors_idxs = np.array(neighbors[idx])
            x_neighbors_idxs = x_neighbors_idxs[x_neighbors_idxs != idx] #remove the point itself

            x_neighbors_level_set_membership = np.array(torch.abs(ns[x_neighbors_idxs] - y) < epsilon_level_set)
            x_neighbors_level_set_idxs = x_neighbors_idxs[x_neighbors_level_set_membership]

            self.local_level_set[idx] = x_neighbors_level_set_idxs

        # b) Approximate the exponential map between samples in the same level set and in an \epsilon-ball via a linear approximation.
        kernel_vectors = {}
        for idx_sample, sample in tqdm(enumerate(ps), desc="Compute pointwise kernel samples...", total=len(ps)):
            kernel_vectors[idx_sample] = torch.zeros((len(self.local_level_set[idx_sample]), n_features), dtype=DTYPE)

            # for idx, neighbor_idx in enumerate(self.local_level_set[idx_sample]):
            #     kernel_vectors[idx_sample][idx] = ps[neighbor_idx] - sample

            neighbors = self.local_level_set[idx_sample]
            diffs = ps[neighbors] - sample
            _norms= torch.linalg.norm(diffs, dim=1, keepdim=True)
            diffs = torch.where(_norms > 0, diffs / _norms, diffs)
            kernel_vectors[idx_sample] = diffs

        return kernel_vectors, self.local_level_set


    def _compute_pointwise_basis(self,
                                 kernel_vectors: dict[int, torch.tensor],
                                kernel_dim: int) -> dict[int, torch.tensor]:
        """
        Computes a 1-D pointwise basis of the kernel distribution at each sample if there exists at least one non-trivial tangent vector in the kernel.

        Args:
            kernel_vectors: dictonary of length (n_samples,), each key is the index of a sample and the value is a tensor of shape (n_kernel_vectors, n_features) containing a sample from the kernel distribution
            kernel_dim: int, the dimension of the Kernel. TODO: this should be actively inferred.

        Returns:
            basis: dictonary of length (n_samples,), each key is the index of a sample and the value is a tensor of shape (n_kernel_vectors, n_features) containing a basis of the kernel distribution at the
        """
        basis = {}
        count_no_tangents, count_one_tangent, count_multiple_tangents = 0,0,0
        n_samples = len(kernel_vectors.keys())
        for idx_sample in tqdm(kernel_vectors.keys(), desc="Compute Point-Wise Bases via PCA..."):

            kernel_vectors_point = kernel_vectors[idx_sample]
            if len(kernel_vectors_point) == 0:
                # No non-trivial tangent vectors known in the kernel so we can't compute a basis.
                count_no_tangents+=1
            elif len(kernel_vectors_point) == 1:
                # Only one non-trivial kernel vector, use this as approximation of the kernel at p.
                # basis[idx_sample] = torch.nn.functional.normalize(kernel_vectors_point, p=2, dim=1)
                count_one_tangent+=1
            else:
                # There exists at least two non-trivial kernel vectors, compute a basis of dimension SYMMETRY_DIM of the Kernel distribution at the sample, normalized to have unit norm.
                _, _, _basis_vector = torch.pca_lowrank(kernel_vectors_point, q=kernel_dim)
                # _basis_vector = _basis_vector.T # PCA returns a vector of shape (1, n_features), we want it to be of shape (n_features, 1)
                # basis[idx_sample] = torch.nn.functional.normalize(_basis_vector, p=2, dim=1).flatten()
                basis[idx_sample] = _basis_vector
                count_multiple_tangents+=1
                if idx_sample==55:
                    breakpoint=True
            
        logging.info(
            f"Computed kernel bases from:\n"
            f"  - multiple tangent vectors for {round(100 * count_multiple_tangents / n_samples, 2)}% of samples (good)\n"
            f"  - one tangent vector for {round(100 * count_one_tangent / n_samples, 2)}% of samples (okay)\n"
            f"  - no tangent vector for {round(100 * count_no_tangents / n_samples, 2)}% of samples (not good, no basis)."
        )
        return basis