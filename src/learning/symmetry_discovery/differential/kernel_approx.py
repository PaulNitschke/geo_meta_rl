from typing import Tuple, Dict, Optional
import warnings
import logging
logging.basicConfig(level=logging.INFO)

from tqdm import tqdm
import numpy as np
import pickle
from scipy.spatial import KDTree
import torch
import faiss

from constants import DTYPE

class KernelFrameEstimator():

    def __init__(self,
                ps: torch.Tensor,
                kernel_dim: int,
                ns: Optional[torch.Tensor] = None,
                epsilon_ball: Optional[float] = None,
                epsilon_level_set: Optional[float] = None):
        
        """Computes pointwise bases (a frame) of a kernel of a function f: M \rightarrow N by performing a first degree Taylor expansion of f.
        
        Args:
        - ps: torch.tensor of shape (n_samples, |M|).
        - kernel_dim: int, dimension of the kernel
        - ns: torch.tensor of shape (n_samples, |N|). Not required for inference only.
        - epsilon_ball: float, radius of ball on which we Taylor approximate f. Not required for inference only.
        - epsilon_level_set: float, up to what tolerance are p,p' in the same level set, e.g. |f(p)-f(p')|< epsilon_level_set. Generally, epsilon_level_set should be smaller than epsilon_ball
                                Not required for inference only.
        """

        self.ps = ps
        self.ns = ns
        self.kernel_dim = kernel_dim
        self.epsilon_ball = epsilon_ball
        self.epsilon_level_set = epsilon_level_set
        self.pointwise_frame = {}
        self._finished_setup_evaluation=False


    def compute(self) -> dict[int, torch.tensor]:
        """
        Computes a pointwise frame of the kernel distribution.
        Returns:
        - dict:
            - keys: indices of each data sample where the kernel frame could be computed.
            - values: basis vectors of shape (kernel_dim, |M])
        """
        assert self.ns is not None, "Kernel approximation requires a readout of the function f at each point p."
        assert self.epsilon_ball is not None, "Kernel approximation requires a radius of the ball on which we Taylor approximate f."
        assert self.epsilon_level_set is not None, "Kernel approximation requires a tolerance level for the level set of f."

        warnings.warn("TODO: Dimension of kernel should be actively inferred, not passed as an argument.")
        kernel_vectors, _ = self._compute_kernel_samples(self.ps, self.ns, self.epsilon_ball, self.epsilon_level_set)
        return self._compute_pointwise_basis(kernel_vectors=kernel_vectors, kernel_dim=self.kernel_dim)
    

    def save(self, file_name:str):
        """Saves the frame samples to file."""
        with open(file_name, 'wb') as f:
            pickle.dump(self.pointwise_frame, f)

    
    def set_frame(self, frame: dict[int, torch.tensor]):
        """Sets the pointwise frame of the kernel distribution and initializes the kernel evaluation."""
        self.pointwise_frame = frame
        self.setup_evaluation()


    def setup_evaluation(self):
        """Sets up the sampling from the kernel distribution by building an approximate k-nearest neighbor graph."""
        self.known_idx = torch.tensor(list(self.pointwise_frame.keys()))
        self.known_ps = self.ps[self.known_idx]
        self.known_frames = torch.stack([self.pointwise_frame[int(i)] for i in self.known_idx])
        self._finished_setup_evaluation=True
        logging.info("Setup kernel frame evaluation.")

    def evaluate(self, 
                    ps: torch.Tensor,
                    bandwidth: float,
                    threshold: float = 0.1) -> torch.Tensor:
        """
        Computes a frame of the kernel distribution via a gaussian kernel at a batch of points..

        Args:
            P: torch.Tensor of shape (b, n), batch of query points.
            bandwidth: float, bandwidth of the Gaussian kernel.
            threshold: float, relative cutoff for kernel weights.
        
        Returns:
            torch.Tensor of shape (B, m, kernel_dim)
        """
        assert self._finished_setup_evaluation, "Call setup_evaluation() before evaluate()."
        assert ps.dim()==2, "Input ps must be a 2D tensor of shape (b, n)."


        dist2 = torch.cdist(ps, self.known_ps, p=2)**2
        weights = torch.exp(-dist2 / (2 * bandwidth**2))

        max_weights, _ = weights.max(dim=1, keepdim=True)
        mask = weights > threshold * max_weights
        weights = weights * mask  # (B, N)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        weighted = (weights[:, :, None, None] * self.known_frames[None, :, :, :]).sum(dim=1)
        U, _, _ = torch.linalg.svd(weighted, full_matrices=False)
        return U[:, :, :self.kernel_dim]


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
        count_no_tangents, count_one_tangent, count_multiple_tangents = 0,0,0
        n_samples = len(kernel_vectors.keys())
        for idx_sample in tqdm(kernel_vectors.keys(), desc="Compute Point-Wise Bases via PCA..."):

            kernel_vectors_point = kernel_vectors[idx_sample]
            if len(kernel_vectors_point) == 0:
                # No non-trivial tangent vectors known in the kernel so we can't compute a basis.
                count_no_tangents+=1
            elif len(kernel_vectors_point) == 1:
                # Only one non-trivial kernel vector, use this as approximation of the kernel at p.
                # self.pointwise_frame[idx_sample] = torch.nn.functional.normalize(kernel_vectors_point, p=2, dim=1)
                count_one_tangent+=1
            else:
                # There exists at least two non-trivial kernel vectors, compute a basis of dimension SYMMETRY_DIM of the Kernel distribution at the sample, normalized to have unit norm.
                _, _, _basis_vector = torch.pca_lowrank(kernel_vectors_point, q=kernel_dim)
                self.pointwise_frame[idx_sample] = _basis_vector
                count_multiple_tangents+=1

            
        logging.info(
            f"Computed kernel bases from:\n"
            f"  - multiple tangent vectors for {round(100 * count_multiple_tangents / n_samples, 2)}% of samples (good)\n"
            f"  - one tangent vector for {round(100 * count_one_tangent / n_samples, 2)}% of samples (okay)\n"
            f"  - no tangent vector for {round(100 * count_no_tangents / n_samples, 2)}% of samples (not good, no basis)."
        )
        return self.pointwise_frame