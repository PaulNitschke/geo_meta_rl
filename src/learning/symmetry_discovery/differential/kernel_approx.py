from typing import Tuple, Dict
import warnings
import logging
logging.basicConfig(level=logging.INFO)

from tqdm import tqdm
import numpy as np
from scipy.spatial import KDTree
import torch

from constants import DTYPE

def pointwise_kernel_approx(p: torch.tensor,
                            n: torch.tensor,
                            kernel_dim: int,
                            epsilon_ball: float,
                            epsilon_level_set: float) -> dict[int, torch.tensor]:
    """Computes pointwise bases of a kernel of a function f: M \rightarrow N by performing a first degree Taylor expansion of f.
    
    Args:
    - p: torch.tensor of shape (n_samples, |M|).
    - n: torch.tensor of shape (n_samples, |N|).
    - kernel_dim: int, dimension of the kernel
    - epsilon_ball: float, radius of ball on which we Taylor approximate f.
    - epsilon_level_set: float, up to what tolerance are p,p' in the same level set, e.g. |f(p)-f(p')|< epsilon_level_set. Generally, epsilon_level_set should be smaller than epsilon_ball

    Returns:
    - dict:
        - keys: indices of each data sample where the kernel could be computed.
        - values: basis vectors of shape
    """
    warnings.warn("TODO: Dimension of kernel should be actively inferred, not passed as an argument.")
    kernel_vectors, _ = _compute_kernel_samples(p, n, epsilon_ball, epsilon_level_set)
    return _compute_pointwise_basis(kernel_vectors=kernel_vectors, kernel_dim=kernel_dim)

def _compute_neighborhood(data, epsilon) -> list:
    """Computes the neighborhood of each point in data.
    Args:
        data: torch.Tensor of shape (n_samples, n_features)
        epsilon: float"

    Returns:
        list of lists of integers, where the ith element is the list of indices of the neighbors of the ith point in data.
    """
    tree = KDTree(data.numpy())
    return tree.query_ball_tree(tree, epsilon)


def _compute_kernel_samples(x_values, y_values, epsilon_ball, epsilon_level_set) -> Tuple[Dict, Dict]:
    """
    Computes a pointwise approximation of samples from the kernel of $f$.

    Args:
        neighbors: list of lists of integers, where the ith element is the list of indices of the neighbors of the ith point in data.
        y_values: torch.Tensor of shape (n_samples,), the readout of f at each point in data.
        epsilon_level_set: float, tolerance level for the level set.

    Returns:
        kernel_vectors: dictonary of length (n_samples,), each key is the index of a sample and the value is a tensor of shape (n_kernel_vectors, n_features) containing a sample from the kernel distribution
        local_level_set: TODO
    """
    warnings.warn("Kernel Approximation currently only supports real-valued functions.")
    _, n_features = x_values.shape

    logging.info("Computing neighborhood of samples via kdtree...")
    neighbors = _compute_neighborhood(x_values, epsilon_ball)

    # a) Approximate which samples are (i) close to a given sample and (ii) belong to the same level.
    local_level_set = {}
    for idx, y in tqdm(enumerate(y_values), desc="Locate samples in neighborhood...", total=len(y_values)):

        x_neighbors_idxs = np.array(neighbors[idx])
        x_neighbors_idxs = x_neighbors_idxs[x_neighbors_idxs != idx] #remove the point itself

        x_neighbors_level_set_membership = np.array(torch.abs(y_values[x_neighbors_idxs] - y) < epsilon_level_set)
        x_neighbors_level_set_idxs = x_neighbors_idxs[x_neighbors_level_set_membership]

        local_level_set[idx] = x_neighbors_level_set_idxs

    # b) Approximate the exponential map between samples in the same level set and in an \epsilon-ball via a linear approximation.
    kernel_vectors = {}
    for idx_sample, sample in tqdm(enumerate(x_values), desc="Compute pointwise kernel samples...", total=len(x_values)):
        kernel_vectors[idx_sample] = torch.zeros((len(local_level_set[idx_sample]), n_features), dtype=DTYPE)

        # for idx, neighbor_idx in enumerate(local_level_set[idx_sample]):
        #     kernel_vectors[idx_sample][idx] = x_values[neighbor_idx] - sample

        neighbors = local_level_set[idx_sample]
        diffs = x_values[neighbors] - sample
        kernel_vectors[idx_sample] = diffs

    return kernel_vectors, local_level_set


def _compute_pointwise_basis(kernel_vectors: dict[int, torch.tensor],
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
        
    logging.info(
        f"Computed kernel bases from:\n"
        f"  - multiple tangent vectors for {round(100 * count_multiple_tangents / n_samples, 2)}% of samples (good)\n"
        f"  - one tangent vector for {round(100 * count_one_tangent / n_samples, 2)}% of samples (okay)\n"
        f"  - no tangent vector for {round(100 * count_no_tangents / n_samples, 2)}% of samples (not good, no basis)."
    )
    return basis