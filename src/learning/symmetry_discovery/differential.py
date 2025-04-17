from typing import Tuple, Dict

import numpy as np
from scipy.spatial import KDTree
import torch

from ....constants import DTYPE

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
        local_level_set: dictonary of length (n_samples,), each key is the index of a sample and the value is a tensor of shape (n_neighbors,) containing the indices of the neighbors that belong to the same level set
        kernel_vectors: dictonary of length (n_samples,), each key is the index of a sample and the value is a tensor of shape (n_kernel_vectors, n_features) containing a sample from the kernel distribution
    """
    neighbors = _compute_neighborhood(x_values, epsilon_ball)

    # a) Approximate which samples are (i) close to a given sample and (ii) belong to the same level.
    local_level_set = {}
    for idx, y in enumerate(y_values):

        x_neighbors_idxs = np.array(neighbors[idx])
        x_neighbors_idxs = x_neighbors_idxs[x_neighbors_idxs != idx] #remove the point itself

        x_neighbors_level_set_membership = np.array(torch.abs(y_values[x_neighbors_idxs] - y) < epsilon_level_set)
        x_neighbors_level_set_idxs = x_neighbors_idxs[x_neighbors_level_set_membership]

        local_level_set[idx] = x_neighbors_level_set_idxs

    # b) Approximate the exponential map between samples in the same level set and in an \epsilon-ball via a linear approximation.
    kernel_vectors = {}
    for idx_sample, sample in enumerate(x_values):
        kernel_vectors[idx_sample] = torch.zeros((len(local_level_set[idx_sample]), n_features), dtype=DTYPE)

        for idx, neighbor_idx in enumerate(local_level_set[idx_sample]):
            kernel_vectors[idx_sample][idx] = x_values[neighbor_idx] - sample

    return kernel_vectors, local_level_set, neighbors


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
    for idx_sample in kernel_vectors.keys():

        kernel_vectors_point = kernel_vectors[idx_sample]
        if len(kernel_vectors_point) == 0:
            # No non-trivial tangent vectors known in the kernel so we can't compute a basis.
            continue
        elif len(kernel_vectors_point) == 1:
            # Only one non-trivial kernel vector, use this as approximation of the kernel at p.
            basis[idx_sample] = torch.nn.functional.normalize(kernel_vectors_point, p=2, dim=1).flatten()
        else:
            # There exists at least two non-trivial kernel vectors, compute a basis of dimension SYMMETRY_DIM of the Kernel distribution at the sample, normalized to have unit norm.
            _, _, _basis_vector = torch.pca_lowrank(kernel_vectors_point, q=kernel_dim)
            _basis_vector = _basis_vector.T # PCA returns a vector of shape (1, n_features), we want it to be of shape (n_features, 1)
            basis[idx_sample] = torch.nn.functional.normalize(_basis_vector, p=2, dim=1).flatten()

    return basis

def pointwise_kernel_approx(s: torch.tensor,
                            y: torch.tensor,
                            epsilon_ball: float,
                            epsilon_level_set: float):
    """Computes pointwise bases of a kernel of a function.
    
    Args:
    - s: torch.tensor of shape (n_samples, )
    
    """
    
    kernel, local_level_set, neighbors = _compute_kernel_samples(s, y, epsilon_ball, epsilon_level_set)
    return _compute_pointwise_basis(kernel, local_level_set)