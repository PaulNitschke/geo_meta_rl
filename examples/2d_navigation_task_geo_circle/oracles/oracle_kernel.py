import torch

def rotation_vector_field(p_batch: torch.tensor, center)->torch.tensor:
    """Returns kernel samples at batched points p from a task."""
    _generator=torch.tensor([[0, -1], [1,0]], requires_grad=False, dtype=torch.float32).unsqueeze(0)
    projected_state = p_batch-center
    gradients = torch.einsum("dmn, bn->bdm", _generator, projected_state)
    norm_gradients = gradients.norm(dim=-1, keepdim=True)
    return gradients/norm_gradients