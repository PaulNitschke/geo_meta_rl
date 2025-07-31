import torch

def rotation_vector_field(p_batch: torch.tensor, center)->torch.tensor:
    """Returns kernel samples at batched points p from a task."""
    _generator=torch.tensor([[0, -1], [1,0]], requires_grad=False, dtype=torch.float32).unsqueeze(0)
    projected_state = p_batch-center
    gradients = torch.einsum("dmn, bn->bdm", _generator, projected_state)
    norm_gradients = gradients.norm(dim=-1, keepdim=True)
    return gradients/norm_gradients

#########################################################Oracle generator and charts for debugging.################################################################
# Define oracle charts and generator, only used for debugging.
ORACLE_GENERATOR=torch.tensor([[0,-1], [1,0]], dtype=torch.float32, requires_grad=False).unsqueeze(0) if not args.learn_generator else None
ORACLE_ENCODER_GEO, ORACLE_DECODER_GEO, ORACLE_ENCODER_SYM, ORACLE_DECODER_SYM=DenseNN([2,2]), DenseNN([2,2]), DenseNN([2,2]), DenseNN([2,2])

with torch.no_grad():
    ORACLE_ENCODER_GEO.linear.weight.copy_(torch.eye(2))
    ORACLE_DECODER_GEO.linear.weight.copy_(torch.eye(2))
    ORACLE_ENCODER_SYM.linear.weight.copy_(torch.eye(2))
    ORACLE_DECODER_SYM.linear.weight.copy_(torch.eye(2))
    ORACLE_ENCODER_GEO.linear.bias.copy_(torch.zeros(2))
    ORACLE_DECODER_GEO.linear.bias.copy_(torch.zeros(2))
    ORACLE_ENCODER_SYM.linear.bias.copy_(-train_goal_locations[0]["goal"])
    ORACLE_DECODER_SYM.linear.bias.copy_(train_goal_locations[0]["goal"])

ORACLE_FRAMES=[lambda ps: rotation_vector_field(ps, center=task['goal']) for task in train_goal_locations]
######################################################################################################################################