"""
2-D Navigation Task, the agent navigates to a base goal location where the reward function is given by the L_2 distance
between the current location and the goal location.
Task space geometry: Rotate base goal position around origin.
Base task symmetry: Rotation around goal location.

Oracle hereditary Symmetry (to be learned):
- Group: Rotation group SO(2).
- Geometry chart: Identity.
- Symmetry chart: Left translation by base goal location.

Number of training tasks: 4.
"""
import logging
import numpy as np
import torch

from garage.envs.point_env import PointEnv
from garage.experiment.deterministic import set_seed

from argparser import get_argparser
from oracles.oracle_kernel import rotation_vector_field
from ...src.utils import DenseNN
from ...src.learning.aa_policy_training.train_policies import train_and_save_pis_and_buffers
from ...src.learning.bb_symmetry.utils import compute_and_save_kernel_bases

#########################################################Main variables to set, set the rest in argparser.############################
SAVE_DIR_BASE="data/local/experiment/2d_navigation/rotation_task_geometry"
CHART_TASK_SPACE_GEO=np.eye(2)
######################################################################################################################################

#########################################################Create tasks.################################################################
logging.info("Creating tasks for 2D navigation task with rotation around the origin symmetry.")
parser=get_argparser()
set_seed(parser.seed)
CircleRotation=PointEnv()
train_goal_locations=CircleRotation.sample_tasks(parser.n_tasks, CHART_TASK_SPACE_GEO)
tasks=[PointEnv().set_task(goal_location) for goal_location in train_goal_locations]
del CircleRotation, train_goal_locations
######################################################################################################################################


#########################################################Oracle generator and charts for debugging.################################################################
# Define oracle charts and generator, only used for debugging.
ORACLE_GENERATOR=torch.tensor([[0,-1], [1,0]], dtype=torch.float32, requires_grad=False).unsqueeze(0)
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


######################################################################################################################################
logging.info("Training policies and saving replay buffers for each task...")
task_dirs=train_and_save_pis_and_buffers(tasks=tasks,
                                         save_dir=SAVE_DIR_BASE,
                                         argparser=parser)
logging.info("Finished training policies and saving replay buffers for each task...")

logging.info("Computing saving pointwise kernel bases for each task...")
for dir in task_dirs:
    compute_and_save_kernel_bases(dir=dir,
                                  argparser=parser)
logging.info("Finished saving pointwise kernel bases for each task...")

#TODO, here we need to compute the frame estimator via a neural network.

logging.info("Starting hereditary geometry discovery...")
