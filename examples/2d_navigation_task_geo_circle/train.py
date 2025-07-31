"""
2-D Navigation Task, the agent navigates to a base goal location where the reward function is given by the L_2 distance
between the current location and the goal location.
Task space geometry: Rotate base goal position around origin.
Base task symmetry: Rotation around goal location.

Hereditary Symmetry (to be learned):
- Group: Rotation group SO(2).
- Geometry chart: Identity.
- Symmetry chart: Left translation by base goal location.

Number of training tasks: 4.
"""
import logging
import numpy as np

from garage.envs.point_env import PointEnv
from garage.experiment.deterministic import set_seed

from ...src.learning.aa_policy_training.train_policies import train_and_save_pis_and_buffers
from ...src.learning.bb_symmetry.utils import compute_and_save_kernel_bases
from zz_argparser import get_argparser

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
