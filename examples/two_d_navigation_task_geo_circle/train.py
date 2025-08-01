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
import os
import logging
import numpy as np
import pdb

from garage.envs.point_env import PointEnv
from garage.experiment.deterministic import set_seed

from examples.two_d_navigation_task_geo_circle import argparser
from examples.two_d_navigation_task_geo_circle import oracles
from src.learning.aa_policy_training.train_policies import train_and_save_pis_and_buffers
from src.learning.bb_symmetry.utils import compute_and_save_kernel_bases
from src.learning.cc_hereditary_geometry.utils import learn_hereditary_symmetry

#########################################################Main variables to set, set the rest in argparser.############################
SAVE_DIR_BASE="data/local/experiment/2d_navigation/rotation_task_geometry"
CHART_TASK_SPACE_GEO=np.eye(2)
######################################################################################################################################

#########################################################Create tasks.################################################################
logging.info("Creating tasks for 2D navigation task with rotation around the origin symmetry.")
parser=argparser.get_argparser()
args=parser.parse_args()

set_seed(args.seed)
CircleRotation=PointEnv()
train_goal_locations=CircleRotation.sample_tasks(args.n_tasks, chart=CHART_TASK_SPACE_GEO)
tasks=[PointEnv().set_task(goal_location) for goal_location in train_goal_locations]
del CircleRotation
######################################################################################################################################


#########################################################Oracle generator and charts for debugging.################################################################
ORACLES = oracles.make_2d_navigation_oracles(train_goal_locations, args.learn_generator)
del train_goal_locations
######################################################################################################################################


######################################################################################################################################
if args.train_policies:
    logging.info("Training policies and saving replay buffers for each task...")
    task_dirs=train_and_save_pis_and_buffers(tasks=tasks,
                                            save_dir=SAVE_DIR_BASE,
                                            args=args)
    logging.info("Finished training policies and saving replay buffers for each task.")
else:
    logging.info("Skipping policy training, using pre-trained policies and replay buffers.")
    task_dirs=[f"{SAVE_DIR_BASE}/task_{i}" for i in range(args.n_tasks)]
    if not all(os.path.exists(dir) for dir in task_dirs):
        raise FileNotFoundError(f"Some task directories do not exist: {task_dirs}")

if args.compute_kernel:
    logging.info("Computing saving pointwise kernel bases for each task...")
    for dir in task_dirs:
        compute_and_save_kernel_bases(dir=dir,
                                    args=args)
    logging.info("Finished saving pointwise kernel bases for each task.")
else:
    logging.info("Skipping kernel computation, using pre-computed kernel bases.")
    #TODO, here we need to load the pre-computed kernel bases.
    # This is not implemented yet, but we assume that the kernel bases are already computed and saved in the task directories.
    # For now, we just log this information.
    logging.info("Assuming pre-computed kernel bases are available in the task directories.")

#TODO, here we need to compute the frame estimator via a neural network.

logging.info("Starting hereditary geometry discovery...")
learn_hereditary_symmetry(dirs=task_dirs,
                          parser=parser,
                          oracles=ORACLES)
logging.info("Finished hereditary geometry discovery.")