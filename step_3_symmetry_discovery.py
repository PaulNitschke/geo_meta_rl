import wandb
import torch

from src.utils import load_replay_buffer_and_kernel, Affine2D
from src.learning.symmetry.hereditary_geometry_discovery import HereditaryGeometryDiscovery

FOLDER_NAME: str="data/local/experiment/circle_rotation"
TASK_NAMES=["sac_circle_rotation_task_0", "sac_circle_rotation_task_1", "sac_circle_rotation_task_2", "sac_circle_rotation_task_3"]

LOAD_WHAT:str="next_observations"
KERNEL_DIM=1
N_SAMPLES=50_000

tasks_ps, tasks_frameestimators=[], []
for task_name in TASK_NAMES:
    ps, frameestimator = load_replay_buffer_and_kernel(task_name, LOAD_WHAT, KERNEL_DIM, N_SAMPLES, FOLDER_NAME)
    tasks_ps.append(ps)
    tasks_frameestimators.append(frameestimator)



ORACLE_GENERATOR=torch.tensor([[0, -1], [1,0]], dtype=torch.float32, requires_grad=False).unsqueeze(0)

train_goal_locations=[
    {'goal': torch.tensor([-0.70506063,  0.70914702])},
 {'goal': torch.tensor([ 0.95243384, -0.30474544])},
 {'goal': torch.tensor([-0.11289421, -0.99360701])},
 {'goal': torch.tensor([-0.81394263, -0.58094525])}]

SEED=42
LEARN_LEFT_ACTIONS=False
LEARN_GENERATOR=False
LEARN_ENCODER_DECODER=True
N_STEPS=10_000
BATCH_SIZE=128
BANDWIDTH=0.5

ENCODER=Affine2D(input_dim=2, output_dim=2)
DECODER=Affine2D(input_dim=2, output_dim=2)

WAND_PROJECT_NAME="circle_hereditary_geometry_discovery"

wandb.init(project=WAND_PROJECT_NAME, config={
    "n_steps": N_STEPS,
    "batch_size": BATCH_SIZE,
    "kernel_dim": KERNEL_DIM,
    "bandwidth": BANDWIDTH,
    "learn_encoder_decoder": LEARN_ENCODER_DECODER,
    "learn_left_actions": LEARN_LEFT_ACTIONS,
    "learn_generator": LEARN_GENERATOR,
    "seed": SEED,
})

her_geo_dis=HereditaryGeometryDiscovery(tasks_ps=tasks_ps,
                                        tasks_frameestimators=tasks_frameestimators, 
                                        kernel_dim=KERNEL_DIM, 
                                        batch_size=BATCH_SIZE, 
                                        seed=SEED, 
                                        bandwidth=BANDWIDTH,
                                        log_wandb=True,
                                        learn_encoder_decoder=LEARN_ENCODER_DECODER,
                                        task_specifications=train_goal_locations,
                                        learn_left_actions=LEARN_LEFT_ACTIONS,
                                        learn_generator=LEARN_GENERATOR,
                                        oracle_generator=ORACLE_GENERATOR,
                                        encoder=ENCODER,
                                        decoder=DECODER)
her_geo_dis.optimize(n_steps=N_STEPS)
wandb.finish()