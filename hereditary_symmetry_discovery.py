from src.utils import load_replay_buffer_and_kernel
import wandb
import torch
import argparse
from src.learning.symmetry.hereditary_geometry_discovery import HereditaryGeometryDiscovery
from src.utils import Affine2D

def train(lr_chart, update_chart_every_n_steps):
    """Trains hereditary symmetry discovery on circle where we change the learning rate for the chart and the update frequency of the chart."""

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




    train_goal_locations=[
        {'goal': torch.tensor([-0.70506063,  0.70914702])},
    {'goal': torch.tensor([ 0.95243384, -0.30474544])},
    {'goal': torch.tensor([-0.11289421, -0.99360701])},
    {'goal': torch.tensor([-0.81394263, -0.58094525])}]

    SEED=42
    LEARN_LEFT_ACTIONS=True
    LEARN_GENERATOR=True
    LEARN_ENCODER_DECODER=True
    USE_ORACLE_ROTATION_KERNEL=True
    N_STEPS=25_000
    BATCH_SIZE=128
    BANDWIDTH=None
    LEARNING_RATE_LEFT_ACTIONS=0.00035
    LEARNING_RATE_GENERATOR=0.00035


    ENCODER=Affine2D(input_dim=2, output_dim=2)
    DECODER=Affine2D(input_dim=2, output_dim=2)
    ORACLE_GENERATOR=torch.tensor([[0, -1], [1,0]], dtype=torch.float32, requires_grad=False).unsqueeze(0) if not LEARN_GENERATOR else None

    WAND_PROJECT_NAME="circle_hereditary_geometry_discovery"

    wandb.init(project=WAND_PROJECT_NAME, name=f"lr_chart:{lr_chart}_update_n:{update_chart_every_n_steps}",config={
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "kernel_dim": KERNEL_DIM,
        "bandwidth": BANDWIDTH,
        "learn_encoder_decoder": LEARN_ENCODER_DECODER,
        "learn_left_actions": LEARN_LEFT_ACTIONS,
        "learn_generator": LEARN_GENERATOR,
        "seed": SEED,
        "use_oracle_rotation_kernel": USE_ORACLE_ROTATION_KERNEL,
        "learning_rate_left_actions": LEARNING_RATE_LEFT_ACTIONS,
        "learning_rate_generator": LEARNING_RATE_GENERATOR,
        "learning_rate_encoder": lr_chart,
        "learning_rate_decoder": lr_chart,
        "update_chart_every_n_steps": update_chart_every_n_steps,
    }, reinit=True)

    her_geo_dis=HereditaryGeometryDiscovery(tasks_ps=tasks_ps,
                                            tasks_frameestimators=tasks_frameestimators, 
                                            kernel_dim=KERNEL_DIM, 
                                            batch_size=BATCH_SIZE, 
                                            seed=SEED, 
                                            bandwidth=BANDWIDTH,
                                            log_wandb=True,
                                            learn_encoder_decoder=LEARN_ENCODER_DECODER,
                                            use_oracle_rotation_kernel=USE_ORACLE_ROTATION_KERNEL,
                                            task_specifications=train_goal_locations,
                                            learn_left_actions=LEARN_LEFT_ACTIONS,
                                            learn_generator=LEARN_GENERATOR,
                                            oracle_generator=ORACLE_GENERATOR,
                                            learning_rate_left_actions=LEARNING_RATE_LEFT_ACTIONS,
                                            learning_rate_generator=LEARNING_RATE_GENERATOR,
                                            learning_rate_encoder=lr_chart,
                                            learning_rate_decoder=lr_chart,
                                            update_chart_every_n_steps=update_chart_every_n_steps,
                                            encoder=ENCODER,
                                            decoder=DECODER)
    her_geo_dis.optimize(n_steps=N_STEPS)
    wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_chart", type=float, default=0.00035, help="Learning rate for the charts.")
    parser.add_argument("--update_chart_every_n_steps", type=int, default=1, help="Update the chart only every n gradient steps.")
    args = parser.parse_args()
    print(f"Training with learning rate for chart: {args.lr_chart} and update frequency: {args.update_chart_every_n_steps}")
    train(args.lr_chart, args.update_chart_every_n_steps)