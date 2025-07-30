import os
import wandb
import torch
from datetime import datetime

from src.learning.symmetry.hereditary_geometry_discovery import HereditaryGeometryDiscovery
from argparser import get_argparser, get_non_default_args
from src.utils import load_replay_buffer_and_kernel, Affine2D

FOLDER_NAME: str="data/local/experiment/circle_rotation"
TASK_NAMES=["sac_circle_rotation_task_0", "sac_circle_rotation_task_1", "sac_circle_rotation_task_2", "sac_circle_rotation_task_3"]

train_goal_locations=[
    {'goal': torch.tensor([-0.70506063,  0.70914702])},
    {'goal': torch.tensor([ 0.95243384, -0.30474544])},
    {'goal': torch.tensor([-0.11289421, -0.99360701])},
    {'goal': torch.tensor([-0.81394263, -0.58094525])}]

LOAD_WHAT:str="next_observations"
N_SAMPLES=50_000


# Define oracle charts and generator, only used for debugging.
ENCODER_GEO=Affine2D(input_dim=2, output_dim=2)
ENCODER_SYM=Affine2D(input_dim=2, output_dim=2)
ORACLE_ENCODER_GEO=Affine2D(input_dim=2, output_dim=2)
ORACLE_DECODER_GEO=Affine2D(input_dim=2, output_dim=2)
ORACLE_ENCODER_SYM=Affine2D(input_dim=2, output_dim=2)
ORACLE_DECODER_SYM=Affine2D(input_dim=2, output_dim=2)

ORACLE_GENERATOR=torch.tensor([[0,-1], [1,0]], dtype=torch.float32, requires_grad=False).unsqueeze(0)

with torch.no_grad():
    ORACLE_ENCODER_GEO.linear.weight.copy_(torch.eye(2))
    ORACLE_DECODER_GEO.linear.weight.copy_(torch.eye(2))
    ORACLE_ENCODER_SYM.linear.weight.copy_(torch.eye(2))
    ORACLE_DECODER_SYM.linear.weight.copy_(torch.eye(2))
    ORACLE_ENCODER_GEO.linear.bias.copy_(torch.zeros(2))
    ORACLE_DECODER_GEO.linear.bias.copy_(torch.zeros(2))
    ORACLE_ENCODER_SYM.linear.bias.copy_(-train_goal_locations[0]["goal"])
    ORACLE_DECODER_SYM.linear.bias.copy_(train_goal_locations[0]["goal"])

def train(parser):
    """Trains hereditary symmetry discovery on circle where we change the learning rate for the chart and the update frequency of the chart."""

    args = parser.parse_args()

    # 1. Load replay buffers and frame estimators.
    tasks_ps, tasks_frameestimators=[], []
    for task_name in TASK_NAMES:
        ps, frameestimator = load_replay_buffer_and_kernel(task_name, LOAD_WHAT, args.kernel_dim, N_SAMPLES, FOLDER_NAME)
        tasks_ps.append(ps)
        tasks_frameestimators.append(frameestimator)

    oracle_generator=ORACLE_GENERATOR if not args.learn_generator else None

    # 2. Setup wandb.
    non_default_args= get_non_default_args(parser, args)
    _run_name = '_'.join(f"{k}:{v}" for k, v in non_default_args.items()) if non_default_args else "default"
    run_name = _run_name + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir=f"data/local/experiment/circle_rotation/{run_name}"
    os.mkdir(save_dir)
    os.mkdir(save_dir + "/pretrain")

    if args.log_wandb:
        WAND_PROJECT_NAME="circle_hereditary_geometry_discovery"
        wandb.init(project=WAND_PROJECT_NAME, name=run_name,config=vars(args))


    # 3. Train.
    her_geo_dis=HereditaryGeometryDiscovery(tasks_ps=tasks_ps,
                                            tasks_frameestimators=tasks_frameestimators, 
                                            oracle_generator=oracle_generator, 
                                            encoder_geo=ENCODER_GEO, 
                                            encoder_sym=ENCODER_SYM,

                                            kernel_dim=args.kernel_dim,
                                            update_chart_every_n_steps=args.update_chart_every_n_steps, 
                                            eval_span_how=args.eval_span_how,
                                            log_lg_inits_how=args.log_lg_inits_how,

                                            batch_size=args.batch_size, 
                                            lr_lgs=args.lr_lgs,
                                            lr_gen=args.lr_gen,
                                            lr_chart=args.lr_chart,
                                            lasso_coef_lgs=args.lasso_coef_lgs, 
                                            lasso_coef_generator=args.lasso_coef_generator, 
                                            lasso_coef_encoder_decoder=args.lasso_coef_encoder_decoder,
                                            n_epochs_pretrain_log_lgs= args.n_epochs_pretrain_log_lgs, 
                                            n_epochs_init_neural_nets= args.n_epochs_init_neural_nets,

                                            seed=args.seed, 
                                            log_wandb=args.log_wandb, 
                                            log_wandb_gradients=args.log_wandb_gradients, 
                                            save_every=args.save_every,
                                            bandwidth=args.bandwidth,

                                            task_specifications=train_goal_locations, 
                                            use_oracle_rotation_kernel=args.use_oracle_rotation_kernel,
                                            save_dir=save_dir,

                                            eval_sym_in_follower=args.eval_sym_in_follower,
                                            oracle_encoder_geo=ORACLE_ENCODER_GEO, 
                                            oracle_decoder_geo=ORACLE_DECODER_GEO,
                                            oracle_encoder_sym=ORACLE_ENCODER_SYM, 
                                            oracle_decoder_sym=ORACLE_DECODER_SYM
                                            )
    
    her_geo_dis.optimize(n_steps_geo=args.n_steps_geo, n_steps_sym=args.n_steps_sym)
    her_geo_dis.save(f"{save_dir}/hereditary_geometry_discovery.pt")
    wandb.finish()


if __name__ == "__main__":
    parser = get_argparser()
    train(parser)