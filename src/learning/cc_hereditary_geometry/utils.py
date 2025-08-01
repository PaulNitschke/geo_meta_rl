import os
from datetime import datetime
import warnings

import wandb

from .hereditary_geometry_discovery import HereditaryGeometryDiscovery
from ..aa_policy_training.utils import load_replay_buffer
from ...utils import get_non_default_args

N_STEPS=50_000 #TODO, this must be inferred dynamically.
warnings.warn("using hard coded number of steps.")

def learn_hereditary_symmetry(dirs,
          parser,
          oracles: dict):
    """
    Helper function to learn hereditary symmetry, loads data, sets up wandb, and trains the model.
    Args:
    -dirs: List of directories containing the replay buffers and frame estimators for each task.
    -oracles: dict, containing the keys 'generator', 'encoder_geo', 'decoder_geo', 'encoder_sym', 'decoder_sym', and 'frames'.
                Either oracle value or None.
    """
    args = parser.parse_args()

    # 1. Load replay buffers and frame estimators.
    tasks_ps, tasks_frameestimators=[], []
    for dir in dirs:
        file_name_replay_buffer= f"{dir}/replay_buffer.pkl"
        tasks_ps.append(load_replay_buffer(file_name_replay_buffer, N_steps=N_STEPS))
        tasks_frameestimators.append(None) #TODO, insert proper frame estimator.


    # 2. Setup wandb.
    non_default_args= get_non_default_args(parser, args)
    _run_name = '_'.join(f"{k}:{v}" for k, v in non_default_args.items()) if non_default_args else "default"
    run_name = _run_name + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir=os.path.join(os.path.dirname(dirs[0]), "wandb",run_name)
    os.makedirs(save_dir)
    os.makedirs(save_dir + "/pretrain")

    if args.log_wandb:
        wandb.init(project=args.wandb_project_name,name=run_name,config=vars(args))


    # 3. Train.
    her_geo_dis=HereditaryGeometryDiscovery(tasks_ps=tasks_ps,
                                            tasks_frameestimators=tasks_frameestimators, 
                                            enc_geo_net_sizes=args.enc_geo_net_sizes, 
                                            enc_sym_net_sizes=args.enc_sym_net_sizes,

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

                                            use_oracle_rotation_kernel=args.use_oracle_rotation_kernel,
                                            save_dir=save_dir,

                                            use_oracle_frames=args.use_oracle_frames,
                                            oracle_generator=oracles.oracle_generator, 
                                            oracle_frames=oracles.oracle_frames,
                                            oracle_encoder_geo=oracles.oracle_encoder_geo, 
                                            oracle_decoder_geo=oracles.oracle_decoder_geo,
                                            oracle_encoder_sym=oracles.oracle_encoder_sym, 
                                            oracle_decoder_sym=oracles.oracle_decoder_sym
                                            )
    
    her_geo_dis.optimize(n_steps_lgs=args.n_steps_lgs, n_steps_gen=args.n_steps_gen, n_steps_sym=args.n_steps_sym)
    her_geo_dis.save(f"{save_dir}/hereditary_geometry_discovery.pt")
    wandb.finish()