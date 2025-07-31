import argparse

def get_argparser():
    parser = argparse.ArgumentParser(description="Hereditary Geometry Discovery Experiment Configuration")

    # Parameters to create task distribution and train policies.
    parser.add_argument("--n_steps_train_pis", type=int, default=100_000, help="Number of steps to train each policy for.")
    parser.add_argument("--n_tasks", type=int, default=4, help="Number of tasks to generate.")
    parser.add_argument("--n_envs", type=int, default=2, help="Number of parallel environments to use when training policies.")

    # Parameters for learning symmetry.
    parser.add_argument("--kernel_in", type=str, help="Input for the kernel learning, e.g. p.")
    parser.add_argument("--kernel_out", type=int, default=128, help="Output for the kernel learning, e.g. n.")
    parser.add_argument("--epsilon_ball", type=float, default=0.005, help="Tolerance under which f(p') \approx f(p).")
    parser.add_argument("--epsilon_level_set", type=float, default=0.005, help="Ball in which we Taylor approximate f(p).")
    parser.add_argument("--kernel_dim", type=int, default=1, help="Dimension of the kernel")

    # Main parameters.
    parser.add_argument("--update_chart_every_n_steps", type=int, default=150, help="Update chart every n steps")
    parser.add_argument("--eval_span_how", type=str, choices=["weights", "ortho_comp"], default="weights", help="How to evaluate span")
    parser.add_argument("--n_steps_lgs", type=int, default=10_000, help="Number of optimization steps for the left action discovery (usually higher).")
    parser.add_argument("--n_steps_gen", type=int, default=10_000, help="Number of optimization steps for learning the generator (usually smaller).")
    parser.add_argument("--n_steps_sym", type=int, default=10_000, help="Number of optimization steps for the symmetry discovery (usually medium).")
    parser.add_argument("--log_lg_inits_how", type=str, choices=["log_linreg", "random"], default="log_linreg", help="Initialization method of the log-left actions.")
    
    # General optimization parameters: Batch size, learning rates, lasso coefficients.
    parser.add_argument("--enc_geo_net_sizes", type=list, default=[2,2], help="Network architecture for the encoder of the geometry chart. Same architecture for the decoder.")
    parser.add_argument("--enc_sym_net_sizes", type=list, default=[2,2], help="Network architecture for the encoder of the symmetry chart. Same architecture for the decoder.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr_lgs", type=float, default=0.00035, help="Learning rate for the left actions.")
    parser.add_argument("--lr_gen", type=float, default=0.00035, help="Learning rate for the generator loss.")
    parser.add_argument("--lr_chart", type=float, default=0.00035, help="Learning rate for the chart.")
    parser.add_argument("--lasso_coef_lgs", type=float, default=0.5, help="Lasso coefficient for the log-left actions.")
    parser.add_argument("--lasso_coef_generator", type=float, default=0.005, help="Lasso coefficient for the generator.")
    parser.add_argument("--lasso_coef_encoder_decoder", type=float, default=0.005, help="Lasso coefficient for the encoder and decoder.")
    parser.add_argument("--n_epochs_pretrain_log_lgs", type=int, default=2_500, help="Number of epochs to initialize the log left actions for.")
    parser.add_argument("--n_epochs_init_neural_nets", type=int, default=10_000, help="Number of epochs to initialize encoder and decoder to identity.")

    # Util parameters.
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--bandwidth", type=float, default=None, help="Bandwidth for the kernel") #TODO, need to take this out.
    parser.add_argument("--log_wandb", type=str2bool, default="true", help="Whether to log results to Weights & Biases.")
    parser.add_argument("--wandb_project_name", type=str, default="circle_hereditary_geometry_discovery", help="Project name for wandb.")
    parser.add_argument("--log_wandb_gradients", type=str2bool, default="false", help="Whether to log network gradients to Weights & Biases.")
    parser.add_argument("--save_every", type=int, default=10_000, help="Checkpoint frequency.")

    # Debugging parameters.
    parser.add_argument("--learn_generator", type=str2bool, default="true", help="Whether to learn the generator, only used for debugging.")
    parser.add_argument("--use_oracle_frame", type=str2bool, default="true", help="Whether to use the hard-coded oracle rotation kernel, only used for debugging.")

    return parser

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')