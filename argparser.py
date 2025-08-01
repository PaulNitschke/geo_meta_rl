import argparse

def get_argparser():
    parser = argparse.ArgumentParser(description="Hereditary Geometry Discovery Experiment Configuration")
    
    # Main parameters.
    parser.add_argument("--kernel_dim", type=int, default=1, help="Dimension of the kernel")
    parser.add_argument("--update_chart_every_n_steps", type=int, default=150, help="Update chart every n steps")
    parser.add_argument("--eval_span_how", type=str, choices=["weights", "ortho_comp"], default="weights", help="How to evaluate span")
    parser.add_argument("--n_steps_lgs", type=int, default=10_000, help="Number of optimization steps for the left action discovery (usually higher).")
    parser.add_argument("--n_steps_gen", type=int, default=10_000, help="Number of optimization steps for learning the generator (usually smaller).")
    parser.add_argument("--n_steps_sym", type=int, default=10_000, help="Number of optimization steps for the symmetry discovery (usually medium).")
    parser.add_argument("--log_lg_inits_how", type=str, choices=["log_linreg", "random"], default="log_linreg", help="Initialization method of the log-left actions.")
    
    # General optimization parameters: Batch size, learning rates, lasso coefficients.
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
    parser.add_argument("--log_wandb_gradients", type=str2bool, default="false", help="Whether to log network gradients to Weights & Biases.")
    parser.add_argument("--save_every", type=int, default=10_000, help="Checkpoint frequency.")

    # Debugging parameters.
    parser.add_argument("--eval_sym_in_follower", type=str2bool, default = "false", help="Whether to include the symmetry loss in the follower loss.")
    parser.add_argument("--learn_generator", type=str2bool, default="true", help="Whether to learn the generator, only used for debugging.")
    parser.add_argument("--use_oracle_rotation_kernel", type=str2bool, default="true", help="Whether to use the hard-coded oracle rotation kernel, only used for debugging.")

    return parser

def get_non_default_args(parser, parsed_args) -> dict:
    """Returns all non-default arguments from the parsed args."""
    defaults = parser.parse_args([])
    return {
    k: v for k, v in vars(parsed_args).items()
    if getattr(parsed_args, k) != getattr(defaults, k)}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')