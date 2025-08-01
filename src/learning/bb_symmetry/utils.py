from src.learning.aa_policy_training.utils import load_replay_buffer
from src.learning.bb_symmetry.kernel_approx import KernelFrameEstimator


def compute_and_save_kernel_bases(dir: str,
                                  args):
    """
    Learn pointwise kernel bases from a replay buffer.
    """
    replay_buffer_name:str=dir+"/replay_buffer.pkl"
    n_steps=int(args.n_steps_train_pis/args.n_envs)
    replay_buffer = load_replay_buffer(replay_buffer_name, N_steps=n_steps)
    assert args.kernel_in in replay_buffer.keys(), f"Kernel input {args.kernel_in} not found in replay buffer keys: {replay_buffer.keys()}"
    assert args.kernel_out in replay_buffer.keys(), f"Kernel output {args.kernel_out} not found in replay buffer keys: {replay_buffer.keys()}"
    
    ps = replay_buffer[args.kernel_in]
    ns = replay_buffer[args.kernel_out]

    frame_estimator = KernelFrameEstimator(ps=ps, kernel_dim=args.kernel_dim, ns=ns, epsilon_ball=args.epsilon_ball, epsilon_level_set=args.epsilon_level_set)
    frame_estimator.compute()

    kernel_bases_name = f"{dir}/kernel_bases.pkl"
    frame_estimator.save(kernel_bases_name)
    
    return frame_estimator


# def load_replay_buffer_and_kernel(task_name:str, load_what:str, kernel_dim: int, n_samples:int, folder_name):
#     """Loads samples and kernel evaluator of a task."""

#     assert load_what in ["observations", "actions", "next_observations"], "Learn hereditary geometry for states, actions or next states."

#     buffer_name= os.path.join(folder_name, f"{task_name}_replay_buffer.pkl")
#     kernel_name= os.path.join(folder_name, f"{task_name}_kernel_bases.pkl")


#     buffer= load_replay_buffer(buffer_name, N_steps=n_samples)
#     ps=buffer[load_what]
#     print(f"Loaded {load_what} from {buffer_name} with shape {ps.shape}")

#     # Load kernel bases
#     frameestimator=KernelFrameEstimator(ps=ps, kernel_dim=kernel_dim)
#     with open(kernel_name, 'rb') as f:
#         kernel_samples = pickle.load(f)
#     frameestimator.set_frame(frame=kernel_samples)

#     return ps, frameestimator