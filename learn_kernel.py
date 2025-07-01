import pickle
from src.utils import load_replay_buffer

from src.learning.symmetry_discovery.differential.kernel_approx import PointwiseKernelApproximation

TASK_NAME="sac_circle_rotation_task_0"
N_SAMPLES:int=50_000
KERNEL_DIM=1
EPSILON_BALL = 0.05
EPSILON_LEVEL_SET = 0.0005 
LEARN_KERNEL_BASES: bool=True


replay_buffer_name:str=TASK_NAME+"_replay_buffer.pkl"
kernel_bases_name:str=TASK_NAME+"_kernel_bases.pkl"


# Step 1: Load data
replay_buffer_task_1= load_replay_buffer(replay_buffer_name, N_steps=N_SAMPLES)

# ps=replay_buffer_task_1["observations"]
next_ps=replay_buffer_task_1["next_observations"]
ns=replay_buffer_task_1["rewards"]

print("Shape of ps: ", next_ps.shape, " (should be (N_steps, |S|))")
print("Shape of ns: ", ns.shape, " (should be (N_steps))")


# Step 2: Compute or load kernel bases
if LEARN_KERNEL_BASES:
    kernel_approx= PointwiseKernelApproximation(ps=next_ps, ns=ns, kernel_dim=KERNEL_DIM, epsilon_ball=EPSILON_BALL, epsilon_level_set=EPSILON_LEVEL_SET)
    kernel_samples = kernel_approx.compute()
    with open(kernel_bases_name, 'wb') as f:
        pickle.dump(kernel_samples, f)
else:
    with open(kernel_bases_name, 'rb') as f:
        kernel_samples = pickle.load(f)


# Step 3: Plot some kernel samples

import numpy as np

import matplotlib.pyplot as plt


# Randomly select 100 indices from the sampled points
num_to_plot = 35
selected_indices = np.random.choice(list(kernel_samples.keys()), num_to_plot, replace=False)

# Select the corresponding points and vectors
points_to_plot = next_ps[selected_indices]
vectors_to_plot = [kernel_samples[idx] for idx in selected_indices]

vectors_to_plot = np.array([vectors_to_plot[i].flatten().numpy() for i in range(len(vectors_to_plot))])

# Plot the points and vectors using quiver
plt.figure(figsize=(8, 8))
plt.quiver(
    points_to_plot[:, 0], points_to_plot[:, 1], 
    vectors_to_plot[:, 0], vectors_to_plot[:, 1], 
    angles='xy', scale_units='xy', scale=5, color='blue', alpha=0.5, label="Kernel samples"
)

goal = np.array([-0.70506063, 0.70914702])
circle_radii = [0.25, 0.5, 0.75, 1]
for radius in circle_radii:
    circle = plt.Circle(goal, radius, color='red', fill=False, linestyle='--')
    plt.gca().add_artist(circle)

# Set plot limits and labels
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Pointwise kernel approximations')
plt.grid()
plt.legend()
plt.savefig("pointwise_kernel_approximations.png", dpi=300, bbox_inches='tight')