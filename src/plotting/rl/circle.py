from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

def plot_rollouts_on_circle(train_trajectories: List[np.ndarray],
                             test_trajectories: List[np.ndarray],
                             train_goal_locations: List[np.ndarray],
                             test_goal_locations: List[np.ndarray],
                             fig_height: int,
                             label_font_size: int,
                             dpi: int,
                             idx_label_test=3,
                             legend: bool=False,
                             savepath: str=None,
                             radius: int=2) -> None:
    """
    Plots trajectories from train and test tasks on the circle navigation task.
    Args:
        train_trajectories (List[np.ndarray]): List of trajectories from training tasks.
        test_trajectories (List[np.ndarray]): List of trajectories from testing tasks.
        train_goal_locations (List[np.ndarray]): Goal locations for training tasks.
        test_goal_locations (List[np.ndarray]): Goal locations for testing tasks.
        idx_label_test (int): Index of the trajectory to label as 'Test'.
        legend (bool): Whether to show the legend.
        savepath (str): Path to save the figure, if None, the figure is not saved.
        radius (int): Radius of the circle on which the trajectories are plotted.
    """

    # 1. Set fixed figure size (square)
    fig, ax = plt.subplots(figsize=(fig_height, fig_height))

    use_label_train: bool = True
    label=None
    cmap = plt.get_cmap('hsv')

    num_trajs = len(test_trajectories)
    
    # Train distribution
    circle = plt.Circle((0, 0), radius, color='black', fill=False, linestyle='--', linewidth=2, label=r'$p(\mathcal{T})$', zorder=3, alpha=0.3)
    ax.add_artist(circle)

    for idx_traj, test_traj in enumerate(test_trajectories):
        color = cmap(0.5 + 0.5 * idx_traj / max(1, num_trajs - 1))
        if idx_traj == idx_label_test:
            label = "Test"
            ax.plot(test_traj[:-1,0], test_traj[:-1,1], marker="x", color=color, alpha=1, label=label)
        else:
            ax.plot(test_traj[:-1,0], test_traj[:-1,1], marker="x", color=color, alpha=1, label=None)
        ax.scatter(test_goal_locations[idx_traj][0], test_goal_locations[idx_traj][1], color=color, marker="X", s=500)

    for idx_traj, train_traj in enumerate(train_trajectories):
        if use_label_train:
            label="Train"
            use_label_train=False
            ax.plot(train_traj[:-1,0], train_traj[:-1,1], marker="x", color="black", alpha=1, label=label)
        else:
            ax.plot(train_traj[:-1,0], train_traj[:-1,1], marker="x", color="black", alpha=1, label=None)
        ax.scatter(train_goal_locations[idx_traj][0], train_goal_locations[idx_traj][1], color="black", marker="X", s=500)


    ax.set_xlim(-radius-0.5, radius+0.5)
    ax.set_ylim(-radius-0.5, radius+0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    if legend:
        ax.legend(ncols=4, loc='upper center', bbox_to_anchor=(0.5, 0.05), fontsize=label_font_size)

    # 2. Save figure
    if savepath:
        fig.savefig(savepath, bbox_inches='tight', pad_inches=0.05, dpi=dpi)

    plt.show()
    plt.close(fig)