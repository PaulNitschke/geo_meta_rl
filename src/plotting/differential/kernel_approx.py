import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import torch

from ....constants import FIG_HEIGHT, DPI

idx_point = 67

plot_lims = 0.085

def plot_point_level_set_tangent(p: torch.tensor,
                                 idx_points: list[int],
                                 local_level_set: dict, 
                                 epsilon_ball: float,
                                 epsilon_level_set: float,
                                 basis,
                                 scale_tangent_vector: float = 0.03,
                                 save_fig: bool = False,
                                 plot_balls: bool = False,
                                 plot_kernel: bool=False,
                                 fig_name: str="kernel_estimation.png"
                                 ):
    
    """
    
    """

    colors = ['g', 'b', 'm', 'c']
    directions = [1, -1]

    # Get point coordinates before plotting to compute width
    p = p[idx_points[0]]
    p_x, p_y = p[0], p[1]
    x_min, x_max = p_x - plot_lims, p_x + plot_lims
    y_min, y_max = p_y - plot_lims, p_y + plot_lims

    # Compute aspect ratio and width
    aspect_ratio = (x_max - x_min) / (y_max - y_min)
    fig_width = FIG_HEIGHT * aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, FIG_HEIGHT))

    for idx, idx_point in enumerate(idx_points):
        direction = idx % 2

        if plot_balls:
            circle = patches.Circle((p_x, p_y), epsilon_ball, fill=True, linestyle='--', linewidth=1, color="blue", alpha=0.2, label="$\epsilon_1$")
            ax.add_patch(circle)

            circle = patches.Circle((p_x, p_y), epsilon_level_set, fill=True, linestyle='--', linewidth=1, color="orange", alpha=0.2, label="$\epsilon_2$")
            ax.add_patch(circle)

        p = s[idx_point]
        p_x = p[0]
        p_y = p[1]

        ax.scatter(p[local_level_set[idx_point]][:, 0], p[local_level_set[idx_point]][:, 1], c=colors[idx], alpha=0.5)
        ax.scatter(p_x, p_y, c=colors[idx], alpha=0.5)

        if plot_kernel:
            _basis_vector = basis[idx_point].numpy().flatten() * scale_tangent_vector * directions[direction]
            ax.quiver(p_x, p_y, _basis_vector[0], _basis_vector[1], angles='xy', scale_units='xy', color=colors[idx], scale=1)

        circle_radius = p[idx_point].norm()
        circle = patches.Circle((0, 0), circle_radius, fill=False, linestyle='--', linewidth=1, edgecolor=colors[idx])
        ax.add_patch(circle)
        ax.set_aspect('equal')



    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    legend_elements = [
        mlines.Line2D([], [], color='black', marker='o', linestyle='None', label=r'$p \in L^{f}_{n}$', alpha=0.5),
    ]
    if plot_kernel:
        legend_elements += [
        mlines.Line2D([0], [0], color='black', marker='>', linestyle='-', markersize=8,
                  label=r'$D^f_p$', linewidth=2, markevery=(1,)),        
        mlines.Line2D([], [], color='black', linestyle='--', label='Level Set')
        ]

    if plot_balls:
        legend_elements += [
            mlines.Line2D([], [], color='blue', marker='o', linestyle='None', label='$\epsilon_1$', alpha=0.5),
            mlines.Line2D([], [], color='orange', marker='o', linestyle='None', label='$\epsilon_2$', alpha=0.5),
            ]

    ax.legend(handles=legend_elements)
    ax.set_xlabel(r'$p_1$')
    ax.set_ylabel(r'$p_2$')
    ax.grid(True)

    if save_fig:
        fig.savefig(fig_name, dpi=DPI, bbox_inches='tight')
