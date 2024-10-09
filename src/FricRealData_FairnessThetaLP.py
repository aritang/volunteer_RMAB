from bugdet_allocation_solver import BudgetSolver

from utils import parse_arguments
import os
import numpy as np
from instance_generator import initialize_instance_and_simulator, initial_state_generator
from instance_generator import initialize_real_instance
from LDS_RMAB_formulation import SIMULATE_wrt_LDS
from bugdet_allocation_solver import BudgetSolver
from result_recorder import write_result
from visualization import plot_pareto_frontier
from fairness_LP import plot_n_save_results_about_pareto_frontier

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import copy  # To make deep copies of the BudgetSolver instance
import pulp  # Ensure pulp is imported for LP solving

def plot_volunteer_allocation(real_instance_generator, budget_solver, theta, args):
    """
    Plots the allocation of budget to volunteers and regions, incorporating solved results from the budget solver.

    Args:
        real_instance_generator (RealRegionalInstanceGenerator): Instance generator with regions and volunteers.
        budget_solver (BudgetSolver): Solver to allocate budget to volunteers and regions.
        theta (float): Fairness lower bound parameter.
        args: Additional arguments including time and experiment name for saving the plot.
    """
    # Set fairness lower bound and solve
    budget_solver.set_fairness_lowerbound(lowerbound=theta)
    budget_solver.solve()

    if not budget_solver.IF_solved:
        raise ValueError("Budget solver failed to solve the allocation problem.")

    # Retrieve solved variables
    mu_pos, mu_neg, budget_allocation = budget_solver.get_variables()

    # Define colors for each region
    num_regions = len(real_instance_generator.regions)

    colors = plt.colormaps['tab10']

    plt.figure(figsize=(14, 14))

    x_min, x_max = -20, 120
    y_min, y_max = -20, 120
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Compute connection weights for all volunteer-region pairs
    connection_weights = np.zeros((len(real_instance_generator.volunteers), len(real_instance_generator.regions)))
    for i, volunteer in enumerate(real_instance_generator.volunteers):
        for k, region in enumerate(real_instance_generator.regions):
            connection_weights[i, k] = mu_pos[i][k * 2 + 1] / (real_instance_generator.context_prob[k]) if real_instance_generator.context_prob[k] > 0 else 0

    # Determine percentiles for connection weights
    non_zero_weights = connection_weights[connection_weights > 0]
    if len(non_zero_weights) > 0:
        lower_percentile = np.percentile(non_zero_weights, 25)
        upper_percentile = np.percentile(non_zero_weights, 80)
    else:
        lower_percentile = upper_percentile = 0

    # Plot volunteers and connections to regions
    for i, volunteer in enumerate(real_instance_generator.volunteers):
        volunteer_x, volunteer_y = volunteer.x, volunteer.y
        # Determine the color of the volunteer based on the region with highest p[i][k]
        dominant_region = np.argmax(args.p[i])
        volunteer_color = colors(dominant_region)

        # Plot connections between volunteer and regions
        for k, region in enumerate(real_instance_generator.regions):
            connection_weight = connection_weights[i, k]
            if connection_weight > 0:
                if connection_weight >= upper_percentile:
                    alpha = 0.6
                    linewidth = 4
                    linestyle = 'solid'
                elif connection_weight >= lower_percentile:
                    alpha = 0.4
                    linewidth = 2
                    linestyle = 'dashed'
                else:
                    alpha = 0.3
                    linewidth = 1
                    linestyle = 'dotted'
                plt.plot([volunteer_x, region.x], [volunteer_y, region.y], color=colors(k), alpha=alpha, linewidth=linewidth, linestyle=linestyle, zorder=1)

    for i, volunteer in enumerate(real_instance_generator.volunteers):
        volunteer_x, volunteer_y = volunteer.x, volunteer.y
        # Determine the color of the volunteer based on the region with highest p[i][k]
        dominant_region = np.argmin([real_instance_generator.compute_distance(volunteer, region) for region in real_instance_generator.regions])
        volunteer_color = colors(dominant_region)
            # Determine the size of the volunteer based on (sum_k p[i][k]) * q[i]
        
        # Determine the size of the volunteer based on (sum_k p[i][k]) * q[i]
        volunteer_size = volunteer.his * 3.6
        # volunteer_size = (np.sum(args.p[i]) * args.q[i]) * 360
        plt.scatter(volunteer_x, volunteer_y, c=[volunteer_color], marker='o', s=volunteer_size, edgecolor=volunteer_color, linewidth=1, label='Volunteers' if i == 0 else "", zorder=3)

        # Plot regions with budget allocation
    for k, region in enumerate(real_instance_generator.regions):
        region_x, region_y = region.x, region.y
        # take allocation_radius**(2/3)
        # allocation_radius = budget_allocation[k] / real_instance_generator.context_prob[k] if real_instance_generator.context_prob[k] > 0 else 0
        # allocation_radius = allocation_radius**(2/3)
        allocation_radius = budget_allocation[k]
        plt.scatter(region_x, region_y, c=[colors(k)], marker='*', s=360 * region.favourability, alpha=0.8, edgecolor=[colors(k)], linewidth=1.5, label=f'Region {k}', zorder = 3)
        if allocation_radius > 0:
            region_circle = Circle((region_x, region_y), allocation_radius, color=colors(k), alpha=0.3)
            plt.gca().add_patch(region_circle)


    # Set labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Volunteer-Region Allocation and Budget Allocation, theta = {theta}')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Create the results directory if it doesn't exist
    this_path = f'./results/{args.str_time}'
    if not os.path.exists(this_path):
        os.makedirs(this_path)
    # Save the plot
    plot_filename = os.path.join(this_path, f'volunteer_allocation_plot_theta_{theta}_{args.exp_name_out}.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the figure to free memory

    # Reset the fairness lower bound for future operations
    budget_solver.reset_fairness_lowerbound()

if __name__ == '__main__':
    args = parse_arguments()
    args.instance_type = "frictional_real_data_based"
    args.homogeneous = False
    # simulator is redundant here.
    generator, all_transitions, context_prob, reward_vector = initialize_real_instance(args = args)
    args.all_transitions = all_transitions
    args.context_prob = context_prob
    args.reward_vector = reward_vector
    p, q, _ = generator.get_original_vectors(all_transitions=all_transitions, context_prob=context_prob)
    
    args.p = p
    args.q = q

    budget_solver = BudgetSolver(
                N=args.N,
                K=args.K,
                B=args.budget,
                all_transitions=all_transitions,
                context_prob=context_prob,
                w=reward_vector,
    )
    
    theta_UB = plot_n_save_results_about_pareto_frontier(budget_solver=budget_solver, args=args)

    for theta in np.linspace(0, theta_UB, 100):
        plot_volunteer_allocation(generator, budget_solver, theta=theta, args=args)
