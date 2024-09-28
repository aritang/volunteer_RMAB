from bugdet_allocation_solver import BudgetSolver

from utils import parse_arguments
import os
import numpy as np
from instance_generator import initialize_instance_and_simulator, initial_state_generator
from volunteer_algorithms import whittle_policy_type_specific
from LDS_RMAB_formulation import SIMULATE_wrt_LDS
from bugdet_allocation_solver import BudgetSolver
from result_recorder import write_result
from visualization import plot_pareto_frontier

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


import numpy as np
import matplotlib.pyplot as plt
import copy  # To make deep copies of the BudgetSolver instance
import pulp  # Ensure pulp is imported for LP solving

# Ensure you have imported or defined the BudgetSolver class with the updated methods
# from your previous code snippet.

def find_theta_UB(budget_solver, theta_min=0, tol=1e-5, max_iter=100):
    """
    Perform binary search to find the maximum theta (theta_UB) such that the LP is feasible.

    Args:
        budget_solver (BudgetSolver): An instance of the BudgetSolver class.
        theta_min (float): The minimum value of theta to start the search.
        tol (float): Tolerance for convergence of the binary search.
        max_iter (int): Maximum number of iterations for the binary search.

    Returns:
        theta_UB (float): The maximum theta at which the LP is feasible.
    """
    # Solve the LP without fairness constraints to get the maximum total reward
    budget_solver.solve()
    if budget_solver.IF_solved:
        theta_max = budget_solver.get_reward()
    else:
        print("Error: LP is infeasible without fairness constraints.")
        return None

    # Start binary search
    iteration = 0
    while theta_max - theta_min > tol and iteration < max_iter:
        theta_mid = (theta_min + theta_max) / 2
        # Set fairness constraint with theta_mid
        budget_solver.set_fairness_lowerbound(lowerbound=theta_mid)
        budget_solver.solve()
        if budget_solver.IF_solved:
            # LP is feasible, try increasing theta
            theta_min = theta_mid
        else:
            # LP is infeasible, decrease theta
            theta_max = theta_mid
        # Reset the fairness constraint for next iteration
        budget_solver.reset_fairness_lowerbound()
        iteration += 1
    theta_UB = theta_min
    return theta_UB

def compute_pareto_frontier(budget_solver, theta_UB, N_theta=100):
    """
    For theta in [0, theta_UB], solve the LP under the fairness constraint,
    and collect the (theta, total_reward) pairs.

    Args:
        budget_solver (BudgetSolver): An instance of the BudgetSolver class.
        theta_UB (float): The maximum theta obtained from find_theta_UB function.
        N_theta (int): Number of theta values to sample between 0 and theta_UB.

    Returns:
        theta_list (list): List of theta values.
        total_rewards (list): List of total rewards corresponding to theta_list.
    """
    theta_list = np.linspace(0, theta_UB, N_theta)
    total_rewards = []

    for theta in theta_list:
        # Create a deep copy to avoid modifying the original solver
        budget_solver.set_fairness_lowerbound(lowerbound=theta)
        budget_solver.solve()
        if budget_solver.IF_solved:
            total_reward = budget_solver.get_reward()
            total_rewards.append(total_reward)
        else:
            # If LP is infeasible, record NaN
            total_rewards.append(np.nan)
        budget_solver.reset_fairness_lowerbound()
    return theta_list, total_rewards


if __name__ == '__main__':
    args = parse_arguments()
    # simulator is redundant here.
    simulator, all_transitions, context_prob, reward_vector = initialize_instance_and_simulator(args = args)
    args.all_transitions = all_transitions
    args.context_prob = context_prob
    args.reward_vector = reward_vector
    p, q, _ = simulator.get_original_vectors()
    args.p = p[0]
    args.q = q[0]

    budget_solver = BudgetSolver(
                N=args.N,
                K=args.K,
                B=args.budget,
                all_transitions=all_transitions,
                context_prob=context_prob,
                w=reward_vector,
    )
    # Step 1: Find theta_UB
    theta_UB = find_theta_UB(budget_solver)
    print(f"Theta Upper Bound: {theta_UB}")

    # Step 2: Compute Pareto frontier
    theta_list, total_rewards = compute_pareto_frontier(budget_solver, theta_UB, N_theta=50)

    # Filter out NaN values for plotting
    valid_indices = ~np.isnan(total_rewards)
    theta_list_valid = np.array(theta_list)[valid_indices]
    total_rewards_valid = np.array(total_rewards)[valid_indices]

    colors = {'pareto_frontier': 'green'}
    line_styles = ['--']
    

    args.theta_UB = theta_UB
    args.theta_list = theta_list
    args.reward_list = total_rewards
    write_result(rewards=max(total_rewards), use_algos=["fairness_LP"], args=args, transition_probabilities=all_transitions, context_prob=context_prob, p = p, q = q, rewards_to_write=None, best_allocation=None)

    plot_pareto_frontier(theta_list, total_rewards, args, colors=colors, line_styles=line_styles)
