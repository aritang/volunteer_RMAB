import numpy as np
import pandas as pd
import random
import time
import datetime
import sys
import os
import argparse
import itertools
import matplotlib.pyplot as plt

from volunteer_simulator import Volunteer_RMABSimulator, randomly_generate_transitions
from volunteer_algorithms import (
    whittle_policy,
    random_policy,
    whittle_policy_type_specific,
)
from instance_generator import InstanceGenerator

"""
brute_search_budget_allocation.py

This script performs a brute-force search over possible budget allocations for a Volunteer Restless Multi-Armed Bandit (RMAB) problem. It aims to find the optimal allocation of budget across different contexts to maximize the expected cumulative reward.

The script initializes a simulator for the RMAB problem based on provided or generated data. It then iterates over all feasible budget allocations, evaluating the performance of a specified policy (e.g., Whittle index policy) under each allocation.

Usage:
    python brute_search_budget_allocation.py --n_arms 60 --budget 20 --num_context 3

Arguments:
    --n_arms, -N: Number of arms (beneficiaries)
    --budget, -B: Total budget
    --num_context, -K: Number of contexts
    --episode_len, -H: Length of each episode
    --n_episodes, -T: Number of episodes to run
    --data, -D: Dataset to use ('synthetic', 'real', 'local_generated')
    --n_epochs, -E: Number of epochs (repeats) for averaging results
    --discount, -d: Discount factor for future rewards
    --seed, -s: Random seed for reproducibility
    --verbose, -V: Verbose output if set
    --local, -L: Running locally if set
    --prefix, -p: Prefix for file writing

Functions:
    - parse_arguments(): Parses command-line arguments.
    - initialize_simulator(args): Initializes the RMAB simulator with given parameters.
    - brute_force_search(simulator, n_episodes, n_epochs, discount): Performs brute-force search over budget allocations.
    - brute_force_search_wrt_allowance(...): Similar to brute_force_search but with an allowance threshold.
    - main(): Main function to execute the script.
"""

def parse_arguments():
    """
    Parses command-line arguments provided to the script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_arms', '-N', help='Number of beneficiaries (arms)', type=int, default=60)
    parser.add_argument('--budget', '-B', help='Total budget', type=int, default=20)
    parser.add_argument('--num_context', '-K', help='Number of contexts', type=int, default=3)
    parser.add_argument('--episode_len', '-H', help='Length of each episode', type=int, default=100)
    parser.add_argument('--n_episodes', '-T', help='Number of episodes', type=int, default=6)
    parser.add_argument('--data', '-D', help='Dataset to use {synthetic, real, local_generated}', type=str, default='synthetic')
    parser.add_argument('--n_epochs', '-E', help='Number of epochs (repeats) for averaging results', type=int, default=1)
    parser.add_argument('--discount', '-d', help='Discount factor for future rewards', type=float, default=0.98)
    parser.add_argument('--alpha', '-a', help='Alpha: for confidence radius (not used)', type=float, default=3)
    parser.add_argument('--n_actions', '-A', help='Number of actions', type=int, default=2)
    parser.add_argument('--seed', '-s', help='Random seed', type=int, default=42)
    parser.add_argument('--verbose', '-V', help='Verbose output if set', action='store_true')
    parser.add_argument('--local', '-L', help='Running locally if set', action='store_true')
    parser.add_argument('--prefix', '-p', help='Prefix for file writing', type=str, default='')
    args = parser.parse_args()
    return args

def initialize_simulator(args):
    """
    Initializes the RMAB simulator with the given arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing simulation parameters.

    Returns:
        tuple:
            - simulator (Volunteer_RMABSimulator): The initialized RMAB simulator.
            - context_prob (list): A list of probabilities for each context.
    """
    # Set the random seed for reproducibility
    np.random.seed(args.seed)

    # Initialize transitions and context probabilities based on the dataset
    if args.data == 'synthetic':
        # Generate synthetic transitions and context probabilities
        all_population_size = args.n_arms
        all_transitions, context_prob = randomly_generate_transitions(
            all_population_size, args.num_context
        )
    elif args.data == 'local_generated':
        # Load locally generated instances
        generator = InstanceGenerator(N=args.n_arms, K=args.num_context, seed=66)
        num_instances = 1
        generator.generate_instances(num_instances)
        instances = generator.load_instances(num_instances)
        all_transitions = instances[0]['transitions']
        context_prob = instances[0]['context_prob']
    else:
        # Raise an exception if the dataset is not implemented
        raise Exception(f'Dataset {args.data} not implemented')

    # Set reward vector (assuming unit rewards for all contexts)
    reward_vector = np.ones(args.num_context)

    # Initialize the simulator with the provided parameters
    simulator = Volunteer_RMABSimulator(
        N=args.n_arms,
        K=args.num_context,
        T=args.episode_len,
        context_prob=context_prob,
        all_transitions=all_transitions,
        budget=args.budget,
        reward_vector=reward_vector,
    )
    return simulator, context_prob

def brute_force_search(simulator, n_episodes, n_epochs, discount):
    """
    Performs a brute-force search over all feasible budget allocations to find the allocation that maximizes the expected reward.

    The function iterates over all possible budget allocations across contexts that satisfy the budget constraint, runs the specified policy for each allocation, and records the average rewards.

    Args:
        simulator (Volunteer_RMABSimulator): The RMAB simulator.
        n_episodes (int): Number of episodes to run for each allocation.
        n_epochs (int): Number of epochs (repeats) for averaging results.
        discount (float): Discount factor for future rewards.

    Returns:
        dict: A dictionary where keys are budget allocations (tuples of integers) and values are the corresponding average rewards.
    """
    # Extract context probabilities and budget information
    context_prob = simulator.context_prob
    B = simulator.budget
    K = simulator.K

    # Calculate upper bounds for budgets in each context
    B_UB = [B / prob for prob in context_prob]
    
    results = {}
    
    # Iterate over all possible budget allocations except for the last context
    for budget_vector in itertools.product(*(range(int(b_ub)+1) for b_ub in B_UB[:-1])):
        # Initialize budget vector for all contexts
        budget_vector_np = np.zeros(K, dtype=int)
        budget_vector_np[:-1] = budget_vector

        # Check if the current budget allocation is feasible
        if sum(np.multiply(context_prob[:-1], budget_vector_np[:-1])) <= B:
            # Calculate remaining budget for the last context
            remaining_budget = B - sum(np.multiply(context_prob[:-1], budget_vector_np[:-1]))
            budget_vector_np[-1] = int(remaining_budget / context_prob[-1])

            # Verify the total budget allocation does not exceed the budget
            if sum(np.multiply(context_prob, budget_vector_np)) <= B:
                # Run the policy with the current budget allocation
                rewards = whittle_policy_type_specific(
                    simulator,
                    budget_vector_np,
                    n_episodes=n_episodes,
                    n_epochs=n_epochs,
                    discount=discount,
                )
                average_reward = np.mean(rewards)
                print(f"Budget Allocation: {budget_vector_np}, Average Reward: {average_reward}")
                # Store the results
                results[tuple(budget_vector_np)] = average_reward
    return results

def brute_force_search_wrt_allowance(simulator, n_episodes, n_epochs, discount, benchmark_rewards, eps_allowance):
    """
    Performs a selective brute-force search over budget allocations based on an allowance threshold.

    Only budget allocations whose benchmark rewards are within a certain allowance of the optimal benchmark reward are evaluated. This reduces computation time by skipping allocations that are unlikely to be optimal.

    Args:
        simulator (Volunteer_RMABSimulator): The RMAB simulator.
        n_episodes (int): Number of episodes to run for each allocation.
        n_epochs (int): Number of epochs (repeats) for averaging results.
        discount (float): Discount factor for future rewards.
        benchmark_rewards (dict): Dictionary of benchmark rewards for each budget allocation.
        eps_allowance (float): Allowance threshold as a fraction of the reward range (e.g., 0.1 for 10%).

    Returns:
        dict: A dictionary where keys are budget allocations (tuples of integers) and values are the corresponding average rewards.
    """
    # Find the maximum and minimum benchmark rewards
    best_allocation = max(benchmark_rewards, key=benchmark_rewards.get)
    max_benchmark_reward = benchmark_rewards[best_allocation]
    worst_allocation = min(benchmark_rewards, key=benchmark_rewards.get)
    min_benchmark_reward = benchmark_rewards[worst_allocation]

    # Calculate the reward range and threshold
    benchmark_range = max_benchmark_reward - min_benchmark_reward
    allowance = benchmark_range * eps_allowance
    threshold_of_run = max_benchmark_reward - allowance
    print(f"Benchmark Reward Range: {min_benchmark_reward} to {max_benchmark_reward}, Threshold: {threshold_of_run}")

    # Extract context probabilities and budget information
    context_prob = simulator.context_prob
    B = simulator.budget
    K = simulator.K

    # Calculate upper bounds for budgets in each context
    B_UB = [B / prob for prob in context_prob]
    
    results = {}
    
    # Iterate over all possible budget allocations except for the last context
    for budget_vector in itertools.product(*(range(int(b_ub)+1) for b_ub in B_UB[:-1])):
        # Initialize budget vector for all contexts
        budget_vector_np = np.zeros(K, dtype=int)
        budget_vector_np[:-1] = budget_vector

        # Check if the current budget allocation is feasible
        if sum(np.multiply(context_prob[:-1], budget_vector_np[:-1])) <= B:
            # Calculate remaining budget for the last context
            remaining_budget = B - sum(np.multiply(context_prob[:-1], budget_vector_np[:-1]))
            budget_vector_np[-1] = int(remaining_budget / context_prob[-1])

            # Verify the total budget allocation does not exceed the budget
            if sum(np.multiply(context_prob, budget_vector_np)) <= B:
                # Check if the benchmark reward meets the threshold
                if benchmark_rewards.get(tuple(budget_vector_np), 0) >= threshold_of_run:
                    # Run the policy with the current budget allocation
                    rewards = whittle_policy_type_specific(
                        simulator,
                        budget_vector_np,
                        n_episodes=n_episodes,
                        n_epochs=n_epochs,
                        discount=discount,
                    )
                    average_reward = np.mean(rewards)
                    # print(f"Budget Allocation: {budget_vector_np}, Average Reward: {average_reward}")
                    # Store the results
                    results[tuple(budget_vector_np)] = average_reward
                else:
                    # Skip allocations below the threshold
                    results[tuple(budget_vector_np)] = 0
    return results

def main():
    """
    Main function to execute the script.

    Process:
    - Parses command-line arguments.
    - Initializes the RMAB simulator based on the arguments.
    - Performs a brute-force search over budget allocations.
    - Identifies and prints the best budget allocation and its corresponding average reward.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Initialize the simulator and get context probabilities
    simulator, context_prob = initialize_simulator(args)

    # Perform brute-force search to find the best budget allocation
    results = brute_force_search(
        simulator,
        args.n_episodes,
        args.n_epochs,
        args.discount
    )
    
    # Identify and print the best budget allocation
    print("\nBest Budget Allocation and Average Reward:")
    best_allocation = max(results, key=results.get)
    best_reward = results[best_allocation]
    print(f"Budget Allocation: {best_allocation}, Average Reward: {best_reward}")

if __name__ == '__main__':
    main()


# import numpy as np
# import pandas as pd
# import random
# import time, datetime
# import sys, os
# import argparse
# import itertools

# import matplotlib.pyplot as plt

# from volunteer_simulator import Volunteer_RMABSimulator, randomly_generate_transitions
# # from uc_whittle import UCWhittle
# # from ucw_value import UCWhittle_value
# from volunteer_algorithms import whittle_policy , random_policy, whittle_policy_type_specific #, WIQL
# from instance_generator import InstanceGenerator

# def parse_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--n_arms',         '-N', help='num beneficiaries (arms)', type=int, default=60)
#     parser.add_argument('--budget',         '-B', help='budget', type=int, default=20)
#     parser.add_argument('--num_context',    '-K', help='context_size', type=int, default='3')

#     parser.add_argument('--episode_len',    '-H', help='episode length', type=int, default=100)
#     parser.add_argument('--n_episodes',     '-T', help='num episodes', type=int, default=6)
#     parser.add_argument('--data',           '-D', help='dataset to use {synthetic, real, local_generated}', type=str, default='synthetic')

#     parser.add_argument('--n_epochs',       '-E', help='number of epochs (num_repeats)', type=int, default=1)
#     parser.add_argument('--discount',       '-d', help='discount factor', type=float, default=0.98)

#     # doesn't seem necessary
#     parser.add_argument('--alpha',          '-a', help='alpha: for conf radius', type=float, default=3)

#     # parser.add_argument('--n_states',       '-S', help='num states', type=int, default=2)
#     parser.add_argument('--n_actions',      '-A', help='num actions', type=int, default=2)
#     parser.add_argument('--seed',           '-s', help='random seed', type=int, default=42)
#     parser.add_argument('--verbose',        '-V', help='if True, then verbose output (default False)', action='store_true')
#     parser.add_argument('--local',          '-L', help='if True, running locally (default False)', action='store_true')
#     parser.add_argument('--prefix',         '-p', help='prefix for file writing', type=str, default='')


#     args = parser.parse_args()
#     return args


# def initialize_simulator(args):
#     np.random.seed(args.seed)
#     if args.data == 'synthetic':
#         all_population_size = args.n_arms
#         all_transitions, context_prob = randomly_generate_transitions(all_population_size, args.num_context)

#     elif args.data == 'local_generated':
#         generator = InstanceGenerator(N=args.n_arms, K=args.num_context, seed=66)
#         num_instances = 1
#         generator.generate_instances(num_instances)
#         instances = generator.load_instances(num_instances)
#         all_transitions, context_prob = instances[0]['transitions'], instances[0]['context_prob']
#     else:
#         raise Exception(f'dataset {args.data} not implemented')

#     reward_vector = np.ones(args.num_context)
#     simulator = Volunteer_RMABSimulator(N=args.n_arms, K=args.num_context, T=args.episode_len, context_prob=context_prob, all_transitions=all_transitions, budget=args.budget, reward_vector=reward_vector)
#     return simulator, context_prob

# def brute_force_search(simulator, n_episodes, n_epochs, discount):
#     context_prob = simulator.context_prob
#     B = simulator.budget
#     K = simulator.K
#     B_UB = [B / prob for prob in context_prob]
    
#     results = {}
    
#     for budget_vector in itertools.product(*(range(int(b_ub)+1) for b_ub in B_UB[:-1])):
        
#         budget_vector_np = np.zeros(K, dtype=int)
#         budget_vector_np[:-1] = budget_vector
#         if sum(np.multiply(context_prob[:-1], budget_vector_np[:-1])) <= B:
#             remaining_budget = B - sum(np.multiply(context_prob[:-1], budget_vector_np[:-1]))
#             budget_vector_np[-1] = int(remaining_budget / context_prob[-1])
#             if sum(np.multiply(context_prob, budget_vector_np)) <= B:
#                 # run is here
#                 rewards = whittle_policy_type_specific(simulator, budget_vector_np, n_episodes=n_episodes, n_epochs=n_epochs, discount=discount)
#                 print(f"B = {budget_vector_np}, reward = {np.mean(rewards)}")
#                 results[tuple(budget_vector_np)] = np.mean(rewards)
#         # print(f"\nreward = {np.mean(rewards)}\nbudget = {budget_vector_np}")
#     return results

# def brute_force_search_wrt_allowance(simulator, n_episodes, n_epochs, discount, benchmark_rewards, eps_allowance):
#     """
#     similar to brute_force_search(...) -> return dict results
#     only difference is that only benchmark_rewards[budget] - optimal_rewards_in_benchmark < eps_allowance * range would be runned. hence saving time
#     """
#     best_allocation = max(benchmark_rewards, key=benchmark_rewards.get)
#     max_benchmark_reward = benchmark_rewards[best_allocation]
#     worst_allocation = min(benchmark_rewards, key=benchmark_rewards.get)
#     min_benchmark_rewards = benchmark_rewards[worst_allocation]
#     benchmark_range = max_benchmark_reward - min_benchmark_rewards
#     allowance = benchmark_range*eps_allowance
#     threshold_of_run = max_benchmark_reward - allowance
#     print(f"min_benchmark_rewards, max_benchmark_reward, threshold_of_run = {min_benchmark_rewards, max_benchmark_reward, threshold_of_run}")

#     context_prob = simulator.context_prob
#     B = simulator.budget
#     K = simulator.K
#     B_UB = [B / prob for prob in context_prob]
    
#     results = {}
    
#     for budget_vector in itertools.product(*(range(int(b_ub)+1) for b_ub in B_UB[:-1])):
        
#         budget_vector_np = np.zeros(K, dtype=int)
#         budget_vector_np[:-1] = budget_vector
#         if sum(np.multiply(context_prob[:-1], budget_vector_np[:-1])) <= B:
#             remaining_budget = B - sum(np.multiply(context_prob[:-1], budget_vector_np[:-1]))
#             budget_vector_np[-1] = int(remaining_budget / context_prob[-1])
#             if sum(np.multiply(context_prob, budget_vector_np)) <= B:
#                 # run is here
#                 if benchmark_rewards[tuple(budget_vector_np)] >= threshold_of_run:
#                     rewards = whittle_policy_type_specific(simulator, budget_vector_np, n_episodes=n_episodes, n_epochs=n_epochs, discount=discount)
#                     # print(f"B = {budget_vector_np}, reward = {np.mean(rewards)}")
#                     results[tuple(budget_vector_np)] = np.mean(rewards)
#                 else:
#                     results[tuple(budget_vector_np)] = 0
#         # print(f"\nreward = {np.mean(rewards)}\nbudget = {budget_vector_np}")
#     return results


# def main():
#     args = parse_arguments()
#     simulator, context_prob = initialize_simulator(args)
#     results = brute_force_search(simulator, args.n_episodes, args.n_epochs, args.discount)
    
#     # Process and save results
#     print("Best budget allocation and rewards:")
#     best_allocation = max(results, key=results.get)
#     print(f"Budget Allocation: {best_allocation}, Reward: {results[best_allocation]}")


# if __name__ == '__main__':
#     main()