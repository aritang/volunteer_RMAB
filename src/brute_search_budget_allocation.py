import numpy as np
import pandas as pd
import random
import time, datetime
import sys, os
import argparse
import itertools

import matplotlib.pyplot as plt

from volunteer_simulator import Volunteer_RMABSimulator, randomly_generate_transitions
# from uc_whittle import UCWhittle
# from ucw_value import UCWhittle_value
from volunteer_algorithms import whittle_policy , random_policy, whittle_policy_type_specific #, WIQL
from instance_generator import InstanceGenerator

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_arms',         '-N', help='num beneficiaries (arms)', type=int, default=60)
    parser.add_argument('--budget',         '-B', help='budget', type=int, default=20)
    parser.add_argument('--num_context',    '-K', help='context_size', type=int, default='3')

    parser.add_argument('--episode_len',    '-H', help='episode length', type=int, default=100)
    parser.add_argument('--n_episodes',     '-T', help='num episodes', type=int, default=6)
    parser.add_argument('--data',           '-D', help='dataset to use {synthetic, real, local_generated}', type=str, default='synthetic')

    parser.add_argument('--n_epochs',       '-E', help='number of epochs (num_repeats)', type=int, default=1)
    parser.add_argument('--discount',       '-d', help='discount factor', type=float, default=0.98)

    # doesn't seem necessary
    parser.add_argument('--alpha',          '-a', help='alpha: for conf radius', type=float, default=3)

    # parser.add_argument('--n_states',       '-S', help='num states', type=int, default=2)
    parser.add_argument('--n_actions',      '-A', help='num actions', type=int, default=2)
    parser.add_argument('--seed',           '-s', help='random seed', type=int, default=42)
    parser.add_argument('--verbose',        '-V', help='if True, then verbose output (default False)', action='store_true')
    parser.add_argument('--local',          '-L', help='if True, running locally (default False)', action='store_true')
    parser.add_argument('--prefix',         '-p', help='prefix for file writing', type=str, default='')


    args = parser.parse_args()
    return args


def initialize_simulator(args):
    np.random.seed(args.seed)
    if args.data == 'synthetic':
        all_population_size = args.n_arms
        all_transitions, context_prob = randomly_generate_transitions(all_population_size, args.num_context)

    elif args.data == 'local_generated':
        generator = InstanceGenerator(N=args.n_arms, K=args.num_context, seed=66)
        num_instances = 1
        generator.generate_instances(num_instances)
        instances = generator.load_instances(num_instances)
        all_transitions, context_prob = instances[0]['transitions'], instances[0]['context_prob']
    else:
        raise Exception(f'dataset {args.data} not implemented')

    reward_vector = np.ones(args.num_context)
    simulator = Volunteer_RMABSimulator(N=args.n_arms, K=args.num_context, T=args.episode_len, context_prob=context_prob, all_transitions=all_transitions, budget=args.budget, reward_vector=reward_vector)
    return simulator, context_prob

def brute_force_search(simulator, n_episodes, n_epochs, discount):
    context_prob = simulator.context_prob
    B = simulator.budget
    K = simulator.K
    B_UB = [B / prob for prob in context_prob]
    
    results = {}
    
    for budget_vector in itertools.product(*(range(int(b_ub)+1) for b_ub in B_UB[:-1])):
        
        budget_vector_np = np.zeros(K, dtype=int)
        budget_vector_np[:-1] = budget_vector
        if sum(np.multiply(context_prob[:-1], budget_vector_np[:-1])) <= B:
            remaining_budget = B - sum(np.multiply(context_prob[:-1], budget_vector_np[:-1]))
            budget_vector_np[-1] = int(remaining_budget / context_prob[-1])
            if sum(np.multiply(context_prob, budget_vector_np)) <= B:
                # run is here
                rewards = whittle_policy_type_specific(simulator, budget_vector_np, n_episodes=n_episodes, n_epochs=n_epochs, discount=discount)
                print(f"B = {budget_vector_np}, reward = {np.mean(rewards)}")
                results[tuple(budget_vector_np)] = np.mean(rewards)
        # print(f"\nreward = {np.mean(rewards)}\nbudget = {budget_vector_np}")
    return results

def brute_force_search_wrt_allowance(simulator, n_episodes, n_epochs, discount, benchmark_rewards, eps_allowance):
    """
    similar to brute_force_search(...) -> return dict results
    only difference is that only benchmark_rewards[budget] - optimal_rewards_in_benchmark < eps_allowance * range would be runned. hence saving time
    """
    best_allocation = max(benchmark_rewards, key=benchmark_rewards.get)
    max_benchmark_reward = benchmark_rewards[best_allocation]
    worst_allocation = min(benchmark_rewards, key=benchmark_rewards.get)
    min_benchmark_rewards = benchmark_rewards[worst_allocation]
    benchmark_range = max_benchmark_reward - min_benchmark_rewards
    allowance = benchmark_range*eps_allowance
    threshold_of_run = max_benchmark_reward - allowance
    print(f"min_benchmark_rewards, max_benchmark_reward, threshold_of_run = {min_benchmark_rewards, max_benchmark_reward, threshold_of_run}")

    context_prob = simulator.context_prob
    B = simulator.budget
    K = simulator.K
    B_UB = [B / prob for prob in context_prob]
    
    results = {}
    
    for budget_vector in itertools.product(*(range(int(b_ub)+1) for b_ub in B_UB[:-1])):
        
        budget_vector_np = np.zeros(K, dtype=int)
        budget_vector_np[:-1] = budget_vector
        if sum(np.multiply(context_prob[:-1], budget_vector_np[:-1])) <= B:
            remaining_budget = B - sum(np.multiply(context_prob[:-1], budget_vector_np[:-1]))
            budget_vector_np[-1] = int(remaining_budget / context_prob[-1])
            if sum(np.multiply(context_prob, budget_vector_np)) <= B:
                # run is here
                if benchmark_rewards[tuple(budget_vector_np)] >= threshold_of_run:
                    rewards = whittle_policy_type_specific(simulator, budget_vector_np, n_episodes=n_episodes, n_epochs=n_epochs, discount=discount)
                    # print(f"B = {budget_vector_np}, reward = {np.mean(rewards)}")
                    results[tuple(budget_vector_np)] = np.mean(rewards)
                else:
                    results[tuple(budget_vector_np)] = 0
        # print(f"\nreward = {np.mean(rewards)}\nbudget = {budget_vector_np}")
    return results


def main():
    args = parse_arguments()
    simulator, context_prob = initialize_simulator(args)
    results = brute_force_search(simulator, args.n_episodes, args.n_epochs, args.discount)
    
    # Process and save results
    print("Best budget allocation and rewards:")
    best_allocation = max(results, key=results.get)
    print(f"Budget Allocation: {best_allocation}, Reward: {results[best_allocation]}")


if __name__ == '__main__':
    main()