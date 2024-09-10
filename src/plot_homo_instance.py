# purpose: plot MIP solved solution (Upper-bound) against budget simulation

import numpy as np
import pandas as pd
import random
import time, datetime
import sys, os
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from volunteer_simulator import Volunteer_RMABSimulator, randomly_generate_transitions
from volunteer_algorithms import whittle_policy , random_policy, whittle_policy_type_specific #, WIQL
from instance_generator import InstanceGenerator
from brute_search_budget_allocation import brute_force_search
from result_recorder import write_result
from visualization import plot_rewards, plot_type_tuple

from bugdet_allocation_solver import BudgetSolver, solve_budget


# def smooth(rewards, weight=0.7):
#     """ smoothed exponential moving average """
#     prev = rewards[0]
#     smoothed = np.zeros(len(rewards))
#     for i, val in enumerate(rewards):
#         smoothed_val = prev * weight + (1 - weight) * val
#         smoothed[i] = smoothed_val
#         prev = smoothed_val

#     return smoothed

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_arms',         '-N', help='num beneficiaries (arms)', type=int, default=60)
    parser.add_argument('--budget',         '-B', help='budget', type=int, default=20)
    parser.add_argument('--num_context',    '-K', help='context_size', type=int, default='2')

    parser.add_argument('--episode_len',    '-H', help='episode length', type=int, default=357)
    parser.add_argument('--n_episodes',     '-T', help='num episodes', type=int, default=6)
    parser.add_argument('--data',           '-D', help='dataset to use {synthetic, real, local_generated}', type=str, default='local_generated')

    parser.add_argument('--n_epochs',       '-E', help='number of epochs (num_repeats)', type=int, default=6)
    parser.add_argument('--discount',       '-d', help='discount factor', type=float, default=0.98)

    # doesn't seem necessary
    parser.add_argument('--alpha',          '-a', help='alpha: for conf radius', type=float, default=3)

    # parser.add_argument('--n_states',       '-S', help='num states', type=int, default=2)
    parser.add_argument('--n_actions',      '-A', help='num actions', type=int, default=2)
    # special treatment
    parser.add_argument('--homogeneous', '-HOMO', help='if homogenous', type=str, default='True')
    parser.add_argument('--seed',           '-s', help='random seed', type=int, default=42)
    parser.add_argument('--verbose',        '-V', type=bool, help='if True, then verbose output (default False)', default=False)
    parser.add_argument('--local',          '-L', help='if True, running locally (default False)', action='store_true')
    parser.add_argument('--prefix',         '-p', help='prefix for file writing', type=str, default='')

    parser.add_argument('--times_augmented',         '-TIMES', help='simulate np.arange(N, N*TIMES, N)', type=int)


    args = parser.parse_args()
    # special treatement
    args.homogeneous = args.homogeneous.lower() in ['true', '1', 'yes']


    # problem setup
    TIMES = args.times_augmented
    n_arms      = args.n_arms
    N = n_arms
    budget      = args.budget
    # n_states    = args.n_states
    context_size = args.num_context
    K = context_size
    n_states = 2*K
    n_actions   = args.n_actions

    # solution/evaluation setup
    discount    = args.discount
    homogeneous    = args.homogeneous
    alpha       = args.alpha #7 - too pessimistic #0.1 - too optimistic

    # experiment setup
    seed        = args.seed
    VERBOSE     = args.verbose
    LOCAL       = args.local
    prefix      = args.prefix
    n_episodes  = args.n_episodes
    episode_len = args.episode_len
    T = episode_len
    n_epochs    = args.n_epochs
    data        = args.data

    np.random.seed(seed)
    random.seed(seed)

    args.exp_name_out = f'{data}_n{n_arms}_b{budget}_seed{seed}_H{episode_len}_L{n_episodes}_epochs{n_epochs}'
    
    args.str_time = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')

    if not os.path.exists(f'results/{data}'):
        os.makedirs(f'results/{data}')

    MIP_result = np.zeros(TIMES)
    NumSim_result = np.zeros(TIMES)
    MIP_budget = dict()

    for TIME in np.arange(1, TIMES + 1):
        N_now = N*TIME
        budget_now = budget*TIME

        if data == 'local_generated':
            all_population_size = N_now
            print('using locally generated data w.r.t. a seed that fixing everything')
            # generate local data using a same seed—if N, K be the same, generated probability transitions should also be the same
            generator = InstanceGenerator(N=N, K=K, seed=seed)
            # let's say we only support one instance for now
            num_instances = 1
            generator.generate_instances(num_instances, homogeneous=homogeneous)
            instance = generator.load_instance()
            all_transitions, context_prob = instance['transitions'], instance['context_prob']
    
            reward_vector = np.ones(K)
        else:
            raise Exception(f'dataset {data} not implemented')


        if VERBOSE:
            generator.print_instance(all_transitions, context_prob)
            # print(f'transitions ----------------\n{np.round(all_transitions, 2)}')

        simulator = Volunteer_RMABSimulator(N = N_now, 
                                            K = K, 
                                            T = T, 
                                            context_prob=context_prob, 
                                            all_transitions=all_transitions, 
                                            budget=budget_now, 
                                            reward_vector=reward_vector
                                            )
    
        opt_value, best_allocation = solve_budget(simulator, MIP=True)
        # print(f"--> LP  directly solved budget allocation = {best_allocation}\n opt_val = {opt_value}")
        # rewards[tuple(best_allocation.tolist())] = opt_value
        MIP_result[TIME] = opt_value
        MIP_budget[TIME] = best_allocation.tolist()

        rewards = whittle_policy_type_specific(simulator, best_allocation, n_episodes=n_episodes, n_epochs=n_epochs, discount=discount)
        NumSim_result[TIME] = np.mean(rewards)

    sns.lineplot(MIP_result, label = "MIP result")
    sns.lineplot(NumSim_result, label = "NumSim result")
