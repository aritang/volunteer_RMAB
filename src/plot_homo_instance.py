# purpose: plot MIP solved solution (Upper-bound) against budget simulation
# use MIP solved solution to induce several good solutions, and use it for simulation
# (i) Budget Allocation = MIP's optimal
# (ii) Budget Allocation = best among a few MIP's closely-optimal options.
# - need to define an optimality gap eps for (ii)


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
from brute_search_budget_allocation import brute_force_search_wrt_allowance
from result_recorder import write_result
from visualization import plot_rewards, plot_type_tuple
from bugdet_allocation_solver import brute_force_plot

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
    parser.add_argument('--n_arms_basic',         '-N', help='the smallest instance', type=int, default=25)
    parser.add_argument('--budget_basic',               '-B', help='corresponding to the smallest instance, the smallest budget', type=int, default=5)
    parser.add_argument('--num_context',          '-K', help='context_size', type=int, default='3')

    parser.add_argument('--episode_len',    '-H', help='episode length', type=int, default=600)
    parser.add_argument('--n_episodes',     '-T', help='num episodes', type=int, default=6)
    parser.add_argument('--data',           '-D', help='dataset to use {local_generated or others}', type=str, default='local_generated')

    parser.add_argument('--n_epochs',       '-E', help='number of epochs (num_repeats)', type=int, default=20)
    parser.add_argument('--discount',       '-d', help='discount factor', type=float, default=0.98)

    # parser.add_argument('--n_states',       '-S', help='num states', type=int, default=2)
    parser.add_argument('--n_actions',      '-A', help='num actions', type=int, default=2)
    # special treatment
    parser.add_argument('--homogeneous',    '-HOMO', help='if homogenous', type=str, default='True')
    parser.add_argument('--seed',           '-s', help='random seed', type=int, default=43)
    parser.add_argument('--verbose',        '-V', type=bool, help='if True, then verbose output (default False)', default=False)
    parser.add_argument('--prefix',         '-p', help='prefix for file writing', type=str, default='')

    parser.add_argument('--times_augmented',     '-TIMES', help='simulate np.arange(N, N*TIMES, N)', type=int, default=10)
    parser.add_argument('--eps_allowance',       '-EPS', help='PERCENTAGE of suboptimality range for brute-force search budget', type=float, default=0.05)


    args = parser.parse_args()
    # special treatement
    args.homogeneous = args.homogeneous.lower() in ['true', '1', 'yes']


    # problem setup
    TIMES = args.times_augmented
    N = args.n_arms_basic
    budget = args.budget_basic

    K = args.num_context
    n_states = 2*K
    n_actions   = args.n_actions

    # solution/evaluation setup
    discount    = args.discount
    homogeneous    = args.homogeneous

    # experiment setup
    seed        = args.seed
    VERBOSE     = args.verbose
    prefix      = args.prefix

    n_episodes  = args.n_episodes
    episode_len = args.episode_len
    T = episode_len
    n_epochs    = args.n_epochs

    # default forever as 'local-generated'
    data        = args.data
    eps_allowance = args.eps_allowance

    np.random.seed(seed)
    random.seed(seed)

    args.exp_name_out = f'{data}_n{N}_b{budget}_seed{seed}_H{episode_len}_L{n_episodes}_epochs{n_epochs}'
    
    args.str_time = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')

    if not os.path.exists(f'results/{data}'):
        os.makedirs(f'results/{data}')

    MIP_result = np.zeros(TIMES)
    SIM_from_MIP_Budget_result = np.zeros(TIMES)
    SIM_approxOPT_result = np.zeros(TIMES)
    MIP_opt_budget = []
    SIM_approxOPT_budget = []

    for TIME in np.arange(1, TIMES + 1):
        print(f"------------------------")
        N_now = N*TIME
        budget_now = budget*TIME

        if data == 'local_generated':
            all_population_size = N_now
            generator = InstanceGenerator(N=N_now, K=K, seed=seed)
            num_instances = 1
            generator.generate_instances(num_instances, homogeneous=homogeneous)
            instance = generator.load_instance()
            all_transitions, context_prob = instance['transitions'], instance['context_prob']
    
            reward_vector = np.ones(K)

        elif data == 'specified':
            homogeneous = True
            K = 2
            p = np.array([0.999, 0])
            q = 0.2
            context_prob = np.array([0.5, 0.5])
            generator = InstanceGenerator(N=N_now, K=K, seed=seed)
            generator.generate_instance_given_probs(p = np.array([p for i in range(N_now)]), q = np.ones(N_now)*q, context_prob=context_prob, name = data)
            instance = generator.load_instance()
            all_transitions, context_prob = instance['transitions'], instance['context_prob']
            reward_vector = np.ones(K)

        else:
            raise Exception(f'dataset {data} not implemented')

        simulator = Volunteer_RMABSimulator(N = N_now, 
                                            K = K, 
                                            T = T, 
                                            context_prob=context_prob, 
                                            all_transitions=all_transitions, 
                                            budget=budget_now, 
                                            reward_vector=reward_vector
                                            )

        # obtain MIP's result, store for plot
        opt_value, MIP_best_allocation = solve_budget(simulator, MIP=True)
        MIP_result[TIME-1] = opt_value/N_now
        MIP_opt_budget.append(MIP_best_allocation.tolist())

        # obtain SIM's approximate optimal result, store for plot
        MIP_rewards = brute_force_plot(simulator)
        SIM_rewards = brute_force_search_wrt_allowance(simulator=simulator, n_episodes=n_episodes, n_epochs=n_epochs, benchmark_rewards=MIP_rewards, eps_allowance=eps_allowance, discount=discount)
        SIM_approx_best_allocation = max(SIM_rewards, key=SIM_rewards.get)
        SIM_approxOPT_budget.append(list(SIM_approx_best_allocation))

        rewards = whittle_policy_type_specific(simulator, np.array(SIM_approx_best_allocation, dtype=int), n_episodes=n_episodes, n_epochs=n_epochs, discount=discount)
        SIM_approxOPT_result[TIME-1] = np.mean(rewards)/N_now

        rewards = whittle_policy_type_specific(simulator, np.array(MIP_best_allocation, dtype=int), n_episodes=n_episodes, n_epochs=n_epochs, discount=discount)
        SIM_from_MIP_Budget_result[TIME-1] = np.mean(rewards)/N_now
        
        print(f"N = {N_now}, B = {budget_now}\nMIP_best_allocation = {tuple(MIP_best_allocation)}\nSIM_approx_best_allocation = {SIM_approx_best_allocation}\nMIP_reward={MIP_result[TIME - 1]}\nSIM_approxOPT_reward={SIM_approxOPT_result[TIME - 1]}\nSIM_from_MIP_Budget_result={SIM_from_MIP_Budget_result[TIME - 1]}\n")

    N_list = np.arange(1, TIMES + 1)*N
    sns.lineplot(x = N_list, y = MIP_result, label = "MIP result")
    sns.lineplot(x = N_list, y = SIM_from_MIP_Budget_result, label = "SIM_from_MIP_Budget_result")
    sns.lineplot(x = N_list, y = SIM_approxOPT_result, label = "SIM_approxOPT_result")
    plt.title(f"alpha (N/K) = {N/budget}")
    plt.xlabel(f"N")
    plt.ylabel(f"reward/(N*T) ")
    

    this_path = f'./results/{args.str_time}_' + args.data
    if not os.path.exists(this_path):
        os.makedirs(this_path)

    plt.savefig(this_path + '/when_N_goes_large.png')

    args_dict = {}
    args_dict = vars(args)
    args_dict['MIP_result'] = MIP_result
    args_dict['SIM_from_MIP_Budget_result'] = SIM_from_MIP_Budget_result
    args_dict['SIM_approxOPT_result'] = SIM_approxOPT_result
    args_dict['MIP_opt_budget'] = MIP_opt_budget
    args_dict['SIM_approxOPT_budget'] = SIM_approxOPT_budget
    
    if data == 'specified':
        args_dict['p'] = p
        args_dict['q'] = q
        args_dict['context_prob'] = context_prob
    else:
        p, q, context_prob = generator.get_original_vectors(all_transitions, context_prob)
        if homogeneous:
            args_dict['p'] = p[0]
            args_dict['q'] = q[0]
        else:
            args_dict['p'] = p
            args_dict['q'] = q
        args_dict['context_prob'] = context_prob

    json_filename = this_path + '/param_settings.json'
    with open(json_filename, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4, default=str)




