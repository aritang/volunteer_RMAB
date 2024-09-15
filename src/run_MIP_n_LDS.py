import numpy as np
import pandas as pd
import random
import time, datetime
import sys, os
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from volunteer_simulator import Volunteer_RMABSimulator, randomly_generate_transitions
from volunteer_algorithms import whittle_policy , random_policy, whittle_policy_type_specific #, WIQL
from instance_generator import InstanceGenerator
from brute_search_budget_allocation import brute_force_search
from bugdet_allocation_solver import brute_force_plot
from result_recorder import LDS_n_SIM_write_result
from visualization import LDS_n_SIM_plot_result, plot_type_tuple
from LDS_RMAB_formulation import brute_force_search_using_LDS, hard_budget_value_LDS

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_arms',         '-N', help='num beneficiaries (arms)', type=int, default=25)
    parser.add_argument('--budget',         '-B', help='budget', type=int, default=5)
    parser.add_argument('--num_context',    '-K', help='context_size', type=int, default='3')

    parser.add_argument('--episode_len',    '-H', help='episode length', type=int, default=600)
    parser.add_argument('--n_episodes',     '-T', help='num episodes', type=int, default=6)
    parser.add_argument('--data',           '-D', help='dataset to use {synthetic, real, local_generated}', type=str, default='local_generated')

    parser.add_argument('--data_name',           '-DN', help='name of specified dataset', type=str, default=None)

    parser.add_argument('--n_epochs',       '-E', help='number of epochs (num_repeats)', type=int, default=20)
    parser.add_argument('--discount',       '-d', help='discount factor', type=float, default=0.98)

    # doesn't seem necessary
    parser.add_argument('--alpha',          '-a', help='alpha: for conf radius', type=float, default=3)

    # parser.add_argument('--n_states',       '-S', help='num states', type=int, default=2)
    parser.add_argument('--n_actions',      '-A', help='num actions', type=int, default=2)
    # special treatment
    parser.add_argument('--homogeneous', '-HOMO', help='if homogenous', type=str, default='True')
    parser.add_argument('--seed',           '-s', help='random seed', type=int, default=43)
    parser.add_argument('--verbose',        '-V', type=bool, help='if True, then verbose output (default False)', default=False)
    parser.add_argument('--local',          '-L', help='if True, running locally (default False)', action='store_true')
    parser.add_argument('--prefix',         '-p', help='prefix for file writing', type=str, default='')
    parser.add_argument('--name',         '-NAME', help='experiment name', type=str, default='LDS method')


    args = parser.parse_args()
    # special treatement
    args.homogeneous = args.homogeneous.lower() in ['true', '1', 'yes']


    # problem setup
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

    args.exp_name_out = f'{data}_n{n_arms}_b{budget}_s{n_states}_a{n_actions}_H{episode_len}_L{n_episodes}_epochs{n_epochs}'
    
    args.str_time = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')

    if not os.path.exists(f'figures/{data}'):
        os.makedirs(f'figures/{data}')

    if not os.path.exists(f'results/{data}'):
        os.makedirs(f'results/{data}')

    if data == 'local_generated':
        if args.data_name == None:
            all_population_size = N
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
            all_population_size = N
            print(f'loading data named {args.data_name}')
            generator = InstanceGenerator(N=N, K=K, seed=seed)
            num_instances = 1
            instance = generator.load_instance(name=args.data_name)
            all_transitions, context_prob = instance['transitions'], instance['context_prob']
            reward_vector = np.ones(K)

    else:
        raise Exception(f'dataset {data} not implemented')


    if VERBOSE:
        generator.print_instance(all_transitions, context_prob)
        # print(f'transitions ----------------\n{np.round(all_transitions, 2)}')


    simulator = Volunteer_RMABSimulator(N = N, 
                                        K = K, 
                                        T = T, 
                                        context_prob=context_prob, 
                                        all_transitions=all_transitions, 
                                        budget=budget, 
                                        reward_vector=reward_vector
                                        )

    # -------------------------------------------------
    # run comparisons
    # -------------------------------------------------
    # use_algos = ['whittle', 'random', 'type_specific_LDS']
    use_algos = ['type_specific_LDS']
    args.use_algos = use_algos 

    rewards  = {}
    rewards_to_write = {}
    runtimes = {}
    # colors   = {'whittle': 'purple', 'ucw_value': 'b', 'ucw_qp': 'c', 'ucw_qp_min': 'goldenrod', 'ucw_ucb': 'darkorange',
    #             'ucw_extreme': 'r', 'wiql': 'limegreen', 'random': 'brown', 'type_specific' : 'goldenrod'}

    # if 'whittle' in use_algos:
    #     print('-------------------------------------------------')
    #     print('whittle policy')
    #     start                 = time.time()
    #     rewards['whittle']    = whittle_policy(simulator, n_episodes, n_epochs, discount)
    #     print(np.mean(rewards['whittle']))
    #     rewards_to_write['whittle'] = np.mean(rewards['whittle'])
    #     runtimes['whittle']   = time.time() - start
    #     print('-------------------------------------------------')


    # if 'random' in use_algos: # random policy
    #     print('-------------------------------------------------')
    #     print('random policy')
    #     start                  = time.time()
    #     rewards['random']      = random_policy(simulator, n_episodes, n_epochs)
    #     runtimes['random']     = time.time() - start
    #     print(np.mean(rewards['random']))
    #     rewards_to_write['random'] = np.mean(rewards['random'])
    #     print('-------------------------------------------------')

    if 'type_specific_LDS' in use_algos: # type-specific whittle policy
        print('-------------------------------------------------')
        print('brute-force search for type-specific whittle policy')
        simulator.constraint_type = "soft"
        start                  = time.time()
        # variable `results` is the tuple-dict storing all the result
        p, q, _ = generator.get_original_vectors(all_transitions, context_prob)
        results = brute_force_search_using_LDS(N = N, K = K, B = budget, context_prob=context_prob, p = p[0], q = q[0])
        LDS_rewards = results
        
        # Process and save results
        print("Best budget allocation and rewards:")
        best_allocation = max(results, key=results.get)
        print(f"Budget Allocation: {best_allocation}, Reward: {results[best_allocation]}")
        

        rewards['type_specific_LDS'] = hard_budget_value_LDS(N = N, K = K, B = best_allocation, context_prob = context_prob, p = p[0], q = q[0])
        rewards_to_write['type_specific_LDS'] = np.mean(rewards['type_specific_LDS'])
        runtimes['type_specific_LDS']     = time.time() - start
        print('-------------------------------------------------')

    print('-------------------------------------------------')
    print('runtime')
    for algo in use_algos:
        print(f'  {algo}:   {runtimes[algo]:.2f} s')

    # LDS_rewards = brute_force_plot(simulator)
    # x_vals = np.arange(n_episodes * episode_len + 1)

    # def get_cum_sum(reward):
    #     cum_sum = reward.cumsum(axis=1).mean(axis=0)
    #     cum_sum = cum_sum / (x_vals + 1)
    #     return smooth(cum_sum)

    p, q, _ = generator.get_original_vectors(all_transitions, context_prob)
    # -------------------------------------------------
    # visualize
    # -------------------------------------------------
    """
    special part: load SIM rewards from previous experiment and plot SIM alongside LDS
    """
    with open('results/classified/N25_B5_K3_seed43_HOMOTrue/11-09-2024_13:56:43/param_settings.json', 'r') as f:
        data = json.load(f)

    SIM_rewards = data['SIM_rewards']
    def convert_key_str_to_int_list(key_str):
        # Remove the parentheses
        key_str_no_parentheses = key_str.strip('()')
        # Split the string by commas
        key_parts = key_str_no_parentheses.split(',')
        # Convert each part to an integer
        key_list = [int(part.strip()) for part in key_parts]
        return key_list
    
    # Create a new dictionary with integer list keys
    SIM_new_rewards = {}

    for key_str, value in SIM_rewards.items():
        # Convert the string key to an integer list
        key_list = convert_key_str_to_int_list(key_str)
        # Use the integer list as the new key
        SIM_new_rewards[tuple(key_list)] = value  # Using tuple since lists are not hashable

    # plot_type_tuple(reward=SIM_new_rewards, context_prob=context_prob, args=args)
    LDS_n_SIM_write_result(LDS_rewards, SIM_new_rewards, args, all_transitions, context_prob, p, q, result_name = args.name)
    LDS_n_SIM_plot_result(LDS_rewards=LDS_rewards, SIM_rewards=SIM_new_rewards, args=args, p = p[0], q= q[0], context_prob=context_prob, result_name=args.name)
    

    