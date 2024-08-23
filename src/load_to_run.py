import numpy as np
import argparse
import time, datetime
import json
from pathlib import Path
import random

import matplotlib.pyplot as plt

from volunteer_simulator import Volunteer_RMABSimulator, randomly_generate_transitions
from volunteer_algorithms import whittle_policy , random_policy, whittle_policy_type_specific #, WIQL
from instance_generator import InstanceGenerator
from brute_search_budget_allocation import brute_force_search
from result_recorder import write_result
from visualization import plot_rewards, plot_type_tuple

"""
load an instance from the result folder (from a previous experiment) to re-run it

key functions:
parse_json_parameters(filepath):
    input `filepath`: paraeter
    return: args, all_transitions, context_prob
    all necessary things to reconstruct and simulate an instance
"""

def load_json_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def run_experiments(args, all_transitions, context_prob):
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

    args.str_time = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')

    np.random.seed(seed)
    random.seed(seed)

    simulator = Volunteer_RMABSimulator(N=N, 
                                        K=K, 
                                        T=T, 
                                        context_prob=context_prob, 
                                        all_transitions=all_transitions, 
                                        budget=budget, 
                                        reward_vector=np.ones(K))
    
    use_algos = ['whittle', 'random', 'type_specific']
    rewards = {}
    runtimes = {}

    rewards  = {}
    rewards_to_write = {}
    runtimes = {}
    # colors   = {'whittle': 'purple', 'ucw_value': 'b', 'ucw_qp': 'c', 'ucw_qp_min': 'goldenrod', 'ucw_ucb': 'darkorange',
    #             'ucw_extreme': 'r', 'wiql': 'limegreen', 'random': 'brown', 'type_specific' : 'goldenrod'}

    if 'whittle' in use_algos:
        print('-------------------------------------------------')
        print('whittle policy')
        start                 = time.time()
        rewards['whittle']    = whittle_policy(simulator, n_episodes, n_epochs, discount)
        print(np.mean(rewards['whittle']))
        rewards_to_write['whittle'] = np.mean(rewards['whittle'])
        runtimes['whittle']   = time.time() - start
        print('-------------------------------------------------')


    if 'random' in use_algos: # random policy
        print('-------------------------------------------------')
        print('random policy')
        start                  = time.time()
        rewards['random']      = random_policy(simulator, n_episodes, n_epochs)
        runtimes['random']     = time.time() - start
        print(np.mean(rewards['random']))
        rewards_to_write['random'] = np.mean(rewards['random'])
        print('-------------------------------------------------')

        if 'type_specific' in use_algos: # type-specific whittle policy
            print('-------------------------------------------------')
            print('brute-force search for type-specific whittle policy')
            simulator.constraint_type = "soft"
            start                  = time.time()
            # variable `results` is the tuple-dict storing all the result

        results = brute_force_search(simulator = simulator, n_episodes=n_episodes, n_epochs=n_epochs, discount=discount)
        # plotting
        plot_type_tuple(results, context_prob, args)
        
        # Process and save results
        print("Best budget allocation and rewards:")
        best_allocation = max(results, key=results.get)
        print(f"Budget Allocation: {best_allocation}, Reward: {results[best_allocation]}")
        rewards['type_specific']      = whittle_policy_type_specific(simulator, best_allocation, n_episodes, n_epochs, discount)
        rewards_to_write['type_specific'] = np.mean(rewards['type_specific'])
        runtimes['type_specific']     = time.time() - start
        print('-------------------------------------------------')

    print('-------------------------------------------------')
    print('runtime')
    for algo in use_algos:
        print(f'  {algo}:   {runtimes[algo]:.2f} s')

    # Write results and visualize

    p, q, _ = simulator.get_original_vectors()


    print('-------------------------------------------------')
    print('Runtime:')
    for algo in use_algos:
        print(f'  {algo}:   {runtimes[algo]:.2f} s')


    write_result(rewards, use_algos, args, all_transitions, context_prob, p, q, rewards_to_write, best_allocation)
    plot_rewards(rewards, use_algos, args)


def parse_json_parameters(filepath):
    data = load_json_data(filepath)
    args = argparse.Namespace(**data)
    context_prob = np.array(data['context_prob'])
    all_transitions = np.array(data['transition_probabilities'])
    return args, all_transitions, context_prob

def main():

    # terminal input: filepath name
    parser = argparse.ArgumentParser(description="Run RMAB experiments from a JSON configuration file.")
    parser.add_argument('filepath', type=str, help='Path to the JSON file containing the experiment setup.')
    args = parser.parse_args()
    
    # Load data from JSON file
    args, all_transitions, context_prob = parse_json_parameters(args.filepath)
    
    # Run experiments
    # print(args)
    run_experiments(args, all_transitions, context_prob)

if __name__ == '__main__':
    main()
