import numpy as np
import pandas as pd
import random
import time, datetime
import sys, os
import argparse

import matplotlib.pyplot as plt

from volunteer_simulator import Volunteer_RMABSimulator, randomly_generate_transitions
# from uc_whittle import UCWhittle
# from ucw_value import UCWhittle_value
from volunteer_algorithms import whittle_policy , random_policy, whittle_policy_type_specific #, WIQL
from instance_generator import InstanceGenerator
from brute_search_budget_allocation import brute_force_search


def smooth(rewards, weight=0.7):
    """ smoothed exponential moving average """
    prev = rewards[0]
    smoothed = np.zeros(len(rewards))
    for i, val in enumerate(rewards):
        smoothed_val = prev * weight + (1 - weight) * val
        smoothed[i] = smoothed_val
        prev = smoothed_val

    return smoothed

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_arms',         '-N', help='num beneficiaries (arms)', type=int, default=60)
    parser.add_argument('--budget',         '-B', help='budget', type=int, default=20)
    parser.add_argument('--num_context',    '-K', help='context_size', type=int, default='3')

    parser.add_argument('--episode_len',    '-H', help='episode length', type=int, default=100)
    parser.add_argument('--n_episodes',     '-T', help='num episodes', type=int, default=6)
    parser.add_argument('--data',           '-D', help='dataset to use {synthetic, real, local_generated}', type=str, default='local_generated')

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
    # real_data   = args.real_data


    # separate out things we don't want to execute on the cluster
    if LOCAL:
        import matplotlib as mpl
        mpl.use('tkagg')

    np.random.seed(seed)
    random.seed(seed)


    if not os.path.exists(f'figures/{data}'):
        os.makedirs(f'figures/{data}')

    if not os.path.exists(f'results/{data}'):
        os.makedirs(f'results/{data}')


    # -------------------------------------------------
    # initialize RMAB simulator
    # -------------------------------------------------

    # not supported for now
    if data in ['real']:
        if data == 'real':
            print('real data')
            transitions = get_armman_data()

        assert n_arms <= transitions.shape[0]
        assert transitions.shape[1] == n_states
        assert transitions.shape[2] == n_actions
        all_population_size = transitions.shape[0]

        if all_population_size < transitions.shape[0]:
            transitions = transitions[0:all_population_size, :]

        all_transitions = np.zeros((all_population_size, n_states, n_actions, n_states))
        all_transitions[:,:,:,1] = transitions
        all_transitions[:,:,:,0] = 1 - transitions
    
    # this should work
    elif data == 'synthetic':
        print('synthetic data')
        all_population_size = N # number of random arms to generate
        all_transitions, context_prob = randomly_generate_transitions(all_population_size, context_size)
        reward_vector = np.ones(K)

    elif data == 'local_generated':
        all_population_size = N
        print('using locally generated data w.r.t. a seed that fixing everything')
        # generate local data using a same seed—if N, K be the same, generated probability transitions should also be the same
        generator = InstanceGenerator(N=N, K=K, seed=66)
        # let's say we only support one instance for now
        num_instances = 1
        generator.generate_instances(num_instances)
        instances = generator.load_instances(num_instances)
        all_transitions, context_prob = instances[0]['transitions'], instances[0]['context_prob']
    
        reward_vector = np.ones(K)

    else:
        raise Exception(f'dataset {data} not implemented')


    all_features = np.arange(all_population_size)

    if VERBOSE: print(f'transitions ----------------\n{np.round(all_transitions, 2)}')
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

    # use_algos = ['whittle', 'ucw_value', 'ucw_qp', 'ucw_extreme', 'wiql', 'random'] # 'ucw_qp_min', 'ucw_ucb',
    use_algos = ['whittle', 'random', 'type_specific'] # 'ucw_qp_min', 'ucw_ucb',

    rewards  = {}
    runtimes = {}
    colors   = {'whittle': 'purple', 'ucw_value': 'b', 'ucw_qp': 'c', 'ucw_qp_min': 'goldenrod', 'ucw_ucb': 'darkorange',
                'ucw_extreme': 'r', 'wiql': 'limegreen', 'random': 'brown', 'type_specific' : 'goldenrod'}

    if 'whittle' in use_algos:
        print('-------------------------------------------------')
        print('whittle policy')
        print('-------------------------------------------------')
        start                 = time.time()
        rewards['whittle']    = whittle_policy(simulator, n_episodes, n_epochs, discount)
        runtimes['whittle']   = time.time() - start


    if 'random' in use_algos: # random policy
        print('-------------------------------------------------')
        print('random policy')
        print('-------------------------------------------------')
        start                  = time.time()
        rewards['random']      = random_policy(simulator, n_episodes, n_epochs)
        runtimes['random']     = time.time() - start

    if 'type_specific' in use_algos: # type-specific whittle policy
        print('-------------------------------------------------')
        print('brute-force search for type-specific whittle policy')
        print('-------------------------------------------------')
        start                  = time.time()
        results = brute_force_search(simulator = simulator, n_episodes=n_episodes, n_epochs=n_epochs, discount=discount)
        
        # Process and save results
        print("Best budget allocation and rewards:")
        best_allocation = max(results, key=results.get)
        print(f"Budget Allocation: {best_allocation}, Reward: {results[best_allocation]}")
        rewards['type_specific']      = whittle_policy_type_specific(simulator, best_allocation, n_episodes, n_epochs, discount)
        runtimes['type_specific']     = time.time() - start

    print('-------------------------------------------------')
    print('runtime')
    for algo in use_algos:
        print(f'  {algo}:   {runtimes[algo]:.2f} s')


    x_vals = np.arange(n_episodes * episode_len + 1)

    def get_cum_sum(reward):
        cum_sum = reward.cumsum(axis=1).mean(axis=0)
        cum_sum = cum_sum / (x_vals + 1)
        return smooth(cum_sum)

    exp_name_out = f'{data}_n{n_arms}_b{budget}_s{n_states}_a{n_actions}_H{episode_len}_L{n_episodes}_epochs{n_epochs}'

    str_time = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')


    # -------------------------------------------------
    # write out CSV
    # -------------------------------------------------
    for algo in use_algos:
        data_df = pd.DataFrame(data=rewards[algo], columns=x_vals)

        runtime = runtimes[algo] / n_epochs
        prepend_df = pd.DataFrame({'seed': seed, 'n_arms': n_arms, 'budget': budget,
                                    'n_states': n_states, 'n_actions': n_actions,
                                    'discount': discount, 'n_episodes': n_episodes, 'episode_len': episode_len,
                                    'n_epochs': n_epochs, 'runtime': runtime, 'time': str_time}, index=[0])

        prepend_df = pd.concat([prepend_df]*n_epochs, ignore_index=True)

        out_df = pd.concat([prepend_df, data_df], axis=1)

        filename = f'results/{data}/reward_{exp_name_out}_{algo}.csv'

        with open(filename, 'a') as f:
            # write header, if file doesn't exist
            if f.tell() == 0:
                print(f'creating file {filename}')
                out_df.to_csv(f)

            # write results (appending) and no header
            else:
                print(f'appending to file {filename}')
                out_df.to_csv(f, mode='a', header=False)

    # -------------------------------------------------
    # visualize
    # -------------------------------------------------
    # plot average cumulative reward
    plt.figure()
    for algo in use_algos:
        plt.plot(x_vals, get_cum_sum(rewards[algo]), c=colors[algo], label=algo)
    plt.legend()
    plt.xlabel(f'timestep $t$ ({n_episodes} episodes of length {episode_len})')
    plt.ylabel('average cumulative reward')
    plt.title(f'{data} - N={n_arms}, B={budget}, discount={discount}, {n_epochs} epochs')
    plt.savefig(f'figures/{data}/cum_reward_{exp_name_out}_{str_time}.pdf')
    if LOCAL: plt.show()

    # plot average reward
    plt.figure()
    for algo in use_algos:
        plt.plot(x_vals, smooth(rewards[algo].mean(axis=0)), c=colors[algo], label=algo)
    plt.legend()
    plt.xlabel(f'timestep $t$ ({n_episodes} episodes of length {episode_len}')
    plt.ylabel('average reward')
    plt.title(f'{data} - N={n_arms}, budget={budget}, discount={discount}, {n_epochs} epochs')
    plt.savefig(f'figures/{data}/avg_reward_{exp_name_out}_{str_time}.pdf')
    if LOCAL: plt.show()
