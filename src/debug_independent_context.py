from utils import parse_arguments
import os
import numpy as np
from instance_generator import initialize_instance_and_simulator, initial_state_generator
from volunteer_algorithms import whittle_policy_type_specific
from LDS_RMAB_formulation import SIMULATE_wrt_LDS
from bugdet_allocation_solver import BudgetSolver
from result_recorder import write_result
from visualization import plot_rewards_over_N

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

if __name__ == '__main__':
    args = parse_arguments()
    N0 = args.N
    budget_0 = args.budget
    # kinda N doesn't matter here. we are going to augment it, say, N \to \infty
    use_algos = ['soft_budget_occupancy_measure', 'independent_context_SIMULATION']
    
    args.size = 10
    averaged_reward_list = {}
    for algo in use_algos:
        averaged_reward_list[algo] = np.zeros(args.size)
    args.use_algos = use_algos
    args.N0 = N0
    

    for size in range(1, args.size + 1):
        args.n_arms = N0*size
        args.N = args.n_arms
        args.budget = int(budget_0*size)
        logging.info(f"running for N = {args.N}")

        # step 1: initialize parameters
        simulator, all_transitions, context_prob, reward_vector = initialize_instance_and_simulator(args = args)
        # auxiliary: store every necessary information in args
        args.all_transitions = all_transitions
        args.context_prob = context_prob
        args.reward_vector = reward_vector
        p, q, _ = simulator.get_original_vectors()
        args.p = p[0]
        args.q = q[0]

        uniform_budget_allocation = np.ones(args.K, dtype=int)
        rewards = {}

        # store average, readable rewards in rewards_to_write
        rewards_to_write = []

        # step 2: run global context simulator:
        if 'global_context_SIMULATION' in use_algos:
            logging.info(f"----running global_context_SIM----")
            rewards['global_context_SIMULATION'] = whittle_policy_type_specific(env=simulator, type_specific_budget=uniform_budget_allocation, n_episodes=args.n_episodes, n_epochs=args.n_epochs, discount=args.discount)
            average_reward = np.mean(rewards['global_context_SIMULATION'])/args.N
            averaged_reward_list['global_context_SIMULATION'][size - 1] = (average_reward)
            logging.info(f"value: {average_reward}")
        
        if 'global_context_LDS' in use_algos:
            logging.info(f"----running global_context_LDS----")
            initial_state_list = initial_state_generator(n_epochs=args.n_epochs, N = args.N, K=args.K, seed=args.seed, context_prob=context_prob)
            rewards['global_context_LDS'] = SIMULATE_wrt_LDS(
                initial_state_list=initial_state_list, 
                N=args.N, 
                K=args.K, 
                budget_allocation=uniform_budget_allocation, 
                context_prob=context_prob, 
                p=args.p, 
                q = args.q, 
                T = args.episode_len, 
                n_episodes=args.n_episodes, 
                n_epochs=args.n_epochs, 
                w=reward_vector
                )
            average_reward = np.mean(rewards['global_context_LDS'])/args.N
            averaged_reward_list['global_context_LDS'][size - 1] = (average_reward)
            logging.info(f"value: {average_reward}")
        
        if 'soft_budget_occupancy_measure' in use_algos:
            logging.info(f"----calculating soft budget MIP(LP) upperbounds----")
            budget_solver = BudgetSolver(
                N = args.N,
                K = args.K,
                B = args.budget,
                all_transitions=all_transitions,
                context_prob=context_prob,
                w=reward_vector,
            )
            budget_solver.solve()
            upper_bound = budget_solver.get_reward()
            rewards['soft_budget_occupancy_measure'] = np.ones((args.n_epochs, args.n_episodes*args.episode_len + 1))*upper_bound
            average_reward = np.mean(rewards['soft_budget_occupancy_measure'])/args.N
            averaged_reward_list['soft_budget_occupancy_measure'][size - 1] = (average_reward)
            logging.info(f"value: {average_reward}")

        if 'soft_budget_FIXED_occupancy_measure' in use_algos:
            logging.info(f"----calculating soft budget MIP(LP) upperbounds----")
            budget_solver = BudgetSolver(
                N = args.N,
                K = args.K,
                B = args.budget,
                all_transitions=all_transitions,
                context_prob=context_prob,
                w=reward_vector,
            )
            budget_solver.set_fix_budgets(uniform_budget_allocation)
            budget_solver.solve()
            upper_bound = budget_solver.get_reward()
            rewards['soft_budget_FIXED_occupancy_measure'] = np.ones((args.n_epochs, args.n_episodes*args.episode_len + 1))*upper_bound
            average_reward = np.mean(rewards['soft_budget_FIXED_occupancy_measure'])/args.N
            averaged_reward_list['soft_budget_FIXED_occupancy_measure'][size - 1] = (average_reward)
            logging.info(f"value: {average_reward}")

        if 'independent_context_SIMULATION' in use_algos:
            logging.info(f"----running INDEPENDENT context simulation----")
            simulator.IF_global_context = False
            rewards['independent_context_SIMULATION'] = whittle_policy_type_specific(env=simulator, type_specific_budget=uniform_budget_allocation, n_episodes=args.n_episodes, n_epochs=args.n_epochs, discount=args.discount)
            simulator.IF_global_context = True
            average_reward = np.mean(rewards['independent_context_SIMULATION'])/args.N
            averaged_reward_list['independent_context_SIMULATION'][size - 1] = (average_reward)
            logging.info(f"value: {average_reward}")

    plot_rewards_over_N(averaged_reward_list, use_algos, args)
    write_result(averaged_reward_list, use_algos, args, all_transitions, context_prob, p, q, rewards_to_write)

    



        

