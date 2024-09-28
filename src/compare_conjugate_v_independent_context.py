from utils import parse_arguments
import os
import numpy as np
from instance_generator import initialize_instance_and_simulator, initial_state_generator
from volunteer_algorithms import whittle_policy_type_specific
from LDS_RMAB_formulation import SIMULATE_wrt_LDS
from bugdet_allocation_solver import BudgetSolver
from result_recorder import write_result
from visualization import plot_rewards

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

if __name__ == '__main__':
    args = parse_arguments()

    # step 1: initialize parameters
    simulator, all_transitions, context_prob, reward_vector = initialize_instance_and_simulator(args = args)
    # auxiliary: store every necessary information in args
    args.all_transitions = all_transitions
    args.context_prob = context_prob
    args.reward_vector = reward_vector
    p, q, _ = simulator.get_original_vectors()
    args.p = p[0]
    args.q = q[0]

    # fixed on analyzing uniform budget allocation, for starter
    uniform_budget_allocation = np.ones(args.K, dtype=int)*args.budget
    known_good_budget_allocation = np.array([12, 0, 8], dtype=int)
    use_algos = ['global_context_SIMULATION-uniform_budget_allocation', 
                 'global_context_SIMULATION-(approx.)best_budget_allocation',
                 'global_context_LDS-uniform_budget_allocation',
                 'global_context_LDS-(approx.)best_budget_allocation',
                 'soft_budget_occupancy_measure-FIXED_uniform_budget_allocation',
                 'soft_budget_occupancy_measure-FIXED_(approx.)best_budget_allocation',
                 'independent_context_SIMULATION',
                 'soft_budget_occupancy_measure']
    args.use_algos = use_algos
    rewards = {}

    # store average, readable rewards in rewards_to_write
    rewards_to_write = {}

    # # step 2: run global context simulator:
    # if 'global_context_SIMULATION-uniform_budget_allocation' in use_algos:
    #     key = 'global_context_SIMULATION-uniform_budget_allocation'
    #     logging.info(f"----{key}----")
    #     rewards[key] = whittle_policy_type_specific(env=simulator, type_specific_budget=uniform_budget_allocation, n_episodes=args.n_episodes, n_epochs=args.n_epochs, discount=args.discount)
    #     average_reward = np.mean(rewards[key])
    #     rewards_to_write[key] = average_reward
    #     logging.info(f"value: {average_reward}")
    
    # if 'global_context_SIMULATION-(approx.)best_budget_allocation' in use_algos:
    #     logging.info(f"----running global_context_SIMULATION-(approx.)best_budget_allocation----")
    #     rewards['global_context_SIMULATION-(approx.)best_budget_allocation'] = whittle_policy_type_specific(env=simulator, type_specific_budget=known_good_budget_allocation, n_episodes=args.n_episodes, n_epochs=args.n_epochs, discount=args.discount)
    #     average_reward = np.mean(rewards['global_context_SIMULATION-(approx.)best_budget_allocation'])
    #     rewards_to_write['global_context_SIMULATION-(approx.)best_budget_allocation'] = average_reward
    #     logging.info(f"value: {average_reward}")
    
    # if 'global_context_LDS-uniform_budget_allocation' in use_algos:
    #     logging.info(f"----running global_context_LDS-uniform_budget_allocation----")
    #     initial_state_list = initial_state_generator(n_epochs=args.n_epochs, N = args.N, K=args.K, seed=args.seed, context_prob=context_prob)
    #     rewards['global_context_LDS-uniform_budget_allocation'] = SIMULATE_wrt_LDS(
    #         initial_state_list=initial_state_list, 
    #         N=args.N, 
    #         K=args.K, 
    #         budget_allocation=uniform_budget_allocation, 
    #         context_prob=context_prob, 
    #         p=args.p, 
    #         q = args.q, 
    #         T = args.episode_len, 
    #         n_episodes=args.n_episodes, 
    #         n_epochs=args.n_epochs, 
    #         w=reward_vector
    #         )
    #     average_reward = np.mean(rewards['global_context_LDS-uniform_budget_allocation'])
    #     rewards_to_write['global_context_LDS-uniform_budget_allocation'] = average_reward
    #     logging.info(f"value: {average_reward}")
    
    # if 'global_context_LDS-(approx.)best_budget_allocation' in use_algos:
    #     logging.info(f"----running global_context_LDS-(approx.)best_budget_allocation----")
    #     initial_state_list = initial_state_generator(n_epochs=args.n_epochs, N = args.N, K=args.K, seed=args.seed, context_prob=context_prob)
    #     rewards['global_context_LDS-(approx.)best_budget_allocation'] = SIMULATE_wrt_LDS(
    #         initial_state_list=initial_state_list, 
    #         N=args.N, 
    #         K=args.K, 
    #         budget_allocation=known_good_budget_allocation, 
    #         context_prob=context_prob, 
    #         p=args.p, 
    #         q = args.q, 
    #         T = args.episode_len, 
    #         n_episodes=args.n_episodes, 
    #         n_epochs=args.n_epochs, 
    #         w=reward_vector
    #         )
    #     average_reward = np.mean(rewards['global_context_LDS-(approx.)best_budget_allocation'])
    #     rewards_to_write['global_context_LDS-(approx.)best_budget_allocation'] = average_reward
    #     logging.info(f"value: {average_reward}")

    # if 'soft_budget_occupancy_measure-FIXED_uniform_budget_allocation' in use_algos:
    #     logging.info(f"----calculating soft_budget_occupancy_measure-FIXED_uniform_budget_allocation----")
    #     budget_solver = BudgetSolver(
    #         N = args.N,
    #         K = args.K,
    #         B = args.budget,
    #         all_transitions=all_transitions,
    #         context_prob=context_prob,
    #         w=reward_vector,
    #     )
    #     budget_solver.set_fix_budgets(uniform_budget_allocation)
    #     budget_solver.solve()
    #     upper_bound = budget_solver.get_reward()
    #     rewards['soft_budget_occupancy_measure-FIXED_uniform_budget_allocation'] = np.ones((args.n_epochs, args.n_episodes*args.episode_len + 1))*upper_bound
    #     average_reward = np.mean(rewards['soft_budget_occupancy_measure-FIXED_uniform_budget_allocation'])
    #     rewards_to_write['soft_budget_occupancy_measure-FIXED_uniform_budget_allocation'] = average_reward
    #     logging.info(f"value: {average_reward}")

    # if 'soft_budget_occupancy_measure-FIXED_(approx.)best_budget_allocation' in use_algos:
    #     logging.info(f"----calculating soft_budget_occupancy_measure-FIXED_(approx.)best_budget_allocation----")
    #     budget_solver = BudgetSolver(
    #         N = args.N,
    #         K = args.K,
    #         B = args.budget,
    #         all_transitions=all_transitions,
    #         context_prob=context_prob,
    #         w=reward_vector,
    #     )
    #     budget_solver.set_fix_budgets(known_good_budget_allocation)
    #     budget_solver.solve()
    #     upper_bound = budget_solver.get_reward()
    #     rewards['soft_budget_occupancy_measure-FIXED_(approx.)best_budget_allocation'] = np.ones((args.n_epochs, args.n_episodes*args.episode_len + 1))*upper_bound
    #     average_reward = np.mean(rewards['soft_budget_occupancy_measure-FIXED_(approx.)best_budget_allocation'])
    #     rewards_to_write['soft_budget_occupancy_measure-FIXED_(approx.)best_budget_allocation'] = average_reward
    #     logging.info(f"value: {average_reward}")

    # if 'independent_context_SIMULATION' in use_algos:
    #     logging.info(f"----running independent_context_SIMULATION----")
    #     simulator.IF_global_context = False
    #     rewards['independent_context_SIMULATION'] = whittle_policy_type_specific(env=simulator, type_specific_budget=uniform_budget_allocation, n_episodes=args.n_episodes, n_epochs=args.n_epochs, discount=args.discount)
    #     simulator.IF_global_context = False
    #     average_reward = np.mean(rewards['independent_context_SIMULATION'])
    #     rewards_to_write['independent_context_SIMULATION'] = average_reward
    #     logging.info(f"value: {average_reward}")

    # if 'soft_budget_occupancy_measure' in use_algos:
    #     logging.info(f"----calculating soft_budget_occupancy_measure----")
    #     budget_solver = BudgetSolver(
    #         N = args.N,
    #         K = args.K,
    #         B = args.budget,
    #         all_transitions=all_transitions,
    #         context_prob=context_prob,
    #         w=reward_vector,
    #     )
    #     budget_solver.solve()
    #     upper_bound = budget_solver.get_reward()
    #     rewards['soft_budget_occupancy_measure'] = np.ones((args.n_epochs, args.n_episodes*args.episode_len + 1))*upper_bound
    #     average_reward = np.mean(rewards['soft_budget_occupancy_measure'])
    #     rewards_to_write['soft_budget_occupancy_measure'] = average_reward
    #     logging.info(f"value: {average_reward}")

    # step 2: run global context simulator:
    if 'global_context_SIMULATION-uniform_budget_allocation' in use_algos:
        key = 'global_context_SIMULATION-uniform_budget_allocation'
        logging.info(f"----{key}----")
        rewards[key] = whittle_policy_type_specific(
            env=simulator,
            type_specific_budget=uniform_budget_allocation,
            n_episodes=args.n_episodes,
            n_epochs=args.n_epochs,
            discount=args.discount
        )
        average_reward = np.mean(rewards[key])
        rewards_to_write[key] = average_reward
        logging.info(f"value: {average_reward}")

    if 'global_context_SIMULATION-(approx.)best_budget_allocation' in use_algos:
        key = 'global_context_SIMULATION-(approx.)best_budget_allocation'
        logging.info(f"----{key}----")
        rewards[key] = whittle_policy_type_specific(
            env=simulator,
            type_specific_budget=known_good_budget_allocation,
            n_episodes=args.n_episodes,
            n_epochs=args.n_epochs,
            discount=args.discount
        )
        average_reward = np.mean(rewards[key])
        rewards_to_write[key] = average_reward
        logging.info(f"value: {average_reward}")

    if 'global_context_LDS-uniform_budget_allocation' in use_algos:
        key = 'global_context_LDS-uniform_budget_allocation'
        logging.info(f"----{key}----")
        initial_state_list = initial_state_generator(
            n_epochs=args.n_epochs,
            N=args.N,
            K=args.K,
            seed=args.seed,
            context_prob=context_prob
        )
        rewards[key] = SIMULATE_wrt_LDS(
            initial_state_list=initial_state_list,
            N=args.N,
            K=args.K,
            budget_allocation=uniform_budget_allocation,
            context_prob=context_prob,
            p=args.p,
            q=args.q,
            T=args.episode_len,
            n_episodes=args.n_episodes,
            n_epochs=args.n_epochs,
            w=reward_vector
        )
        average_reward = np.mean(rewards[key])
        rewards_to_write[key] = average_reward
        logging.info(f"value: {average_reward}")

    if 'global_context_LDS-(approx.)best_budget_allocation' in use_algos:
        key = 'global_context_LDS-(approx.)best_budget_allocation'
        logging.info(f"----{key}----")
        initial_state_list = initial_state_generator(
            n_epochs=args.n_epochs,
            N=args.N,
            K=args.K,
            seed=args.seed,
            context_prob=context_prob
        )
        rewards[key] = SIMULATE_wrt_LDS(
            initial_state_list=initial_state_list,
            N=args.N,
            K=args.K,
            budget_allocation=known_good_budget_allocation,
            context_prob=context_prob,
            p=args.p,
            q=args.q,
            T=args.episode_len,
            n_episodes=args.n_episodes,
            n_epochs=args.n_epochs,
            w=reward_vector
        )
        average_reward = np.mean(rewards[key])
        rewards_to_write[key] = average_reward
        logging.info(f"value: {average_reward}")

    if 'soft_budget_occupancy_measure-FIXED_uniform_budget_allocation' in use_algos:
        key = 'soft_budget_occupancy_measure-FIXED_uniform_budget_allocation'
        logging.info(f"----calculating {key}----")
        budget_solver = BudgetSolver(
            N=args.N,
            K=args.K,
            B=args.budget,
            all_transitions=all_transitions,
            context_prob=context_prob,
            w=reward_vector,
        )
        budget_solver.set_fix_budgets(uniform_budget_allocation)
        budget_solver.solve()
        upper_bound = budget_solver.get_reward()
        rewards[key] = np.ones(
            (args.n_epochs, args.n_episodes * args.episode_len + 1)
        ) * upper_bound
        average_reward = np.mean(rewards[key])
        rewards_to_write[key] = average_reward
        logging.info(f"value: {average_reward}")

    if 'soft_budget_occupancy_measure-FIXED_(approx.)best_budget_allocation' in use_algos:
        key = 'soft_budget_occupancy_measure-FIXED_(approx.)best_budget_allocation'
        logging.info(f"----calculating {key}----")
        budget_solver = BudgetSolver(
            N=args.N,
            K=args.K,
            B=args.budget,
            all_transitions=all_transitions,
            context_prob=context_prob,
            w=reward_vector,
        )
        budget_solver.set_fix_budgets(known_good_budget_allocation)
        budget_solver.solve()
        upper_bound = budget_solver.get_reward()
        rewards[key] = np.ones(
            (args.n_epochs, args.n_episodes * args.episode_len + 1)
        ) * upper_bound
        average_reward = np.mean(rewards[key])
        rewards_to_write[key] = average_reward
        logging.info(f"value: {average_reward}")

    if 'independent_context_SIMULATION' in use_algos:
        key = 'independent_context_SIMULATION'
        logging.info(f"----running {key}----")
        simulator.IF_global_context = False
        rewards[key] = whittle_policy_type_specific(
            env=simulator,
            type_specific_budget=uniform_budget_allocation,
            n_episodes=args.n_episodes,
            n_epochs=args.n_epochs,
            discount=args.discount
        )
        simulator.IF_global_context = False
        average_reward = np.mean(rewards[key])
        rewards_to_write[key] = average_reward
        logging.info(f"value: {average_reward}")

    if 'soft_budget_occupancy_measure' in use_algos:
        key = 'soft_budget_occupancy_measure'
        logging.info(f"----calculating {key}----")
        budget_solver = BudgetSolver(
            N=args.N,
            K=args.K,
            B=args.budget,
            all_transitions=all_transitions,
            context_prob=context_prob,
            w=reward_vector,
        )
        budget_solver.solve()
        upper_bound = budget_solver.get_reward()
        rewards[key] = np.ones(
            (args.n_epochs, args.n_episodes * args.episode_len + 1)
        ) * upper_bound
        average_reward = np.mean(rewards[key])
        rewards_to_write[key] = average_reward
        logging.info(f"value: {average_reward}")


    write_result(rewards, use_algos, args, all_transitions, context_prob, p, q, rewards_to_write)
    plot_rewards(rewards, use_algos, args)