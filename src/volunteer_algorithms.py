# volunteer_algorithm.py

"""
Volunteer Algorithm Module

This module implements various algorithms for the Volunteer Restless Multi-Armed Bandit (RMAB) problem, including:

- `whittle_policy`: Implements the Whittle index policy for RMAB.
- `get_type_specific_budget_naive`: Computes a naive type-specific budget allocation based on Whittle indices.
- `whittle_policy_type_specific`: Implements the Whittle index policy with type-specific budget constraints.
- `random_policy`: Implements a random action policy for comparison.

These functions can be used to simulate policies on an RMAB environment, typically instantiated using the `Volunteer_RMABSimulator`.

Usage:
    - Import the module and use the functions with an appropriate RMAB environment.

Example:
    from volunteer_algorithm import whittle_policy, random_policy
    rewards = whittle_policy(env, n_episodes=1, n_epochs=6, discount=0.99)
"""

import numpy as np
import random
import heapq

import logging

from volunteer_simulator import Volunteer_RMABSimulator, randomly_generate_transitions
from volunteer_compute_whittle import arm_compute_whittle, arm_compute_whittle_all_states
from utils import Memoizer

def whittle_policy(env, n_episodes=1, n_epochs=6, discount=0.99):
    """
    Implements the Whittle index policy for the RMAB problem.

    Args:
        env (Volunteer_RMABSimulator): The RMAB environment.
        n_episodes (int, optional): Number of episodes to run. Defaults to 1.
        n_epochs (int, optional): Number of epochs for averaging results. Defaults to 6.
        discount (float, optional): Discount factor for future rewards. Defaults to 0.99.

    Returns:
        numpy.ndarray: A 2D array of shape (n_epochs, T + 1) containing rewards at each timestep for each epoch.
    """
    N         = env.N
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.T * n_episodes

    env.reset_all()

    memoizer = Memoizer('optimal')

    all_reward = np.zeros((n_epochs, T + 1))

    for epoch in range(n_epochs):
        if epoch != 0:
            env.reset_instance()
        true_transitions = env.all_transitions

        # Record initial reward
        all_reward[epoch, 0] = env.get_reward_external()

        for t in range(1, T + 1):
            state = env.observe()

            # Compute Whittle index for each arm
            state_WI = np.zeros(N)
            top_WI = []
            min_chosen_subsidy = -1
            for i in range(N):
                arm_transitions = true_transitions[i, :, :, :]

                # Memoization to speed up computation
                check_set_val = memoizer.check_set(arm_transitions, state[i])
                if check_set_val != -1:
                    state_WI[i] = check_set_val
                else:
                    state_WI[i] = arm_compute_whittle(
                        transitions=arm_transitions,
                        state=state[i],
                        reward_vector=env.reward_vector,
                        discount=discount,
                        subsidy_break=-10
                    )
                    memoizer.add_set(arm_transitions, state[i], state_WI[i])

                # Maintain a heap of top Whittle indices
                if len(top_WI) < budget:
                    heapq.heappush(top_WI, (state_WI[i], i))
                else:
                    heapq.heappushpop(top_WI, (state_WI[i], i))
                    min_chosen_subsidy = top_WI[0][0]  # Smallest Whittle index in the heap

            # Select arms with highest Whittle indices up to the budget
            sorted_WI = np.argsort(state_WI)[::-1]
            action = np.zeros(N, dtype=np.int8)
            action[sorted_WI[:budget]] = 1

            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)

            if done and t < T:
                env.reset()

            all_reward[epoch, t] = reward

    return all_reward

def get_type_specific_budget_naive(env):
    """
    Computes a naive type-specific budget allocation based on average Whittle indices.

    Args:
        env (Volunteer_RMABSimulator): The RMAB environment.

    Returns:
        numpy.ndarray: An array of size K containing the budget allocation for each context type.
    """
    # Compute Whittle indices for all arms and states
    whittle_index_all = np.zeros((env.N, 2 * env.K))
    for i in range(env.N):
        whittle_index_all[i] = arm_compute_whittle_all_states(
            transitions=env.all_transitions[i],
            reward_vector=env.reward_vector
        )
    # Average Whittle indices over arms for each context (considering active states)
    whittle_index_avg = np.average(whittle_index_all[:, 1::2], axis=0)
    print("Average Whittle index over arms:", whittle_index_avg)

    # Compute budget allocation proportional to average Whittle indices
    delta = env.budget / np.sum(env.context_prob * whittle_index_avg)
    budget_allocation = np.array(whittle_index_avg * delta, dtype=int)
    print(f"Budget allocation = {np.round(budget_allocation)}")
    print(f"Checking average budget: sum = {np.sum(budget_allocation * env.context_prob)}")
    return budget_allocation

def whittle_policy_type_specific(env, type_specific_budget, n_episodes=1, n_epochs=6, discount=0.99):
    """ 
    Implements the Whittle index policy with type-specific budget constraints.

    Args:
        env (Volunteer_RMABSimulator): The RMAB environment.
        type_specific_budget (numpy.ndarray): An array of size K specifying the budget for each context type.
        n_episodes (int, optional): Number of episodes to run. Defaults to 1.
        n_epochs (int, optional): Number of epochs for averaging results. Defaults to 6.
        discount (float, optional): Discount factor for future rewards. Defaults to 0.99.

    Returns:
        numpy.ndarray: A 2D array of shape (n_epochs, T + 1) containing rewards at each timestep for each epoch.
    """
    env.constraint_type = "soft"
    N         = env.N
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.T * n_episodes

    env.reset_all()

    memoizer = Memoizer('optimal')

    all_reward = np.zeros((n_epochs, T + 1))

    for epoch in range(n_epochs):
        if epoch != 0:
            env.reset_instance()
        true_transitions = env.all_transitions

        # Record initial reward
        all_reward[epoch, 0] = env.get_reward_external()

        for t in range(1, T + 1):
            state = env.observe()

            # Compute Whittle index for each arm
            state_WI = np.zeros(N)
            for i in range(N):
                arm_transitions = true_transitions[i, :, :, :]

                # Memoization to speed up computation
                check_set_val = memoizer.check_set(arm_transitions, state[i])
                if check_set_val != -1:
                    state_WI[i] = check_set_val
                else:
                    state_WI[i] = arm_compute_whittle(
                        transitions=arm_transitions,
                        state=state[i],
                        reward_vector=env.reward_vector,
                        discount=discount,
                        subsidy_break=-10
                    )
                    memoizer.add_set(arm_transitions, state[i], state_WI[i])

            # Select arms with highest Whittle indices up to the type-specific budget
            sorted_WI = np.argsort(state_WI)[::-1]
            # logging.debug(f"Whittle Index: {state_WI}")
            # logging.debug(f"states: {state}")
            # logging.debug(f"sorted_WI: {sorted_WI}")
            action = np.zeros(N, dtype=np.int8)
            # Allocate actions based on type-specific budget
            # logging.debug(f"budget to use now: {type_specific_budget[env.context]}")
            action[sorted_WI[:type_specific_budget[env.context]]] = 1
            logging.debug(f"number of arms pulled at time {t}: {np.sum(action)}")
            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)
            logging.debug(f"step_wise reward at time {t}: {reward}")

            if done and t < T:
                env.reset()

            all_reward[epoch, t] = reward

        # Optionally, print used budget per epoch
        # print(f"Epoch {epoch} used budget = {env.report_avg_budget()}")
    return all_reward

def random_policy(env, n_episodes, n_epochs):
    """
    Implements a random action policy for the RMAB problem.

    Args:
        env (Volunteer_RMABSimulator): The RMAB environment.
        n_episodes (int): Number of episodes to run.
        n_epochs (int): Number of epochs for averaging results.

    Returns:
        numpy.ndarray: A 2D array of shape (n_epochs, T + 1) containing rewards at each timestep for each epoch.
    """
    N         = env.N
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.T * n_episodes

    env.reset_all()

    all_reward = np.zeros((n_epochs, T + 1))

    for epoch in range(n_epochs):
        if epoch != 0:
            env.reset_instance()
        # Record initial reward
        all_reward[epoch, 0] = env.get_reward_external()
        for t in range(1, T + 1):
            state = env.observe()

            # Randomly select arms up to the budget
            selected_idx = np.random.choice(N, size=budget, replace=False)
            action = np.zeros(N, dtype=np.int8)
            action[selected_idx] = 1

            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)

            if done and t < T:
                env.reset()

            all_reward[epoch, t] = reward

    return all_reward

if __name__ == '__main__':
    # Example usage of the module
    N = 20
    K = 3
    T = 100
    budget = 10
    reward_vector = np.ones(K)
    # Generate random transitions and context probabilities
    all_transitions, context_prob = randomly_generate_transitions(N, K, homogeneous=False)
    # Initialize the RMAB simulator
    simulator = Volunteer_RMABSimulator(
        N=N,
        K=K,
        T=T,
        context_prob=context_prob,
        all_transitions=all_transitions,
        budget=budget,
        reward_vector=reward_vector
    )
    # Run Whittle policy
    rewards = whittle_policy(simulator)
    print("Rewards obtained from Whittle policy:", rewards)


# import numpy as np
# import random
# import heapq

# from volunteer_simulator import Volunteer_RMABSimulator, randomly_generate_transitions
# from volunteer_compute_whittle import arm_compute_whittle, arm_compute_whittle_all_states
# from utils import Memoizer

# def whittle_policy(env, n_episodes=1, n_epochs=6, discount=0.99):
#     """
#     whittle policy 
#     return reward where reward.shape = ((n_epochs, env.T * n_episodes + 1))
#     """
#     N         = env.N
#     n_states  = env.number_states
#     n_actions = env.all_transitions.shape[2]
#     budget    = env.budget
#     T         = env.T * n_episodes

#     env.reset_all()

#     memoizer = Memoizer('optimal')

#     all_reward = np.zeros((n_epochs, T + 1))

#     for epoch in range(n_epochs):
#         if epoch != 0: env.reset_instance()
#         true_transitions = env.all_transitions

#         # print('first state', env.observe())
#         all_reward[epoch, 0] = env.get_reward_external()

#         for t in range(1, T + 1):
#             state = env.observe()

#             # select optimal action based on known transition probabilities
#             # compute whittle index for each arm
#             state_WI = np.zeros(N)
#             top_WI = []
#             min_chosen_subsidy = -1 #0
#             for i in range(N):
#                 arm_transitions = true_transitions[i, :, :, :]

#                 # memoize to speed up
#                 check_set_val = memoizer.check_set(arm_transitions, state[i])
#                 if check_set_val != -1:
#                     state_WI[i] = check_set_val
#                 else:
#                     state_WI[i] = arm_compute_whittle(transitions=arm_transitions, state=state[i], reward_vector=env.reward_vector, discount=discount, subsidy_break=-10)
#                     memoizer.add_set(arm_transitions, state[i], state_WI[i])

#                 if len(top_WI) < budget:
#                     heapq.heappush(top_WI, (state_WI[i], i))
#                 else:
#                     # add state_WI to heap
#                     heapq.heappushpop(top_WI, (state_WI[i], i))
#                     min_chosen_subsidy = top_WI[0][0]  # smallest-valued item

#             # pull K highest indices
#             sorted_WI = np.argsort(state_WI)[::-1]
#             # print(f'   state {state} state_WI {np.round(state_WI, 2)} sorted {np.round(sorted_WI[:budget], 2)}')

#             action = np.zeros(N, dtype=np.int8)
#             action[sorted_WI[:budget]] = 1

#             next_state, reward, done, _ = env.step(action)

#             if done and t < T: env.reset()

#             all_reward[epoch, t] = reward

#     return all_reward

# def get_type_specific_budget_naive(env):
#     """
#     naive approach: compute whittle index and assign budget proportional to Whittle Index
#     """
#     whittle_index_all = np.zeros((env.N, 2*env.K))
#     for i in range(env.N):
#         whittle_index_all[i] = arm_compute_whittle_all_states(
#             transitions=env.all_transitions[i], 
#             reward_vector=env.reward_vector
#             )
#     whittle_index_avg = np.average(whittle_index_all[:, 1::2], axis=0)
#     # print("`whittle_index_avg` shape check: ", whittle_index_avg.shape)
#     print("average-over-arms whittle index:", whittle_index_avg)

#     """
#     assume Whittle index (or, whatever) weight w[k],
#     context_prob f[k]

#     sum_k f[k]B[k] = B
#     B[k] = w[k]*delta, for all k
#     <=> B[k] = w[k]*delta where
#         delta = B/(sum_k f[k]w[k])

#     """
#     delta = env.budget/np.sum(env.context_prob*whittle_index_avg)
#     budget_allocation = np.array(whittle_index_avg*delta, dtype = int)
#     print(f"budget_allocation = {np.round(budget_allocation)}")
#     print(f"checking average budget: sum = {np.sum(budget_allocation*env.context_prob)}")
#     return budget_allocation

# def whittle_policy_type_specific(env, type_specific_budget, n_episodes=1, n_epochs=6, discount=0.99):
#     """ 
#     whittle index policy with type-specific constraints 
#     input: type_specific_budget (of size K)
#     similarly:
#     return reward where reward.shape = ((n_epochs, env.T * n_episodes + 1))
#     """
#     env.constraint_type = "soft"
#     N         = env.N
#     n_states  = env.number_states
#     n_actions = env.all_transitions.shape[2]
#     budget    = env.budget
#     T         = env.T * n_episodes

#     env.reset_all()

#     memoizer = Memoizer('optimal')

#     all_reward = np.zeros((n_epochs, T + 1))

#     for epoch in range(n_epochs):
#         if epoch != 0: env.reset_instance()
#         true_transitions = env.all_transitions

#         # print('first state', env.observe())
#         all_reward[epoch, 0] = env.get_reward_external()

#         for t in range(1, T + 1):
#             state = env.observe()

#             # select optimal action based on known transition probabilities
#             # compute whittle index for each arm
#             state_WI = np.zeros(N)
#             # top_WI = []
#             # min_chosen_subsidy = -1 #0
#             for i in range(N):
#                 arm_transitions = true_transitions[i, :, :, :]

#                 # memoize to speed up
#                 check_set_val = memoizer.check_set(arm_transitions, state[i])
#                 if check_set_val != -1:
#                     state_WI[i] = check_set_val
#                 else:
#                     state_WI[i] = arm_compute_whittle(transitions=arm_transitions, state=state[i], reward_vector=env.reward_vector, discount=discount, subsidy_break=-10)
#                     memoizer.add_set(arm_transitions, state[i], state_WI[i])

#                 # if len(top_WI) < type_specific_budget[env.context]:
#                 #     heapq.heappush(top_WI, (state_WI[i], i))
#                 # else:
#                 #     # add state_WI to heap
#                 #     heapq.heappushpop(top_WI, (state_WI[i], i))
#                 #     min_chosen_subsidy = top_WI[0][0]  # smallest-valued item

#             # pull K highest indices
#             sorted_WI = np.argsort(state_WI)[::-1]
#             # print(f'   state {state} state_WI {np.round(state_WI, 2)} sorted {np.round(sorted_WI[:budget], 2)}')
#             action = np.zeros(N, dtype=np.int8)
#             # seems like only change is here
#             action[sorted_WI[:type_specific_budget[env.context]]] = 1

#             next_state, reward, done, _ = env.step(action)

#             if done and t < T: env.reset()

#             all_reward[epoch, t] = reward

#         # print(f"epoch {epoch} used budget = {env.report_avg_budget()}")
#     return all_reward


# def random_policy(env, n_episodes, n_epochs):
#     """ random action each timestep """
#     N         = env.N
#     n_states  = env.number_states
#     n_actions = env.all_transitions.shape[2]
#     budget    = env.budget
#     T         = env.T * n_episodes

#     env.reset_all()

#     all_reward = np.zeros((n_epochs, T + 1))

#     for epoch in range(n_epochs):
#         if epoch != 0: env.reset_instance()
#         # print('first state', env.observe())
#         all_reward[epoch, 0] = env.get_reward_external()
#         for t in range(1, T + 1):
#             state = env.observe()

#             # select arms at random
#             selected_idx = np.random.choice(N, size=budget, replace=False)
#             action = np.zeros(N, dtype=np.int8)
#             action[selected_idx] = 1

#             next_state, reward, done, _ = env.step(action)

#             if done and t < T: env.reset()

#             all_reward[epoch, t] = reward

#     return all_reward


# if __name__ == '__main__':
#     N  = 20
#     K = 3
#     T = 100
#     budget = 10
#     reward_vector = np.ones(K)
#     all_transitions, context_prob = randomly_generate_transitions(N, K, homogeneous = False)
#     simulator = Volunteer_RMABSimulator(N = N, 
#                                     K = K, 
#                                     T = T, 
#                                     context_prob=context_prob, 
#                                     all_transitions=all_transitions, 
#                                     budget=budget, 
#                                     reward_vector=reward_vector
#                                     )

