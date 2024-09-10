import numpy as np
import random
import heapq

from volunteer_simulator import Volunteer_RMABSimulator, randomly_generate_transitions
from volunteer_compute_whittle import arm_compute_whittle, arm_compute_whittle_all_states
from utils import Memoizer

def whittle_policy(env, n_episodes=1, n_epochs=6, discount=0.99):
    """
    whittle policy 
    return reward where reward.shape = ((n_epochs, env.T * n_episodes + 1))
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
        if epoch != 0: env.reset_instance()
        true_transitions = env.all_transitions

        # print('first state', env.observe())
        all_reward[epoch, 0] = env.get_reward_external()

        for t in range(1, T + 1):
            state = env.observe()

            # select optimal action based on known transition probabilities
            # compute whittle index for each arm
            state_WI = np.zeros(N)
            top_WI = []
            min_chosen_subsidy = -1 #0
            for i in range(N):
                arm_transitions = true_transitions[i, :, :, :]

                # memoize to speed up
                check_set_val = memoizer.check_set(arm_transitions, state[i])
                if check_set_val != -1:
                    state_WI[i] = check_set_val
                else:
                    state_WI[i] = arm_compute_whittle(transitions=arm_transitions, state=state[i], reward_vector=env.reward_vector, discount=discount, subsidy_break=-10)
                    memoizer.add_set(arm_transitions, state[i], state_WI[i])

                if len(top_WI) < budget:
                    heapq.heappush(top_WI, (state_WI[i], i))
                else:
                    # add state_WI to heap
                    heapq.heappushpop(top_WI, (state_WI[i], i))
                    min_chosen_subsidy = top_WI[0][0]  # smallest-valued item

            # pull K highest indices
            sorted_WI = np.argsort(state_WI)[::-1]
            # print(f'   state {state} state_WI {np.round(state_WI, 2)} sorted {np.round(sorted_WI[:budget], 2)}')

            action = np.zeros(N, dtype=np.int8)
            action[sorted_WI[:budget]] = 1

            next_state, reward, done, _ = env.step(action)

            if done and t < T: env.reset()

            all_reward[epoch, t] = reward

    return all_reward

def get_type_specific_budget_naive(env):
    """
    naive approach: compute whittle index and assign budget proportional to Whittle Index
    """
    whittle_index_all = np.zeros((env.N, 2*env.K))
    for i in range(env.N):
        whittle_index_all[i] = arm_compute_whittle_all_states(
            transitions=env.all_transitions[i], 
            reward_vector=env.reward_vector
            )
    whittle_index_avg = np.average(whittle_index_all[:, 1::2], axis=0)
    # print("`whittle_index_avg` shape check: ", whittle_index_avg.shape)
    print("average-over-arms whittle index:", whittle_index_avg)

    """
    assume Whittle index (or, whatever) weight w[k],
    context_prob f[k]

    sum_k f[k]B[k] = B
    B[k] = w[k]*delta, for all k
    <=> B[k] = w[k]*delta where
        delta = B/(sum_k f[k]w[k])

    """
    delta = env.budget/np.sum(env.context_prob*whittle_index_avg)
    budget_allocation = np.array(whittle_index_avg*delta, dtype = int)
    print(f"budget_allocation = {np.round(budget_allocation)}")
    print(f"checking average budget: sum = {np.sum(budget_allocation*env.context_prob)}")
    return budget_allocation

def whittle_policy_type_specific(env, type_specific_budget, n_episodes=1, n_epochs=6, discount=0.99):
    """ 
    whittle index policy with type-specific constraints 
    input: type_specific_budget (of size K)
    similarly:
    return reward where reward.shape = ((n_epochs, env.T * n_episodes + 1))
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
        if epoch != 0: env.reset_instance()
        true_transitions = env.all_transitions

        # print('first state', env.observe())
        all_reward[epoch, 0] = env.get_reward_external()

        for t in range(1, T + 1):
            state = env.observe()

            # select optimal action based on known transition probabilities
            # compute whittle index for each arm
            state_WI = np.zeros(N)
            # top_WI = []
            # min_chosen_subsidy = -1 #0
            for i in range(N):
                arm_transitions = true_transitions[i, :, :, :]

                # memoize to speed up
                check_set_val = memoizer.check_set(arm_transitions, state[i])
                if check_set_val != -1:
                    state_WI[i] = check_set_val
                else:
                    state_WI[i] = arm_compute_whittle(transitions=arm_transitions, state=state[i], reward_vector=env.reward_vector, discount=discount, subsidy_break=-10)
                    memoizer.add_set(arm_transitions, state[i], state_WI[i])

                # if len(top_WI) < type_specific_budget[env.context]:
                #     heapq.heappush(top_WI, (state_WI[i], i))
                # else:
                #     # add state_WI to heap
                #     heapq.heappushpop(top_WI, (state_WI[i], i))
                #     min_chosen_subsidy = top_WI[0][0]  # smallest-valued item

            # pull K highest indices
            sorted_WI = np.argsort(state_WI)[::-1]
            # print(f'   state {state} state_WI {np.round(state_WI, 2)} sorted {np.round(sorted_WI[:budget], 2)}')
            action = np.zeros(N, dtype=np.int8)
            # seems like only change is here
            action[sorted_WI[:type_specific_budget[env.context]]] = 1

            next_state, reward, done, _ = env.step(action)

            if done and t < T: env.reset()

            all_reward[epoch, t] = reward

        # print(f"epoch {epoch} used budget = {env.report_avg_budget()}")
    return all_reward


def random_policy(env, n_episodes, n_epochs):
    """ random action each timestep """
    N         = env.N
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.T * n_episodes

    env.reset_all()

    all_reward = np.zeros((n_epochs, T + 1))

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()
        # print('first state', env.observe())
        all_reward[epoch, 0] = env.get_reward_external()
        for t in range(1, T + 1):
            state = env.observe()

            # select arms at random
            selected_idx = np.random.choice(N, size=budget, replace=False)
            action = np.zeros(N, dtype=np.int8)
            action[selected_idx] = 1

            next_state, reward, done, _ = env.step(action)

            if done and t < T: env.reset()

            all_reward[epoch, t] = reward

    return all_reward


# def WIQL(env, n_episodes, n_epochs):
#     """ Whittle index-based Q-Learning
#     [Biswas et al. 2021]

#     input: N, budget, alpha(c), initial states
#     """
#     N         = env.cohort_size
#     n_states  = env.number_states
#     n_actions = env.all_transitions.shape[2]
#     budget    = env.budget
#     T         = env.episode_len * n_episodes

#     env.reset_all()

#     all_reward = np.zeros((n_epochs, T + 1))

#     def alpha_func(c):
#         """ learning parameter
#         alpha = 0: agent doesn't learn anything new
#         alpha = 1: stores only most recent information; overwrites previous records """
#         assert 0 <= c <= 1
#         return c

#     for epoch in range(n_epochs):
#         if epoch != 0: env.reset_instance()

#         # initialize
#         Q_vals    = np.zeros((N, n_states, n_actions))
#         lamb_vals = np.zeros((N, n_states))

#         # print('first state', env.observe())
#         all_reward[epoch, 0] = env.get_reward()
#         for t in range(1, T + 1):
#             state = env.observe()

#             # select M arms using epsilon-decay policy
#             epsilon = N / (N + t)

#             # with probability epsilon, select B arms uniformly at random
#             if np.random.binomial(1, epsilon):
#                 selected_idx = np.random.choice(N, size=budget, replace=False)
#             else:
#                 state_lamb_vals = np.array([lamb_vals[i, state[i]] for i in range(N)])
#                 # select top arms according to their lambda values
#                 selected_idx = np.argpartition(state_lamb_vals, -budget)[-budget:]
#                 selected_idx = selected_idx[np.argsort(state_lamb_vals[selected_idx])][::-1] # sort indices

#             action = np.zeros(N, dtype=np.int8)
#             action[selected_idx] = 1

#             # take suitable actions on arms
#             # execute chosen policy; observe reward and next state
#             next_state, reward, done, _ = env.step(action)
#             if done and t < T: env.reset()

#             # update Q, lambda
#             c = .5 # None
#             for i in range(N):
#                 for s in range(n_states):
#                     for a in range(n_actions):
#                         prev_Q = Q_vals[i, s, a]
#                         state_i = next_state[i]
#                         prev_max_Q = np.max(Q_vals[i, state_i, :])

#                         alpha = alpha_func(c)

#                         Q_vals[i, s, a] = (1 - alpha) * prev_Q + alpha * (reward + prev_max_Q)

#                     lamb_vals[i, s] = Q_vals[i, s, 1] - Q_vals[i, s, 0]

#             all_reward[epoch, t] = reward

#     return all_reward

if __name__ == '__main__':
    N  = 20
    K = 3
    T = 100
    budget = 10
    reward_vector = np.ones(K)
    all_transitions, context_prob = randomly_generate_transitions(N, K, homogeneous = False)
    simulator = Volunteer_RMABSimulator(N = N, 
                                    K = K, 
                                    T = T, 
                                    context_prob=context_prob, 
                                    all_transitions=all_transitions, 
                                    budget=budget, 
                                    reward_vector=reward_vector
                                    )

