"""
standard Whittle index computation based on binary search

POSSIBLE OPTIMIZATIONS TO HELP SPEED
- keep track of top k WIs so far. then in future during binary search, if we go below that WI just quit immediately
"""

import sys
import numpy as np

import heapq  # priority queue

from volunteer_simulator import Volunteer_RMABSimulator, randomly_generate_transitions

whittle_threshold = 1e-4
value_iteration_threshold = 1e-3

def arm_value_iteration(transitions, state, lamb_val, discount, reward_vector, threshold=value_iteration_threshold):
    """ value iteration for a single arm at a time

    value iteration for the MDP defined by transitions with lambda-adjusted reward function
    return action corresponding to pi^*(s_I) 
    """
    assert discount < 1

    n_states, n_actions, _ = transitions.shape
    value_func = np.random.rand(n_states)
    difference = np.ones((n_states))
    iters = 0

    # lambda-adjusted reward function
    def reward(s, a, s_prime):
        if s%2 == 1 and s_prime%2 == 0 and a == 1:
            return 1 - a * lamb_val
        else:
            return -a*lamb_val

    while np.max(difference) >= threshold:
        iters += 1
        orig_value_func = np.copy(value_func)

        # calculate Q-function
        Q_func = np.zeros((n_states, n_actions))
        for s in range(n_states):
            for a in range(n_actions):
                for s_prime in range(n_states):
                    Q_func[s, a] += transitions[s, a, s_prime]*(reward(s, a, s_prime) + discount*value_func[s_prime])

                # # transitioning to state = 0
                # Q_func[s, a] += (1 - transitions[s, a]) * (reward(s, a) + discount * value_func[0])

                # # transitioning to state = 1
                # Q_func[s, a] += transitions[s, a] * (reward(s, a) + discount * value_func[1])

            value_func[s] = np.max(Q_func[s, :])

        difference = np.abs(orig_value_func - value_func)

    # print(f'q values {Q_func[state, :]}, action {np.argmax(Q_func[state, :])}')
    # print(f"iterations: {iters}")
    return np.argmax(Q_func[state, :])


def get_init_bounds(transitions):
    lb = -1
    ub = 1
    return lb, ub


def arm_compute_whittle(transitions, state, reward_vector, discount, subsidy_break, eps=whittle_threshold):
    """
    compute whittle index for a single arm using binary search

    subsidy_break = the min value at which we stop iterating

    param transitions:
    param eps: epsilon convergence
    returns Whittle index
    """
    lb, ub = get_init_bounds(transitions) # return lower and upper bounds on WI
    top_WI = []
    while abs(ub - lb) > eps:
        lamb_val = (lb + ub) / 2
        # print('lamb', lamb_val, lb, ub)

        # we've already filled our knapsack with higher-valued WIs
        if ub < subsidy_break:
            # print('breaking early!', subsidy_break, lb, ub)
            return -10

        action = arm_value_iteration(transitions, state, lamb_val, discount, reward_vector)
        assert action == 0 or action == 1, "action is not binary"
        if action == 0:
            # optimal action is passive: subsidy is too high
            ub = lamb_val
        elif action == 1:
            # optimal action is active: subsidy is too low
            lb = lamb_val
    subsidy = (ub + lb) / 2
    return subsidy

def arm_compute_whittle_all_states(transitions, reward_vector, discount = 0.99, subsidy_break = -20, eps=whittle_threshold):
    n_states, n_action, _ = transitions.shape
    W_index = np.zeros(n_states)
    for state in range(n_states):
        W_index[state] = arm_compute_whittle(transitions, state, reward_vector=reward_vector, discount=discount, subsidy_break=subsidy_break)
    return W_index

'''
Testing the functionality of the simulator
'''
if __name__ == '__main__':
    N  = 100
    K = 3
    reward_vector = np.ones(K)
    all_transitions, context_prob = randomly_generate_transitions(N, K, homogeneous = True)
    W_index = np.zeros(K*2)
    for state in range(K*2):
        W_index[state] = arm_compute_whittle(all_transitions[0], state, reward_vector=reward_vector, discount=0.99, subsidy_break=-20)
    print("whittle index computed", W_index.round(3))
    