# LDS_RMAB_formulation.py

"""
This module provides functions to model and analyze the Linear Dynamical System (LDS) formulation of the Restless Multi-Armed Bandit (RMAB) problem under a hard budget constraint.

Important Note:
    The functions in this module assume that the system is homogeneous, i.e., all arms have identical parameters.

Functions:
    - construct_P(N, B, p, q): Constructs the state-transition matrix P for the LDS.
    - hard_budget_value_LDS(N, K, B, context_prob, p, q): Computes the expected reward under a hard budget constraint.
    - brute_force_search_using_LDS(N, K, B, context_prob, p, q): Performs brute-force search over budget allocations to maximize the expected reward.
"""

import numpy as np
from scipy.stats import binom
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

def construct_P(N, B, p, q):
    """
    Constructs the state-transition matrix P for the LDS model under a hard budget constraint.

    Assumes a homogeneous system where all arms have identical parameters.

    Args:
        N (int): Total number of arms.
        B (int): Budget (number of arms that can be activated).
        p (float): Probability of success when an arm is activated.
        q (float): Probability of success when an arm is not activated.

    Returns:
        numpy.ndarray: State-transition matrix P of shape (N+1, N+1), where P[i, j] represents the probability of transitioning from state j to state i.
    """
    P = np.zeros((N + 1, N + 1))
    for n in range(N + 1):
        # Probability distribution of number of successes when not activating arms
        P_delta_1 = binom.pmf(np.arange(N - n + 1), N - n, q)
        # Probability distribution of number of failures when activating arms
        P_delta_2 = binom.pmf(np.arange(min(B, n) + 1), min(B, n), p)
        # Convolve the two distributions
        P_convolved = np.convolve(P_delta_1, P_delta_2[::-1])
        # Update the transition matrix
        P[n - len(P_delta_2) + 1:, n] = P_convolved
    return P

def hard_budget_value_LDS(N, K, B, context_prob, p, q):
    """
    Computes the expected reward under a hard budget constraint using the LDS model.

    Assumes a homogeneous system where all arms have identical parameters.

    Args:
        N (int): Total number of arms.
        K (int): Number of contexts (types).
        B (numpy.ndarray): Budget allocation array of shape (K,).
        context_prob (numpy.ndarray): Context probabilities array of shape (K,).
        p (numpy.ndarray): Success probabilities when activating arms, array of shape (K,).
        q (float): Success probability when not activating an arm (scalar).

    Returns:
        float: The expected reward under the given budget allocation.
    """
    P_total = np.zeros((N + 1, N + 1))
    # Aggregate the transition matrices for all contexts
    for k in range(K):
        P_k = construct_P(N, B[k], p[k], q)
        P_total += context_prob[k] * P_k
    # Compute the stationary distribution
    Sigma, U = np.linalg.eig(P_total)
    u0 = U[:, 0].real
    u0 /= sum(u0)  # Normalize to get probabilities
    # Compute the expected reward
    expected_reward = np.sum([
        context_prob[k] * p[k] * u0[n] * min(n, B[k])
        for n in range(N + 1)
        for k in range(K)
    ])
    return expected_reward


def SIMULATE_wrt_LDS(initial_state_list, N, K, budget_allocation, context_prob, p, q, T, n_episodes, n_epochs, w):
    """
    input: all necessary info to use simulate the problem
    process: convert initial_state_list to x0. it's regardless of context. only number of active arms matter
    
    Returns:
        reward: A 2D array of shape (n_epochs, T + 1) containing rewards at each timestep for each epoch.
        where T + 1 = n_episodes*T
    """
    x0_list = []
    for state in initial_state_list:
        state, context = state
        x0 = np.zeros(N + 1)
        n = int(np.sum(state%2))
        x0[n] = 1
        x0_list.append(x0)
    all_rewards = _SIMULATE_wrt_LDS(N, K, budget_allocation, context_prob, p, q, T, n_episodes, n_epochs, x0_list, w)
    return all_rewards
    
def _SIMULATE_wrt_LDS(N, K, budget_allocation, context_prob, p, q, T, n_episodes, n_epochs, x0_list, w):
    """
    psedu-simulate the homogeneous contextual-RMAB system using the LDS approach

    Args:
        T, n_episodes, n_epochs: int
        x0: initial state(s)
        N (int): Total number of arms.
        K (int): Number of contexts (types).
        B (int): budget_allocation
        context_prob (numpy.ndarray): Context probabilities array of shape (K,).
        p (numpy.ndarray): Success probabilities when activating arms, array of shape (K,).
        q (float): Success probability when not activating an arm (scalar).

    Returns:
        reward: A 2D array of shape (n_epochs, T + 1) containing rewards at each timestep for each epoch.
        where T + 1 = n_episodes*T
    """

    P_total = np.zeros((N + 1, N + 1))
    # Aggregate the transition matrices for all contexts
    for k in range(K):
        P_k = construct_P(N, budget_allocation[k], p[k], q)
        P_total += context_prob[k] * P_k

    def reward_star(x):
        """
        Args:
            x (np array of size N + 1): state distribution
            context_prob: nah
            w (np array of size K): reward vector
            budget_allocation (np array of size K): budget allocation
            p (np array of len K): probability of reward
        """
        reward = 0
        for i, x_i in enumerate(x):
            for f_k, w_k, B_k, p_k in zip(context_prob, w, budget_allocation, p):
                reward += min(i, B_k)*p_k*f_k*w_k*x_i
        return reward

    # Compute the stationary distribution
    T_total = T * n_episodes
    all_reward = np.zeros((n_epochs, T_total + 1))
    for epoch, x0 in zip(range(n_epochs), x0_list):
        xt = x0
        all_reward[epoch, 0] = reward_star(x = x0)
        for t in range(1, T_total + 1):
            xt = P_total@xt
            all_reward[epoch, t] = reward_star(x = xt)
            if t%T == 0:
                xt = x0
    return all_reward


def brute_force_search_using_LDS(N, K, B, context_prob, p, q):
    """
    Performs a brute-force search over all feasible budget allocations to find the allocation that maximizes the expected reward.

    Assumes a homogeneous system where all arms have identical parameters.

    Args:
        N (int): Total number of arms.
        K (int): Number of contexts (types).
        B (int): Total budget.
        context_prob (numpy.ndarray): Context probabilities array of shape (K,).
        p (numpy.ndarray): Success probabilities when activating arms, array of shape (K,).
        q (float): Success probability when not activating an arm (scalar).

    Returns:
        dict: A dictionary where keys are budget allocations (tuples) and values are the corresponding expected rewards.
    """
    B_UB = [B / prob for prob in context_prob]
    results = {}
    # Generate all possible budget allocations
    for budget_vector in itertools.product(*(range(int(b_ub) + 1) for b_ub in B_UB[:-1])):
        budget_vector_np = np.zeros(K, dtype=int)
        budget_vector_np[:-1] = budget_vector
        # Check if the budget allocation is feasible
        if sum(np.multiply(context_prob[:-1], budget_vector_np[:-1])) <= B:
            remaining_budget = B - sum(np.multiply(context_prob[:-1], budget_vector_np[:-1]))
            budget_vector_np[-1] = int(remaining_budget / context_prob[-1])
            if sum(np.multiply(context_prob, budget_vector_np)) <= B:
                # Compute the expected reward for this allocation
                reward = hard_budget_value_LDS(N=N, K=K, B=budget_vector_np, context_prob=context_prob, p=p, q=q)
                results[tuple(budget_vector_np)] = reward
    return results
