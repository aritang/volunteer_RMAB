import numpy as np
from scipy.stats import binom
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

def construct_P(N, B, p, q):
    """
    purpose: construct state-transition matrix P (N+1*N+1) that can be used to analyze stationary probability distribution of the system
    input: N, B integers, p, q real
    return: P of (N, N) where P_ij = Prob[n = j -> n = i | pull min(B, n) arms]
    """
    P = np.zeros((N + 1, N + 1))
    for n in range(N + 1):
        P_delta_1 = binom.pmf(np.arange(N - n + 1), N - n, q)
        P_delta_2 = binom.pmf(np.arange(min(B, n) + 1), min(B, n), p)
        P_convolved = np.convolve(P_delta_1, P_delta_2[::-1])
        # print(len(P_convolved))
        P[n - len(P_delta_2) + 1:, n] = P_convolved

    return P

def hard_budget_value_LDS(N, K, B, context_prob, p, q):
    """
    budget_allocation context_prob, p are all (K)-shaped np array
    """
    P = np.zeros((N + 1, N + 1))
    for k in range(K):
        P += context_prob[k]*construct_P(N, B[k], p[k], q)
    Sigma, U = np.linalg.eig(P)
    u0 = U[:, 0].real
    u0 /= sum(u0)
    return np.sum([context_prob[k]*p[k]*u0[n]*min(n, B[k]) for n in range(N + 1) for k in range(K)])

def brute_force_search_using_LDS(N, K, B, context_prob, p, q):
    B_UB = [B / prob for prob in context_prob]
    results = {}

    for budget_vector in itertools.product(*(range(int(b_ub)+1) for b_ub in B_UB[:-1])):
        
        budget_vector_np = np.zeros(K, dtype=int)
        budget_vector_np[:-1] = budget_vector
        if sum(np.multiply(context_prob[:-1], budget_vector_np[:-1])) <= B:
            remaining_budget = B - sum(np.multiply(context_prob[:-1], budget_vector_np[:-1]))
            budget_vector_np[-1] = int(remaining_budget / context_prob[-1])
            if sum(np.multiply(context_prob, budget_vector_np)) <= B:
                # run is here
                rewards = hard_budget_value_LDS(N = N, K = K, B = budget_vector_np, context_prob=context_prob, p = p, q = q)
                # print(f"B = {budget_vector_np}, reward = {np.mean(rewards)}")
                results[tuple(budget_vector_np)] = np.mean(rewards)
        # print(f"\nreward = {np.mean(rewards)}\nbudget = {budget_vector_np}")
    return results