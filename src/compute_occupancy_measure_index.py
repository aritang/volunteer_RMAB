"""
compute occupancy-measure based index computation based on the method by Xiong et al

basic function:
input: a contextual RMAB instance
output: index for all arms, context and state
"""

import numpy as np
from bugdet_allocation_solver import BudgetSolver

def compute_occupancy_measure_index(N, K, budget, all_transitions, context_prob, w = None, eps = 1e-5):

    _, n_states, n_actions, __ = all_transitions.shape

    BudgetLPSolver = BudgetSolver(
        N=N,
        K=K,
        B=budget,
        all_transitions=all_transitions,
        context_prob=context_prob,
        MIP=False,
        w=w
    )
    BudgetLPSolver.solve()
    mu_positive, mu_negative, opt_budget_allocation = BudgetLPSolver.get_variables()
    p = BudgetLPSolver.p
    
    # obtain results
    if w is None:
        w = np.ones(K)

    reward_vector = np.zeros((N, n_states))
    for i in range (N):
        for k in range(K):
            reward_vector[i][k * 2 + 1] = p[i][k]*w[k]

    OM_index = np.zeros((N, n_states))
    for i in range(N):
        for n in range(n_states):
            mu_all = mu_positive[i][n] + mu_negative[i][n]
            if mu_all < eps:
                OM_index[i][n] = 0
            else:
                OM_index[i][n] = reward_vector[i][n]*mu_positive[i][n]/mu_all
    
    return OM_index

'''

Testing OM_index by comparing it with Whittle Index
'''

if __name__ == '__main__':
    from volunteer_compute_whittle import arm_compute_whittle
    from instance_generator import InstanceGenerator
    import matplotlib.pyplot as plt
    import seaborn as sns

    N = 25
    K = 3
    budget = 5
    generator = InstanceGenerator(N=25, K=3, seed=43)
    generator.generate_instances(homogeneous=False)

    instance = generator.load_instance()
    all_transitions, context_prob = instance['transitions'], instance['context_prob']

    # default: num_state = K*2
    W_index = np.zeros((N, K*2))
    for i in range(N):
        for state in range(K*2):
            W_index[i][state] = arm_compute_whittle(all_transitions[i], state, reward_vector=np.ones(K), discount=0.99, subsidy_break=-20)

    OM_index = compute_occupancy_measure_index(
        N = N,
        K = K,
        budget=budget,
        all_transitions=all_transitions,
        context_prob=context_prob,
        w = np.ones(K)
    )

    # Heatmap plot for W_index and OM_index side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    sns.heatmap(W_index, ax=axes[0], cmap="coolwarm", annot=False)
    axes[0].set_title("Whittle Index (W_index)")

    sns.heatmap(OM_index, ax=axes[1], cmap="coolwarm", annot=False)
    axes[1].set_title("Occupancy Measure Index (OM_index)")

    plt.tight_layout()
    plt.show()

    # Optional: You can also print rounded versions of W_index and OM_index for comparison
    # print("Whittle Index (W_index):", W_index.round(3))
    # print("Occupancy Measure Index (OM_index):", OM_index.round(3))