import numpy as np
import pandas as pd
import random
import time, datetime
import sys, os
import argparse
import json
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_theme()

from volunteer_simulator import Volunteer_RMABSimulator, randomly_generate_transitions
from volunteer_algorithms import whittle_policy , random_policy, whittle_policy_type_specific #, WIQL
from instance_generator import InstanceGenerator
from brute_search_budget_allocation import brute_force_search
from result_recorder import write_result
from visualization import plot_rewards, plot_type_tuple

import pulp
solver = pulp.getSolver('GUROBI')
pulp.LpSolverDefault.msg = 0

def get_original_vectors(all_transitions, context_prob):
    """
    input: all prob. matrices
    """
    N = all_transitions.shape[0]
    K = len(context_prob)
    p = np.zeros((N, K))
    q = np.zeros(N)

    for i in range(N):
        # crucial change made here
        # previously 
        # q[i] = np.sum(all_transitions[i, 0, :, 1::2])
        q[i] = np.sum(all_transitions[i, 0, 0, 1::2])
        for k in range(K):
            p[i, k] = np.sum(all_transitions[i, k*2 + 1, 1, ::2])
    return p, q, context_prob

class BudgetSolver:
    def __init__(self, N, K, B, all_transitions, context_prob, w = None, MIP=True):
        
        self.N = N
        self.K = K
        self.budget = B
        self.context_prob = context_prob
        self.transitions = all_transitions
        self.number_states   = 2*K
        self.p, self.q, _ = get_original_vectors(all_transitions, context_prob)
        self.MIP = MIP

        # w is the K-dim vector of weights for each type
        if w == None:
            self.w = np.ones(self.K)
        else:
            self.w = w

        # Initialize the LP model
        self.model = pulp.LpProblem("Budget Solver", pulp.LpMaximize)

        # self.mu.shape = (number_states,), which indicates
        # mu(s, a = 1) for s in [number_states]
        # as corresponding to transitions
        self.mu = pulp.LpVariable.dicts("mu", [(i, j) for i in range(N) for j in range(self.number_states)], lowBound=0)
        self.mu_negative = pulp.LpVariable.dicts("mu_negative", [(i, j) for i in range(N) for j in range(self.number_states)], lowBound=0)
        if self.MIP:
            self.B = pulp.LpVariable.dicts("B", range(self.budget), lowBound=0, cat=pulp.LpInteger)
        else:
            self.B = pulp.LpVariable.dicts("B", range(self.budget), lowBound=0)

        self.set_objective()
        self.set_budget_constraints()
        self.set_probability_constraints()
        self.set_transition_constraints()

        self.IF_solved = False

    def set_objective(self):
        
        # Objective function
        self.model += pulp.lpSum([self.p[i][k]*self.w[k]*self.mu[(i, k*2 + 1)] for k in range(self.K) for i in range(self.N)]), "Objective"        

    def set_transition_constraints(self):
        for i in range(self.N):
            for j in range(self.number_states):
                self.model += pulp.lpSum([self.transitions[i, l, 1, j]*self.mu[(i, l)] for l in range(self.number_states)]) + pulp.lpSum([self.transitions[i, l, 0, j]*(self.mu_negative[(i, l)]) for l in range(self.number_states)]) == self.mu[(i, j)] + self.mu_negative[(i, j)], f"transition constraints: mu_({i, j}) = sum_(a, l) P[...]mu_(a, l)"

    def set_probability_constraints(self):
        for i in range(self.N):
            self.model += pulp.lpSum([self.mu[(i, j)] for j in range(self.number_states)]) + pulp.lpSum([self.mu_negative[(i, j)] for j in range(self.number_states)]) == 1

    def set_budget_constraints(self):
        for k in range(self.K):
            self.model += pulp.lpSum([self.mu[(i, 2*k)] for i in range(self.N)]) + pulp.lpSum([self.mu[(i, 2*k + 1)] for i in range(self.N)]) <= self.B[k]*self.context_prob[k], f"budget constraint for type {k}"

        self.model += pulp.lpSum([self.B[k]*self.context_prob[k] for k in range(self.K)]) <= self.budget
        self.budget_constraints = []

    def set_fix_budgets(self, budgets=None):
        # remove constraints
        for constraint in self.budget_constraints:
            self.model.constraints.pop(constraint.name)

        # Clear the list of budget constraints
        self.budget_constraints.clear()

        # Add new budget constraints
        for k in range(self.K):
            constraint = self.B[k] == budgets[k]
            constraint_name = f"fixing_B_{k}"
            self.model += constraint, constraint_name
            self.budget_constraints.append(self.model.constraints[constraint_name])

        self.IF_solved = False
        

    def solve(self):
        self.model.solve()
        # print(f"solving status {pulp.LpStatus[self.model.status]}")
        # print(f"Optimal value for Budget Allocation: {pulp.value(self.model.objective)}")
        if pulp.LpStatus[self.model.status] == pulp.LpStatus[1]:
            self.IF_solved = True

    def get_budget_allocation(self):
        if self.IF_solved == True:
            self.budget_allocation = np.zeros(self.K)
            for k in range(self.K):
                self.budget_allocation[k] = pulp.value(self.B[k])
            # print("B allocation:", self.budget_allocation.round(2))
            return self.budget_allocation
        else:
            print("warning! problem not solved, self.IF_solved is False")
            return None
        
    def report_result(self):
        self.mu_np = np.array([
            [pulp.value(self.mu[(i, j)]) for j in range(self.number_states)] for i in range(self.N)
        ])
        self.mu_negative_np = np.array([
            [pulp.value(self.mu_negative[(i, j)]) for j in range(self.number_states)] for i in range(self.N)
        ])
        # for i in range(self.N):
            # sns.lineplot(self.mu_np[i, 1::2])
        # plt.show()
        print(f"budget allocated: {self.get_budget_allocation().round(2)}")
        print(f"budget used: {np.round([np.sum(self.mu_np[:, k*2 + 1]) for k in range(self.K)], 2)}")
        print(f"theoretical result: {np.sum([np.sum(self.mu_np[i, k*2 + 1])*self.p[i][k] for k in range(self.K) for i in range(self.N)])}")


def solve_budget(simulator, MIP):
    p, q, context_prob = simulator.get_original_vectors()
    # print("parameters(p, q, f)\n", p[0], q[0], context_prob)

    # simple solver
    # SimpleLPSolver = SimpleLP(N = simulator.N, K = simulator.K, p = p, w = simulator.reward_vector, f=context_prob, B = simulator.budget, q = q)
    # SimpleLPSolver.solve()
    # SimpleLPSolver.check_result()

    # new solver
    BudgetLPSolver = BudgetSolver(N=simulator.N, K=simulator.K, B = simulator.budget, all_transitions=simulator.all_transitions, context_prob=context_prob, MIP = MIP)
    BudgetLPSolver.solve()
    # BudgetLPSolver.report_result()
    return pulp.value(BudgetLPSolver.model.objective), BudgetLPSolver.get_budget_allocation()

def brute_force_plot(simulator, verbose=False):
    """
    input simulator
    solve using MIP to obtain the results dict w.r.t. budgets
    """
    p, q, context_prob = simulator.get_original_vectors()
    if verbose:
        print("parameters(p, q, f)\n", p[0], q[0], context_prob)
    
    BudgetLPSolver = BudgetSolver(N=simulator.N, K=simulator.K, B = simulator.budget, all_transitions=simulator.all_transitions, context_prob=context_prob, MIP = False)

    B = simulator.budget
    K = simulator.K
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
                BudgetLPSolver.set_fix_budgets(budget_vector_np)
                BudgetLPSolver.solve()
                if verbose:
                    print(f"B = {budget_vector_np}, reward = {pulp.value(BudgetLPSolver.model.objective)}")
                results[tuple(budget_vector_np)] = pulp.value(BudgetLPSolver.model.objective)
        # print(f"\nreward = {np.mean(rewards)}\nbudget = {budget_vector_np}")
    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_arms',         '-N', help='num beneficiaries (arms)', type=int, default=60)
    parser.add_argument('--budget',         '-B', help='budget', type=int, default=20)
    parser.add_argument('--num_context',    '-K', help='context_size', type=int, default='2')

    parser.add_argument('--episode_len',    '-H', help='episode length', type=int, default=357)
    parser.add_argument('--n_episodes',     '-T', help='num episodes', type=int, default=6)
    parser.add_argument('--data',           '-D', help='dataset to use {synthetic, real, local_generated}', type=str, default='local_generated')

    parser.add_argument('--n_epochs',       '-E', help='number of epochs (num_repeats)', type=int, default=6)
    parser.add_argument('--discount',       '-d', help='discount factor', type=float, default=0.98)

    # doesn't seem necessary
    parser.add_argument('--alpha',          '-a', help='alpha: for conf radius', type=float, default=3)

    # parser.add_argument('--n_states',       '-S', help='num states', type=int, default=2)
    parser.add_argument('--n_actions',      '-A', help='num actions', type=int, default=2)

    # parser.add_argument('--homogeneous',      '-HOMO', help='if homogenous', type=bool, default=True)
    parser.add_argument('--homogeneous', '-HOMO', help='if homogenous', type=str, default='True')

    parser.add_argument('--seed',           '-s', help='random seed', type=int, default=42)
    parser.add_argument('--verbose',        '-V', type=bool, help='if True, then verbose output (default False)', default=False)
    parser.add_argument('--local',          '-L', help='if True, running locally (default False)', action='store_true')
    parser.add_argument('--prefix',         '-p', help='prefix for file writing', type=str, default='')


    args = parser.parse_args()
    args.str_time = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    args.exp_name_out = "LP_solved_budget_allocation"
    args.homogeneous = args.homogeneous.lower() in ['true', '1', 'yes']

    if args.data == 'local_generated':
        print('using locally generated data w.r.t. a seed that fixing everything')
        # generate local data using a same seed—if N, K be the same, generated probability transitions should also be the same
        generator = InstanceGenerator(N=args.n_arms, K=args.num_context, seed=args.seed)
        num_instances = 1
        print(f'homogeneous {args.homogeneous}')
        print(f'seed is {args.seed}')
        generator.generate_instances(num_instances, homogeneous=args.homogeneous)
        instance = generator.load_instance()
        all_transitions, context_prob = instance['transitions'], instance['context_prob']
        reward_vector = np.ones(args.num_context)

    simulator = Volunteer_RMABSimulator(N = args.n_arms, 
                                    K = args.num_context, 
                                    T = args.episode_len, 
                                    context_prob=context_prob, 
                                    all_transitions=all_transitions, 
                                    budget=args.budget, 
                                    reward_vector=reward_vector
                                    )
    
    rewards = brute_force_plot(simulator)
    
    plot_type_tuple(rewards, context_prob=context_prob, args=args)

    reward_algo = dict()
    # solving LP
    opt_value, best_allocation = solve_budget(simulator, MIP=False)
    print(f"--> LP  directly solved budget allocation = {best_allocation}\n opt_val = {opt_value}")
    # rewards[tuple(best_allocation.tolist())] = opt_value
    reward_algo['LP_Solving_Budget'] = opt_value
    reward_algo['LP OPT_Budget Allocation'] = best_allocation.tolist()

    # solving MIP
    opt_value, best_allocation = solve_budget(simulator, MIP=True)
    print(f"--> MIP directly solved budget allocation = {best_allocation}\n opt_val = {opt_value}")
    # rewards[tuple(best_allocation.tolist())] = opt_value
    reward_algo['MIP_Solving_Budget'] = opt_value
    reward_algo['MIP OPT_Budget Allocation'] = best_allocation.tolist()


    p, q, _ = simulator.get_original_vectors()
    converted_rewards = {str(k): v for k, v in rewards.items()}

    write_result(
        rewards=reward_algo, 
        use_algos=["LP_Solving_Budget", "MIP_Solving_Budget"], 
        args=args, 
        transition_probabilities=all_transitions, 
        context_prob=context_prob, 
        p = p, 
        q = q, 
        rewards_to_write=converted_rewards, 
        best_allocation=best_allocation)