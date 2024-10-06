import numpy as np
import pandas as pd
import random
import time
import datetime
import sys
import os
import argparse
import json
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from volunteer_simulator import Volunteer_RMABSimulator, randomly_generate_transitions
from volunteer_algorithms import (
    whittle_policy,
    random_policy,
    whittle_policy_type_specific,
)
from instance_generator import InstanceGenerator
from brute_search_budget_allocation import brute_force_search
from result_recorder import write_result
from visualization import plot_rewards, plot_type_tuple

import pulp

# Initialize the solver
solver = pulp.getSolver('GUROBI')
pulp.LpSolverDefault.msg = 0

"""
budget_allocation_solver.py

This script solves the budget allocation problem for the Volunteer Restless Multi-Armed Bandit (RMAB) model using Linear Programming (LP) and Mixed-Integer Programming (MIP). It provides functions and classes that can be imported and used in other scripts:

Functions and Classes:
- `get_original_vectors(all_transitions, context_prob)`: Extracts the original transition probability vectors p and q from the transition matrices.
- `BudgetSolver`: A class to formulate and solve the budget allocation problem using LP or MIP.
- `solve_budget(simulator, MIP)`: Solves the budget allocation problem using LP or MIP and returns the optimal value and allocation.
- `brute_force_plot(simulator, verbose=False)`: Performs a brute-force search over all feasible budget allocations and computes the optimal rewards.

Usage:
    python budget_allocation_solver.py [options]

Arguments:
    --n_arms, -N: Number of arms (beneficiaries)
    --budget, -B: Total budget
    --num_context, -K: Number of contexts
    --episode_len, -H: Length of each episode
    --n_episodes, -T: Number of episodes to run
    --data, -D: Dataset to use ('synthetic', 'real', 'local_generated')
    --n_epochs, -E: Number of epochs (repeats) for averaging results
    --discount, -d: Discount factor for future rewards
    --homogeneous, -HOMO: Whether the instance is homogeneous ('True' or 'False')
    --seed, -s: Random seed for reproducibility
    --verbose, -V: Verbose output if set
    --local, -L: Running locally if set
    --prefix, -p: Prefix for file writing

Example:
    python budget_allocation_solver.py --n_arms 60 --budget 20 --num_context 3

The script performs the following:
- Defines a `BudgetSolver` class to formulate and solve the LP/MIP problem.
- Provides functions to extract parameters and solve the budget allocation.
- Plots the results of a brute-force search over possible budget allocations.
- Executes the main routine to perform the allocation and record results.
"""

def get_original_vectors(all_transitions, context_prob):
    """
    Extracts the original transition probability vectors p and q from the transition matrices.

    Args:
        all_transitions (numpy.ndarray): Transition probability matrices of shape (N, number_states, 2, 2).
        context_prob (list): List of probabilities for each context.

    Returns:
        tuple:
            - p (numpy.ndarray): Array of shape (N, K) where p[i, k] represents the probability of success for arm i in context k.
            - q (numpy.ndarray): Array of shape (N,) where q[i] represents the probability of success for arm i without action.
            - context_prob (list): The input list of context probabilities.
    """
    N = all_transitions.shape[0]
    K = len(context_prob)
    p = np.zeros((N, K))
    q = np.zeros(N)

    for i in range(N):
        # Compute q[i]: Probability of success without action for arm i
        q[i] = np.sum(all_transitions[i, 0, 0, 1::2])
        for k in range(K):
            # Compute p[i, k]: Probability of success with action in context k for arm i
            p[i, k] = np.sum(all_transitions[i, k*2 + 1, 1, ::2])
    return p, q, context_prob

class BudgetSolver:
    """
    A class to formulate and solve the budget allocation problem using Linear Programming (LP) or Mixed-Integer Programming (MIP).

    Attributes:
        N (int): Number of arms.
        K (int): Number of contexts.
        budget (int): Total budget.
        context_prob (list): List of probabilities for each context.
        transitions (numpy.ndarray): Transition probability matrices.
        number_states (int): Total number of states (2 * K).
        p (numpy.ndarray): Probability of success with action.
        q (numpy.ndarray): Probability of success without action.
        w (numpy.ndarray): Weight vector for each context.
        MIP (bool): Whether to use MIP (True) or LP (False).

        model (pulp.LpProblem): The LP/MIP model.
        mu (dict): Decision variables for the state-action occupancy measures.
        mu_negative (dict): Auxiliary variables for negative occupancy measures.
        B (dict): Decision variables for budget allocation.

        IF_solved (bool): Flag indicating if the model has been solved.
    """

    def __init__(self, N, K, B, all_transitions, context_prob, w=None, MIP=True):
        """
        Initializes the BudgetSolver with the given parameters.
        Set up variables.
        Set up (i) probability and (ii) budget constriants.

        Args:
            N (int): Number of arms.
            K (int): Number of contexts.
            B (int): Total budget.
            all_transitions (numpy.ndarray): Transition probability matrices.
            context_prob (list): List of probabilities for each context.
            w (numpy.ndarray, optional): Weight vector for each context. Defaults to ones if None.
            MIP (bool, optional): Whether to use MIP. Defaults to True.
        """
        self.N = N
        self.K = K
        self.budget = B
        self.context_prob = context_prob
        self.transitions = all_transitions
        self.number_states = 2 * K  # Each context has two states

        # Extract original vectors p and q
        self.p, self.q, _ = get_original_vectors(all_transitions, context_prob)
        self.MIP = MIP

        # Initialize weight vector w
        self.w = np.ones(self.K) if w is None else w

        # Initialize the LP/MIP model
        self.model = pulp.LpProblem("Budget Solver", pulp.LpMaximize)

        # Decision variables for occupancy measures
        self.mu = pulp.LpVariable.dicts(
            "mu", [(i, j) for i in range(N) for j in range(self.number_states)], lowBound=0
        )
        self.mu_negative = pulp.LpVariable.dicts(
            "mu_negative", [(i, j) for i in range(N) for j in range(self.number_states)], lowBound=0
        )

        # Decision variables for budget allocation
        if self.MIP:
            self.B = pulp.LpVariable.dicts("B", range(self.K), lowBound=0, cat=pulp.LpInteger)
        else:
            self.B = pulp.LpVariable.dicts("B", range(self.K), lowBound=0)

        # Set up the LP/MIP problem
        self.set_objective()
        self.set_budget_constraints()
        self.set_probability_constraints()
        self.set_transition_constraints()

        self.IF_solved = False  # Indicates whether the problem has been solved

    def set_objective(self):
        """
        Sets the objective function of the LP/MIP problem to maximize the expected reward.
        """
        # Objective: Maximize the expected reward over all arms and contexts
        self.model += pulp.lpSum(
            [
                self.p[i][k] * self.w[k] * self.mu[(i, k * 2 + 1)]
                for k in range(self.K)
                for i in range(self.N)
            ]
        ), "Objective"

    def set_transition_constraints(self):
        """
        Sets the transition constraints ensuring that the occupancy measures are consistent with the transition probabilities.
        """
        for i in range(self.N):
            for j in range(self.number_states):
                # Constraint: mu_i,j = sum over all possible previous states and actions
                self.model += (
                    pulp.lpSum(
                        [
                            self.transitions[i, l, 1, j] * self.mu[(i, l)]
                            for l in range(self.number_states)
                        ]
                    )
                    + pulp.lpSum(
                        [
                            self.transitions[i, l, 0, j] * (self.mu_negative[(i, l)])
                            for l in range(self.number_states)
                        ]
                    )
                    == self.mu[(i, j)] + self.mu_negative[(i, j)],
                    f"Transition constraints for arm {i}, state {j}",
                )

    def set_probability_constraints(self):
        """
        Sets the probability constraints ensuring that the sum of occupancy measures equals 1 for each arm.
        """
        for i in range(self.N):
            # Constraint: Sum of occupancy measures equals 1
            self.model += (
                pulp.lpSum([self.mu[(i, j)] for j in range(self.number_states)])
                + pulp.lpSum([self.mu_negative[(i, j)] for j in range(self.number_states)])
                == 1,
                f"Probability constraint for arm {i}",
            )

    def set_budget_constraints(self):
        """
        Sets the budget constraints for each context and the total budget.
        """
        # Budget constraints for each context
        for k in range(self.K):
            self.model += (
                pulp.lpSum([self.mu[(i, 2 * k)] for i in range(self.N)])
                + pulp.lpSum([self.mu[(i, 2 * k + 1)] for i in range(self.N)])
                <= self.B[k] * self.context_prob[k],
                f"Budget constraint for context {k}",
            )

        # Total budget constraint
        self.model += (
            pulp.lpSum([self.B[k] * self.context_prob[k] for k in range(self.K)]) <= self.budget,
            "Total budget constraint",
        )
        self.budget_constraints = []

    def set_fix_budgets(self, budgets=None):
        """
        Fixes the budget allocation to specified values by modifying the LP/MIP constraints.

        Args:
            budgets (numpy.ndarray): Array of budget allocations for each context.
        """
        # Remove existing budget constraints
        for constraint in self.budget_constraints:
            self.model.constraints.pop(constraint.name)

        # Clear the list of budget constraints
        self.budget_constraints.clear()

        # Add new budget constraints to fix budgets
        for k in range(self.K):
            constraint = self.B[k] == budgets[k]
            constraint_name = f"Fixing_B_{k}"
            self.model += constraint, constraint_name
            self.budget_constraints.append(self.model.constraints[constraint_name])

        self.IF_solved = False  # Reset the solved flag

    def update_variable_bounds(self, fixed_vars, bounds):
        """
        Update the variable bounds for the LP model before solving.

        This method applies variable fixings and bounds to the LP variables based on
        the dictionaries provided. It modifies the variables in the LP model in place.

        Parameters:
        -----------
        fixed_vars : dict
            A dictionary where keys are variable names (strings) and values are the fixed
            values to assign to those variables. Variables are expected to be named like 'B0', 'B1', etc.
            Example:
                fixed_vars = {'B0': 3, 'B2': 5}

        bounds : dict
            A dictionary where keys are variable names (strings) and values are tuples
            representing the (lower bound, upper bound) for the variables.
            Variables are expected to be named like 'B0', 'B1', etc.
            Example:
                bounds = {'B1': (2, 4), 'B3': (None, 6)}

            - If the lower bound is None, it defaults to 0.
            - If the upper bound is None, it means there is no upper bound.

        Returns:
        --------
        None

        Logic:
        ------
        - For each variable in 'fixed_vars':
            - Fix the variable by setting its lower and upper bounds to the fixed value.
        - For each variable in 'bounds':
            - Update the variable's lower and upper bounds according to the provided values.

        Notes:
        ------
        - It is assumed that variable names correspond to variables in self.B and follow
          the naming convention 'B0', 'B1', ..., 'B{K-1}' where K is the number of types.
        - This method should be called before solving the LP model to apply the branching constraints.
        """
        # Update variable bounds before solving

        # Apply variable fixings
        for var_name, value in fixed_vars.items():
            # Extract the index k from the variable name 'B_k'
            index_k = int(var_name[2:])  # Assumes var_name is like 'B_0', 'B_1', etc.
            var = self.B[index_k]
            # Fix the variable by setting both its lower and upper bounds to the fixed value
            var.lowBound = value
            var.upBound = value

        # Apply variable bounds
        for var_name, (lb, ub) in bounds.items():
            # Extract the index k from the variable name 'B_k'
            index_k = int(var_name[2:])  # Assumes var_name is like 'B_0', 'B_1', etc.
            var = self.B[index_k]
            # Set the lower bound; if lb is None, default to 0
            var.lowBound = lb if lb is not None else 0
            # Set the upper bound; if ub is None, it means there is no upper bound
            var.upBound = ub if ub is not None else None
        
        self.IF_solved = False  # Reset the solved flag

    def reset_variable_bounds(self):
        """
        Reset the variable bounds to their default values after solving.

        This method resets the lower and upper bounds of all 'B' variables in the LP model
        back to their original defaults (lower bound = 0, upper bound = None).

        Returns:
        --------
        None

        Logic:
        ------
        - Iterate over all 'B' variables in self.B.
        - Set each variable's lower bound to 0.
        - Set each variable's upper bound to None (no upper bound).

        Notes:
        ------
        - This method should be called after solving the LP model to clean up any temporary
          variable bounds applied during branching in the branch-and-bound algorithm.
        """
        # Reset variable bounds after solving
        for var in self.B.values():
            var.lowBound = 0        # Reset lower bound to default (0)
            var.upBound = None      # Reset upper bound to default (no upper bound)
        self.IF_solved = False  # Reset the solved flag

    def set_fairness_lowerbound(self, lowerbound=0):
        """
        Sets constraints ensuring that the reward obtained for each context is no less than the specified lower bound.

        Args:
            lowerbound (float): The minimum acceptable reward for each context.
        """
        # Initialize or clear the list of fairness constraints
        self.fairness_constraints = []
        for k in range(self.K):
            # Define the constraint expression
            constraint_expr = pulp.lpSum(
                [
                    self.p[i][k] * self.w[k] * self.mu[(i, k * 2 + 1)]
                    for i in range(self.N)
                ]
            ) >= lowerbound*self.context_prob[k]
            # Name the constraint
            constraint_name = f"Fairness_constraint_{k}"
            # Add the constraint to the model
            self.model += constraint_expr, constraint_name
            # Keep track of the constraint name for future removal
            self.fairness_constraints.append(constraint_name)
        self.IF_solved = False  # Reset the solved flag

    def reset_fairness_lowerbound(self):
        """
        Removes the fairness constraints from the model.
        """
        # Remove the fairness constraints from the model
        for constraint_name in self.fairness_constraints:
            if constraint_name in self.model.constraints:
                del self.model.constraints[constraint_name]
        # Clear the list of fairness constraints
        self.fairness_constraints = []
        self.IF_solved = False  # Reset the solved flag

    def get_budget_solution(self):
        """
        Retrieve the current solution of the LP model.

        Returns:
        --------
        solution : dict
            A dictionary where keys are variable names (strings) and values are the variable
            values from the current LP solution.
            Example:
                solution = {'B0': 3.0, 'B1': 2.0, 'mu_(0,1)': 0.5, ...}

        Logic:
        ------
        - Iterate over all variables in the LP model.
        - For each variable, store its name and value in the solution dictionary.

        Notes:
        ------
        - This method should be called after the LP model has been solved.
        - The solution includes all variables in the model, not just the 'B' variables.
        """
        # Return the current solution as a dictionary
        solution = {}
        for var in self.model.variables():
            if "B" in var.name:
                solution[var.name] = var.varValue
        return solution
    
    def get_totalN_rewards(self):

        self.mu_np = np.array(
            [[pulp.value(self.mu[(i, j)]) for j in range(self.number_states)] for i in range(self.N)]
        )
        
        self.mu_negative_np = np.array(
            [
                [pulp.value(self.mu_negative[(i, j)]) for j in range(self.number_states)]
                for i in range(self.N)
            ]
        )
        theoretical_reward = np.sum(
            [
                np.sum(self.mu_np[i, k * 2 + 1]) * self.p[i][k] * self.w[k]
                for k in range(self.K)
                for i in range(self.N)
            ]
        )
        return theoretical_reward

    def solve(self):
        """
        Solves the LP/MIP problem.

        Sets the IF_solved flag to True if the problem is solved optimally.
        """
        self.model.solve()
        if pulp.LpStatus[self.model.status] == 'Optimal':
            self.IF_solved = True
        else:
            print(f"Solver Status: {pulp.LpStatus[self.model.status]}")
            self.IF_solved = False

    def get_budget_allocation(self):
        """
        Retrieves the optimal budget allocation after solving the LP/MIP problem.

        Returns:
            numpy.ndarray or None: The budget allocation array if solved, else None.
        """
        if self.IF_solved:
            self.budget_allocation = np.zeros(self.K)
            for k in range(self.K):
                self.budget_allocation[k] = pulp.value(self.B[k])
            return self.budget_allocation
        else:
            print("Warning: Problem not solved optimally.")
            return None

    def report_result(self):
        """
        Reports the results by printing the budget allocation and theoretical reward.

        Also computes and stores the occupancy measures in numpy arrays.
        """
        # Extract occupancy measures
        self.mu_np = np.array(
            [[pulp.value(self.mu[(i, j)]) for j in range(self.number_states)] for i in range(self.N)]
        )
        self.mu_negative_np = np.array(
            [
                [pulp.value(self.mu_negative[(i, j)]) for j in range(self.number_states)]
                for i in range(self.N)
            ]
        )
        # Print budget allocation and used budget
        print(f"Budget allocated: {self.get_budget_allocation().round(2)}")
        used_budget = np.round([np.sum(self.mu_np[:, k * 2 + 1]) for k in range(self.K)], 2)
        print(f"Budget used: {used_budget}")
        # Calculate and print theoretical reward
        theoretical_reward = np.sum(
            [
                np.sum(self.mu_np[i, k * 2 + 1]) * self.p[i][k]
                for k in range(self.K)
                for i in range(self.N)
            ]
        )
        print(f"Theoretical expected reward: {theoretical_reward}")

    def get_reward(self):
        """
        Reports the results by returning **theoretical reward**.

        Also computes and stores the occupancy measures in numpy arrays.
        """
        return pulp.value(self.model.objective)
        # Extract occupancy measures
        self.mu_np = np.array(
            [[pulp.value(self.mu[(i, j)]) for j in range(self.number_states)] for i in range(self.N)]
        )
        self.mu_negative_np = np.array(
            [
                [pulp.value(self.mu_negative[(i, j)]) for j in range(self.number_states)]
                for i in range(self.N)
            ]
        )
        theoretical_reward = np.sum(
            [
                np.sum(self.mu_np[i, k * 2 + 1]) * self.p[i][k]
                for k in range(self.K)
                for i in range(self.N)
            ]
        )
        return theoretical_reward
    
    def get_variables(self):
        """
        return:
            mu (numpy)
            mu_negative (numpy), 
            budget_allocation vectors (numpy) 
        """
        mu_np = np.array(
            [[pulp.value(self.mu[(i, j)]) for j in range(self.number_states)] for i in range(self.N)]
        )
        mu_negative_np = np.array(
            [
                [pulp.value(self.mu_negative[(i, j)]) for j in range(self.number_states)]
                for i in range(self.N)
            ]
        )

        return mu_np, mu_negative_np, self.get_budget_allocation()


def solve_budget(simulator, MIP):
    """
    Solves the budget allocation problem using LP or MIP and returns the optimal value and allocation.

    Args:
        simulator (Volunteer_RMABSimulator): The RMAB simulator instance.
        MIP (bool): Whether to use MIP (True) or LP (False).

    Returns:
        tuple:
            - opt_value (float): The optimal objective value.
            - best_allocation (numpy.ndarray): The optimal budget allocation.
    """
    # Extract parameters from the simulator
    p, q, context_prob = simulator.get_original_vectors()

    # Initialize the BudgetSolver
    BudgetLPSolver = BudgetSolver(
        N=simulator.N,
        K=simulator.K,
        B=simulator.budget,
        all_transitions=simulator.all_transitions,
        context_prob=context_prob,
        MIP=MIP,
    )
    # Solve the LP/MIP problem
    BudgetLPSolver.solve()
    # Retrieve the optimal value and budget allocation
    opt_value = pulp.value(BudgetLPSolver.model.objective)
    best_allocation = BudgetLPSolver.get_budget_allocation()
    return opt_value, best_allocation

def brute_force_plot(simulator, verbose=False):
    """
    NAME_FUNCTION DISJUNCTURE:
    Use LP/MIP.
    Performs a brute-force search over all feasible budget allocations and computes the optimal rewards.

    Args:
        simulator (Volunteer_RMABSimulator): The RMAB simulator instance.
        verbose (bool, optional): If True, prints detailed information. Defaults to False.

    Returns:
        dict: A dictionary where keys are budget allocations (tuples) and values are the corresponding optimal rewards.
    """
    # Extract parameters
    p, q, context_prob = simulator.get_original_vectors()
    if verbose:
        print("Parameters (p, q, context_prob):\n", p[0], q[0], context_prob)

    # Initialize the BudgetSolver without MIP for efficiency
    BudgetLPSolver = BudgetSolver(
        N=simulator.N,
        K=simulator.K,
        B=simulator.budget,
        all_transitions=simulator.all_transitions,
        context_prob=context_prob,
        MIP=False,
    )

    B = simulator.budget
    K = simulator.K
    # Calculate upper bounds for budgets in each context
    B_UB = [B / prob for prob in context_prob]

    results = {}

    # Iterate over all possible budget allocations except for the last context
    for budget_vector in itertools.product(*(range(int(b_ub) + 1) for b_ub in B_UB[:-1])):
        # Initialize budget vector for all contexts
        budget_vector_np = np.zeros(K, dtype=int)
        budget_vector_np[:-1] = budget_vector

        # Check if the current budget allocation is feasible
        if sum(np.multiply(context_prob[:-1], budget_vector_np[:-1])) <= B:
            # Calculate remaining budget for the last context
            remaining_budget = B - sum(np.multiply(context_prob[:-1], budget_vector_np[:-1]))
            budget_vector_np[-1] = int(remaining_budget / context_prob[-1])

            # Verify the total budget allocation does not exceed the budget
            if sum(np.multiply(context_prob, budget_vector_np)) <= B:
                # Fix the budgets in the solver and solve
                BudgetLPSolver.set_fix_budgets(budget_vector_np)
                BudgetLPSolver.solve()
                opt_value = pulp.value(BudgetLPSolver.model.objective)
                if verbose:
                    print(f"Budget Allocation: {budget_vector_np}, Optimal Reward: {opt_value}")
                # Store the results
                results[tuple(budget_vector_np)] = opt_value
    return results

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_arms', '-N', help='Number of beneficiaries (arms)', type=int, default=60)
    parser.add_argument('--budget', '-B', help='Total budget', type=int, default=20)
    parser.add_argument('--num_context', '-K', help='Number of contexts', type=int, default=2)
    parser.add_argument('--episode_len', '-H', help='Episode length', type=int, default=357)
    parser.add_argument('--n_episodes', '-T', help='Number of episodes', type=int, default=6)
    parser.add_argument('--data', '-D', help='Dataset to use {synthetic, real, local_generated}', type=str, default='local_generated')
    parser.add_argument('--n_epochs', '-E', help='Number of epochs (repeats)', type=int, default=6)
    parser.add_argument('--discount', '-d', help='Discount factor', type=float, default=0.98)
    parser.add_argument('--alpha', '-a', help='Alpha: for confidence radius', type=float, default=3)
    parser.add_argument('--n_actions', '-A', help='Number of actions', type=int, default=2)
    parser.add_argument('--homogeneous', '-HOMO', help='Whether instance is homogeneous', type=str, default='True')
    parser.add_argument('--seed', '-s', help='Random seed', type=int, default=42)
    parser.add_argument('--verbose', '-V', type=bool, help='Verbose output if set', default=False)
    parser.add_argument('--local', '-L', help='Running locally if set', action='store_true')
    parser.add_argument('--prefix', '-p', help='Prefix for file writing', type=str, default='')

    args = parser.parse_args()
    args.str_time = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    args.exp_name_out = "LP_solved_budget_allocation"
    args.homogeneous = args.homogeneous.lower() in ['true', '1', 'yes']

    # Initialize context probabilities and transitions
    if args.data == 'local_generated':
        print('Using locally generated data with a fixed seed.')
        generator = InstanceGenerator(N=args.n_arms, K=args.num_context, seed=args.seed)
        num_instances = 1
        print(f'Homogeneous: {args.homogeneous}')
        print(f'Seed: {args.seed}')
        generator.generate_instances(num_instances, homogeneous=args.homogeneous)
        instance = generator.load_instance()
        all_transitions, context_prob = instance['transitions'], instance['context_prob']
        reward_vector = np.ones(args.num_context)

    # Initialize the simulator
    simulator = Volunteer_RMABSimulator(
        N=args.n_arms,
        K=args.num_context,
        T=args.episode_len,
        context_prob=context_prob,
        all_transitions=all_transitions,
        budget=args.budget,
        reward_vector=reward_vector,
    )

    # Perform brute-force search and plot results
    rewards = brute_force_plot(simulator)

    # Plot the results
    plot_type_tuple(rewards, context_prob=context_prob, args=args)

    reward_algo = {}
    # Solve budget allocation using LP
    opt_value_lp, best_allocation_lp = solve_budget(simulator, MIP=False)
    print(f"--> LP Solved Budget Allocation: {best_allocation_lp}\nOptimal Value: {opt_value_lp}")
    reward_algo['LP_Solving_Budget'] = opt_value_lp
    reward_algo['LP OPT_Budget Allocation'] = best_allocation_lp.tolist()

    # Solve budget allocation using MIP
    opt_value_mip, best_allocation_mip = solve_budget(simulator, MIP=True)
    print(f"--> MIP Solved Budget Allocation: {best_allocation_mip}\nOptimal Value: {opt_value_mip}")
    reward_algo['MIP_Solving_Budget'] = opt_value_mip
    reward_algo['MIP OPT_Budget Allocation'] = best_allocation_mip.tolist()

    # Extract p and q vectors
    p, q, _ = simulator.get_original_vectors()
    converted_rewards = {str(k): v for k, v in rewards.items()}

    # Write the results to a file
    write_result(
        rewards=reward_algo,
        use_algos=["LP_Solving_Budget", "MIP_Solving_Budget"],
        args=args,
        transition_probabilities=all_transitions,
        context_prob=context_prob,
        p=p,
        q=q,
        rewards_to_write=converted_rewards,
        best_allocation=best_allocation_lp,
    )
