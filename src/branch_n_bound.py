from dataclasses import dataclass, field
import numpy as np
import heapq
import logging
# Configure logging level and format
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
# You can adjust the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) depending on the amount of detail you want.
import pulp
from typing import List

import networkx as nx
import matplotlib.pyplot as plt

solver = pulp.getSolver('GUROBI')
pulp.LpSolverDefault.msg = 0

from volunteer_algorithms import whittle_policy_type_specific
from bugdet_allocation_solver import BudgetSolver
from instance_generator import initialize_instance_and_simulator
from utils import parse_arguments
from result_recorder import write_result
from bugdet_allocation_solver import brute_force_plot


@dataclass(order=True)
class Node:
    priority: float
    fixed_vars: dict = field(default_factory=dict, compare=False)
    bounds: dict = field(default_factory=dict, compare=False)
    lp_value: float = field(default=0, compare=False)
    # node.solution = {'B_0': 28.0, 'B_1': 0.0, 'B_2': 0.0}
    solution: dict = field(default_factory=dict, compare=False)

    depth: int = field(default=0, compare=False)
    oracle_value: float = field(default=None, compare=False)
    parent: 'Node' = field(default=None, compare=False, repr=False)
    children: List['Node'] = field(default_factory=list, compare=False, repr=False)
    node_id: int = field(default_factory=lambda: Node.next_id(), compare=False)

    pruned: bool = field(default=False, compare=False)  # the pruned flag, for visualization

    _id_counter = 0

    @classmethod
    def next_id(cls):
        cls._id_counter += 1
        return cls._id_counter

    def __post_init__(self):
        # Ensure that priority is set correctly (negative for max-heap behavior)
        self.priority = -self.lp_value

class Oracle:
    """
    almost a wrapper. 
    simulate on real environment
    """
    def __init__(self, args, simulator):
        """
        Initialize the Oracle with the necessary parameters.

        Parameters:
        -----------
        """
        self.args = args
        self.env = simulator
    
    def evaluate(self, budget_allocation, n_epochs= None, n_episodes = None):
        """
        Compute the oracle reward R^S(B) for a given budget allocation.

        Parameters:
        -----------
        budget_allocation : list or array-like
            A vector representing the budget allocated to each type/context.

        Returns:
        --------
        oracle_reward : float
            The total reward computed by the oracle function.
        """

        if n_epochs== None:
            n_epochs= self.args.n_epochs
        
        if n_episodes == None:
            n_episodes = self.args.n_episodes

        # Run the policy and obtain the total reward
        all_reward = whittle_policy_type_specific(
            self.env,
            type_specific_budget=budget_allocation,
            n_episodes=self.args.n_episodes,
            n_epochs=self.args.n_epochs,
            discount=self.args.discount
        )
        mean_reward = np.mean(all_reward)
        logging.info(f"oracle for B_vector: {budget_allocation}, value: {mean_reward}")


        return mean_reward


def hierarchy_pos_overlapping(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    Positions nodes in a hierarchical layout.

    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        The graph to layout.
    root : node, optional
        The root of the tree. If None, a root will be determined automatically.
    width : float, optional
        Total horizontal space allocated for drawing.
    vert_gap : float, optional
        Vertical gap between levels of the tree.
    vert_loc : float, optional
        Vertical location of root.
    xcenter : float, optional
        Horizontal location of root.

    Returns:
    --------
    pos : dict
        A dictionary mapping nodes to positions.
    """
    import networkx as nx

    def _hierarchy_pos(G, node, left, right, current_level, level_positions, parent=None):
        # Initialize the list for the current level if it doesn't exist
        if current_level not in level_positions:
            level_positions[current_level] = []

        if node not in level_positions[current_level]:
            level_positions[current_level].append(node)
        else:
            # Node already placed at this level
            return

        # Compute the horizontal position
        num_nodes_in_level = len(level_positions[current_level])
        horizontal_spacing = (right - left) / num_nodes_in_level if num_nodes_in_level > 0 else 0
        x = left + (num_nodes_in_level - 1) * horizontal_spacing + horizontal_spacing / 2

        pos[node] = (x, vert_loc - current_level * vert_gap)

        children = list(G.successors(node))
        if children:
            child_left = left
            child_right = right
            child_spacing = (child_right - child_left) / len(children)
            for i, child in enumerate(children):
                child_left_bound = child_left + i * child_spacing
                child_right_bound = child_left + (i + 1) * child_spacing
                _hierarchy_pos(G, child, child_left_bound, child_right_bound, current_level + 1, level_positions, node)

    if not nx.is_tree(G):
        raise TypeError('Cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        root = next(iter(nx.topological_sort(G)))  # Try to find a root node

    pos = {}
    level_positions = {}
    current_level = 0
    level_positions[current_level] = []
    _hierarchy_pos(G, root, 0.0, width, current_level, level_positions)
    return pos

def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, scale=1.0):
    import networkx as nx

    def _hierarchy_pos(G, node, left, right, current_level, level_positions, parent=None):
        # Initialize the list for the current level if it doesn't exist
        if current_level not in level_positions:
            level_positions[current_level] = []

        if node not in level_positions[current_level]:
            level_positions[current_level].append(node)
        else:
            # Node already placed at this level
            return

        index_in_level = level_positions[current_level].index(node)
        num_nodes_in_level = len(level_positions[current_level])
        horizontal_spacing = (right - left) / max(num_nodes_in_level - 1, 1)
        x = left + index_in_level * horizontal_spacing

        # Apply scaling to positions
        x_scaled = x * scale
        y_scaled = (vert_loc - current_level * vert_gap) * scale

        pos[node] = (x_scaled, y_scaled)

        children = list(G.successors(node))
        if children:
            child_left = left + index_in_level * horizontal_spacing
            child_right = left + (index_in_level + 1) * horizontal_spacing
            child_width = (child_right - child_left)
            for i, child in enumerate(children):
                _hierarchy_pos(G, child, child_left, child_right, current_level + 1, level_positions, node)

    if not nx.is_tree(G):
        raise TypeError('Cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        root = next(iter(nx.topological_sort(G)))  # Try to find a root node

    pos = {}
    level_positions = {}
    current_level = 0
    level_positions[current_level] = []
    _hierarchy_pos(G, root, 0.0, width, current_level, level_positions)
    return pos


def parse_bounds(bounds):
    """
    Parses the bounds dictionary into a readable string.
    
    Parameters:
    -----------
    bounds : dict
        A dictionary of variable bounds.
    
    Returns:
    --------
    bounds_str : str
        A string representation of the bounds.
    """
    if not bounds:
        return "Bounds: None"
    bound_strings = []
    for var, (lb, ub) in bounds.items():
        lb_str = f"{lb}" if lb is not None else "-∞"
        ub_str = f"{ub}" if ub is not None else "∞"
        bound_strings.append(f"{lb_str} ≤ {var} ≤ {ub_str}")
    return "Bounds:\n" + "\n".join(bound_strings)

class BranchAndBoundSolver:
    def __init__(self, budget_solver, oracle):
        self.budget_solver = budget_solver
        self.oracle = oracle  # Function to compute R_oracle(B)
        self.best_feasible_solution = None
        self.best_lower_bound = float('-inf')
        self.node_queue = []
        self.all_nodes = []
        self.node_counter = 0  # For debugging or logging purposes
        self.visited_states = set() # Set of state keys for visited nodes

        # Initialize logging
        logging.info("Initialized BranchAndBoundSolver")

    def node_state_key(self, node):
        """
        Generates a unique key representing the state of the node based on its bounds and fixed variables.

        Parameters:
        -----------
        node : Node
            The node whose state key is to be generated.

        Returns:
        --------
        state_key : tuple
            A tuple that uniquely represents the node's state.
        """
        # Sort the bounds and fixed_vars items to ensure consistent ordering
        bounds_items = tuple(sorted(node.bounds.items()))
        fixed_vars_items = tuple(sorted(node.fixed_vars.items()))
        
        # Create the state key as a tuple of bounds and fixed variables
        state_key = (bounds_items, fixed_vars_items)
        return state_key

    def solve(self):
        # Initialize root node
        root_node = Node(priority=0, fixed_vars={}, bounds={}, lp_value=0, depth=0)
        self.add_node_to_queue(root_node)
        logging.info("Starting Branch and Bound algorithm")

        while self.node_queue:
            logging.info("------- thank u, next ---------")
            current_node = heapq.heappop(self.node_queue)
            self.node_counter += 1
            # Generate the state key for the current node
            state_key = self.node_state_key(current_node)
            if state_key in self.visited_states:
                logging.debug(f"Skipping node {current_node.node_id} at depth {current_node.depth} (state already visited)")
                continue
                
            # Mark this state as visited
            self.visited_states.add(state_key)

            # self.all_nodes.append(current_node)

            logging.info(f"Processing node at depth {current_node.depth}, node count: {self.node_counter}")
            logging.info(f"Current node bounds: {current_node.bounds}, fixed_vars: {current_node.fixed_vars}")

            # Apply bounds and fixings
            self.budget_solver.update_variable_bounds(current_node.fixed_vars, current_node.bounds)

            # Solve LP relaxation
            self.budget_solver.solve()
            lp_status = pulp.LpStatus[self.budget_solver.model.status]

            if lp_status != 'Infeasible':
                current_node.lp_value = self.budget_solver.get_totalN_rewards()

                logging.info(f"LP at {current_node.depth}, node count: {self.node_counter}, value: {current_node.lp_value}")
            else:
                logging.info(f"LP infeasible at {current_node.depth}, node count: {self.node_counter}")
                current_node.lp_value = None  # Indicate that LP is infeasible
                # prune node right now, if infeasible
                self.budget_solver.reset_variable_bounds()
                current_node.pruned = True
                self.all_nodes.append(current_node)
                continue
            
            # Prune feasible node
            if self.prune_node(current_node):
                current_node.pruned = True
                self.all_nodes.append(current_node)
                self.budget_solver.reset_variable_bounds()
                continue

            # if not pruned, test current_node.oracle_value and search for more
            current_node.solution = self.budget_solver.get_budget_solution()
            # print(f"current_node.solution = {current_node.solution}")

            LP_budget_allocation_now = np.array(self.budget_solver.get_budget_allocation(), dtype=int)
            current_node.oracle_value = self.oracle.evaluate(LP_budget_allocation_now)
            if current_node.oracle_value > self.best_lower_bound:
                self.best_lower_bound = current_node.oracle_value
                self.best_feasible_solution = LP_budget_allocation_now

            # Branch on variable
            child_nodes = self.branch_on_variable(current_node)
            for child_node in child_nodes:
                self.add_node_to_queue(child_node)

            current_node.pruned = False
            self.all_nodes.append(current_node)
            # Clean up
            self.budget_solver.reset_variable_bounds()

    def prune_node(self, node):
        if node.lp_value is not None and node.lp_value <= self.best_lower_bound:
            logging.info(f"Pruning node at depth {node.depth} due to bound ({node.lp_value} <= {self.best_lower_bound})")
            return True
        return False

    def evaluate_oracle(self, node):
        """
        returns a feasible LP based on node and oracle solution
        """
        # Extract B variables from the solution
        B_solution = {var_name: int(value) for var_name, value in node.solution.items() if var_name.startswith('B')}
        B_vector = [B_solution[f'B_{k}'] for k in range(self.budget_solver.K)]
        
        # Call the oracle function (R^S(B))
        oracle_reward = self.oracle.evaluate(B_vector)
        
        return oracle_reward
    
    def add_node_to_queue(self, node):
        # Set node priority (negative lp_value for max-heap)
        node.priority = -node.lp_value if node.lp_value is not None else float('-inf')
        heapq.heappush(self.node_queue, node)
        logging.debug(f"Added node at depth {node.depth} to queue with priority {node.priority}")

    def branch_on_variable(self, node):

        # Choose variable with maximum fractional part
        selectable_vars_key = [
        var_name for var_name in node.solution.keys()
        if var_name.startswith('B')
        ]
        assert selectable_vars_key, "No variables to branch on, what?"

        # Randomly select one of the variables to branch on
        branching_var = np.random.choice(selectable_vars_key)
        var_value = node.solution[branching_var]

        logging.info(f"Branching on variable {branching_var} with value {var_value}")

        # Get the current bounds for the branching variable
        if branching_var in node.bounds:
            ori_LB, ori_UB = node.bounds[branching_var]
        else:
            # Get default bounds
            # For 'B' variables, default lower bound is 0
            ori_LB = 0
            # Upper bound depends on budget and cost
            k = int(branching_var[2:])  # Get k from 'B_k'
            ori_UB = int(self.budget_solver.budget / self.budget_solver.context_prob[k])

        # Handle None values in ori_LB and ori_UB
        if ori_LB is None:
            ori_LB = 0  # Assume lower bound is 0
        if ori_UB is None:
            k = int(branching_var[2:])  # Get k from 'Bk'
            ori_UB = int(self.budget_solver.budget / self.budget_solver.context_prob[k])

        # Compute the mid-point of [ori_LB, ori_UB]
        mid_point = int((ori_LB + ori_UB) / 2)

        # Ensure mid_point is within bounds
        if mid_point < ori_LB:
            mid_point = int(ori_LB)
        if mid_point > ori_UB:
            mid_point = int(ori_UB)
        # Prepare bounds for child nodes
        left_bounds = node.bounds.copy()
        right_bounds = node.bounds.copy()

        # Left child: variable <= mid_point
        left_bounds[branching_var] = (ori_LB, mid_point)

        # Right child: variable >= mid_point + 1
        right_bounds[branching_var] = (mid_point + 1, ori_UB)

        # List to hold valid child nodes
        children = []

        # Check and create left child node if bounds are valid
        left_LB, left_UB = left_bounds[branching_var]
        if left_LB <= left_UB:
            left_node = Node(
                priority=0,
                fixed_vars=node.fixed_vars.copy(),
                bounds=left_bounds,
                depth=node.depth + 1,
                parent=node,
            )
            node.children.append(left_node)
            children.append(left_node)
            logging.info(f"Created left child with bounds {left_bounds}")
        else:
            logging.info(f"Left child has invalid bounds for {branching_var}: {left_LB} > {left_UB}, skipping left child.")

        # Check and create right child node if bounds are valid
        right_LB, right_UB = right_bounds[branching_var]
        if right_LB <= right_UB:
            right_node = Node(
                priority=0,
                fixed_vars=node.fixed_vars.copy(),
                bounds=right_bounds,
                depth=node.depth + 1,
                parent=node,
            )
            node.children.append(right_node)
            children.append(right_node)
            logging.info(f"Created right child with bounds {right_bounds}")
        else:
            logging.info(f"Right child has invalid bounds for {branching_var}: {right_LB} > {right_UB}, skipping right child.")

        return children
    

    def visualize_tree(self):
        """
        Visualizes the branching tree using NetworkX and Matplotlib with a custom hierarchical layout.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.DiGraph()

        # Add nodes and edges to the graph
        for node in self.all_nodes:
            node_label = f"Node {node.node_id}\nDepth: {node.depth}\n"
            if node.lp_value is None:
                node_label += "LP: Infeasible"
            else:
                node_label += f"LP: {node.lp_value:.2f}"
            if node.oracle_value is not None:
                node_label += f"\nOracle: {node.oracle_value:.2f}"
            # Include bounds if desired
            bounds_str = parse_bounds(node.bounds)
            node_label += f"\n{bounds_str}"

            G.add_node(node.node_id, label=node_label, pruned=node.pruned)

            if node.parent is not None:
                G.add_edge(node.parent.node_id, node.node_id)

        # Use the custom hierarchy_pos function
        pos = hierarchy_pos(
            G,
            root=self.all_nodes[0].node_id,
            width=1.0,
            vert_gap=0.5,
            xcenter=0.5,
            scale=2.0  # Increase scale to spread nodes more
        )

        # Extract labels
        labels = nx.get_node_attributes(G, 'label')

        # Determine node colors based on the 'pruned' flag
        node_colors = []
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            if node_data['pruned']:
                node_colors.append('pink')         # Pruned nodes
            else:
                node_colors.append('lightblue')   # Preserved nodes

        # Create figure and axes
        plt.figure(figsize=(24, 16))  # Larger figure size

        # Draw the nodes and edges
        nx.draw(
            G,
            pos,
            with_labels=False,
            arrows=True,
            node_size=2000,      # Increase node size
            node_color=node_colors,
            linewidths=0.25
        )
        nx.draw_networkx_labels(
            G,
            pos,
            labels,
            font_size=10        # Increase font size
        )

        # Add legend
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='pink', label='Pruned Node')
        blue_patch = mpatches.Patch(color='lightblue', label='Preserved Node')
        plt.legend(handles=[red_patch, blue_patch])

        plt.title('Branch and Bound Tree')
        plt.axis('off')
        plt.tight_layout()
        return plt
    

def main():
    """
    Main function to execute the script.

    Process:
    - Parses command-line arguments.
    - Initializes the RMAB simulator based on the arguments.
    - Initialize BudgetSolver based on the arguments and transitions
    - run branch n bound
    """
    args = parse_arguments()
    simulator, all_transitions, context_prob, reward_vectors = initialize_instance_and_simulator(args)

    oracle = Oracle(args, simulator)
    budget_solver = BudgetSolver(N = args.N, K = args.K, B = args.budget, all_transitions=all_transitions, context_prob=context_prob, w=reward_vectors, MIP = True)
    
    BnB_solver = BranchAndBoundSolver(budget_solver=budget_solver, oracle=oracle)
    BnB_solver.solve()
    plt = BnB_solver.visualize_tree()

    this_path = f'./results/{args.str_time}' + "BnB"

    MIP_rewards = brute_force_plot(simulator)
    MIP_rewards_valid = {}
    for key in MIP_rewards.keys():
        MIP_rewards_valid[str(key)] = MIP_rewards[key]
    args.MIP_rewards = MIP_rewards_valid

    p, q, _ = simulator.get_original_vectors()
    args.BnB_best_lowerbound = BnB_solver.best_lower_bound
    args.BnB_best_alllocation = BnB_solver.best_feasible_solution
    write_result(rewards=None, use_algos=["Branch_n_Bound"], args=args, transition_probabilities=all_transitions, context_prob=context_prob, p = p, q = q, rewards_to_write=BnB_solver.best_lower_bound, best_allocation=BnB_solver.
    best_feasible_solution, result_name="BnB")
    plt.savefig(this_path + f'/Branch_n_bound_result-{args.exp_name_out}.pdf')

    



if __name__ == '__main__':
    main()