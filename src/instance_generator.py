# instance_generator.py

"""
wrapper: initialize_instance_and_simulator
take input: args
output: simulator, all_transitions, context_prob, reward_vector



InstanceGenerator

This module provides functionality to generate and manage instances for the Volunteer Restless Multi-Armed Bandit (RMAB) problem. It includes:

- `InstanceGenerator` class: Generates random instances with specified parameters and saves them for future use.
- Methods to construct transition matrices and extract parameters like `p`, `q`, and context probabilities.
- Functions to generate instances with given probabilities and load existing instances

Usage:
    - Generate instances:
        generator = InstanceGenerator(N=25, K=3, seed=42)
        generator.generate_instances(M=5)
    - Load an instance:
        instance = generator.load_instance()
    - Generate instance with given probabilities:
        generator.generate_instance_given_probs(p, q, context_prob, M=1, name="custom_instance")
"""

import json
import numpy as np
import os
import random
from volunteer_simulator import Volunteer_RMABSimulator
import logging
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.WARNING)  # Set the logging level


def initialize_instance_and_simulator(args):
    """
    Initializes the RMAB simulator with the given arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing simulation parameters.

    Returns:
        tuple:
            - simulator (Volunteer_RMABSimulator): The initialized RMAB simulator.
            - all_transitions (that big Prob matrix)
            - context_prob (list): A list of probabilities for each context.
            - reward_vector (for now, np.ones)
    """
    # Set the random seed for reproducibility
    np.random.seed(args.seed)

    if args.instance_type == 'seed_based':
        print('using seed_based vanilla generation')
        # generate local data using a same seed—if N, K be the same, generated probability transitions should also be the same
        generator = InstanceGenerator(N=args.N, K=args.K, seed=args.seed)
        generator.generate_instances(homogeneous=args.homogeneous)
        instance = generator.load_instance()
        all_transitions, context_prob = instance['transitions'], instance['context_prob']
    
        reward_vector = np.ones(args.K)
    elif args.instance_type == 'frictional_real_data_based':
        print(f'loading data instance_type {args.instance_type}')
        generator = RealRegionalInstanceGenerator(N=args.N, K=args.K, seed=args.seed)
        generator.generate_instances()
        instance = generator.load_instance()
        all_transitions, context_prob = instance['transitions'], instance['context_prob']
        reward_vector = np.ones(args.K)
        this_path = f'./results/{args.str_time}'
        if not os.path.exists(this_path):
            os.makedirs(this_path)
        generator.save_two_description_plots(dir=this_path)
    else:
        raise Exception(f'datatype {args.instance_type} not implemented')

    # Set reward vector (assuming unit rewards for all contexts)
    reward_vector = np.ones(args.num_context)

    # Initialize the simulator with the provided parameters
    simulator = Volunteer_RMABSimulator(
        N=args.n_arms,
        K=args.num_context,
        T=args.episode_len,
        context_prob=context_prob,
        all_transitions=all_transitions,
        budget=args.budget,
        reward_vector=reward_vector,
    )
    return simulator, all_transitions, context_prob, reward_vector


def initialize_real_instance(args):
    """
    Initializes the Contexual RMAB simulator based on args, default with real data.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing simulation parameters.

    Returns:
        tuple:
            - (*new) Real_Instance_Generator
                - (NONE! deleted) simulator
            - all_transitions (that big Prob matrix)
            - context_prob (list): A list of probabilities for each context.
            - reward_vector (for now, np.ones)
           teer_RMABSimulator): The initialized RMAB simulator.
    """
    # Set the random seed for reproducibility
    np.random.seed(args.seed)

    if args.instance_type == 'frictional_real_data_based':
        print(f'loading data instance_type {args.instance_type}')
        generator = RealRegionalInstanceGenerator(N=args.N, K=args.K, seed=args.seed)
        generator.generate_instances()
        instance = generator.load_instance()
        all_transitions, context_prob = instance['transitions'], instance['context_prob']
        reward_vector = np.ones(args.K)
        this_path = f'./results/{args.str_time}'
        if not os.path.exists(this_path):
            os.makedirs(this_path)
        generator.save_two_description_plots(dir=this_path)
    else:
        raise Exception(f'only frictional_real_data_based is allowed.\n though now args.instance_type is {args.instance_type}')

    # # Initialize the simulator with the provided parameters
    # simulator = Volunteer_RMABSimulator(
    #     N=args.n_arms,
    #     K=args.num_context,
    #     T=args.episode_len,
    #     context_prob=context_prob,
    #     all_transitions=all_transitions,
    #     budget=args.budget,
    #     reward_vector=reward_vector,
    # )
    return generator, all_transitions, context_prob, reward_vector


class InstanceGenerator:
    """
    A class to generate and manage instances for the Volunteer RMAB problem.
    saving structure:
        ├── instances/
        │   ├── seed_based/
        │   │   ├── N_25_K_3_homo_True_seed_66/
        │   │   │   ├── instance_data.json
        │   ├── instance_type_name/
        │   │   ├── type_identifier/
        │   │   │   ├── instance_data.json
        

    Attributes:
        N (int): Number of arms.
        K (int): Number of contexts.
        seed (int): Random seed for reproducibility.
        data_dir (str): Directory to save generated instances.
        identifier (optional, str): Directory for special additional info
    """

    def __init__(self, N, K, seed=66, data_dir='data', homogeneous = False, identifier=None):
        """
        Initializes the InstanceGenerator with specified parameters.

        Args:
            N (int): Number of arms.
            K (int): Number of contexts.
            seed (int, optional): Random seed. Defaults to 66.
            data_dir (str, optional): Directory to save data. Defaults to 'data'.
        """
        self.N = N
        self.K = K
        self.seed = seed
        self.data_dir = data_dir
        self.homogeneous = homogeneous

        # DEFAULT 'seed_based'
        self.instance_type = 'seed_based'

        if identifier is None:
            self.identifier = f'N_{self.N}_K_{self.K}_homo_{self.homogeneous}_seed_{self.seed}'
        else:
            self.identifier = identifier

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def generate_instances(self, homogeneous=True, identifier = None):
        """
        Generates seed-based instances and saves them to disk.

        Args:
            homogeneous (bool, optional): If True, generates homogeneous instances. Defaults to True.
        """
        # overwrite if necessary
        if homogeneous is not None:
            self.homogeneous = homogeneous

        transitions, context_prob = self.randomly_generate_transitions(
            N=self.N, K=self.K, homogeneous=self.homogeneous
        )

        # instance_dir = os.path.join(
        #     self.data_dir, f'N_{self.N}_K_{self.K}_seed_{self.seed}_homo_{self.homogeneous}_num_{m}'
        # )
        # if not os.path.exists(instance_dir):
        #     os.makedirs(instance_dir)

        instance = {
            'transitions': transitions.tolist(),
            'context_prob': context_prob.tolist()
        }

        if identifier is not None:
            pattern = identifier
        else:
            pattern = self.identifier
        instance_dir = os.path.join(
            self.data_dir,
            self.instance_type, 
            pattern
            )

        self.save_instance(instance, instance_dir)

    def save_instance(self, instance, instance_dir):
        """
        Saves an instance to the specified directory.

        Args:
            instance (dict): The instance data containing transitions and context probabilities.
            instance_dir (str): The directory to save the instance.
        """
        # file_path = os.path.join(instance_dir, 'instance_data.json')
        # with open(file_path, 'w') as f:
        #     json.dump(instance, f, indent=4)
        # print(f"{file_path} saved successfully")
        if not os.path.exists(instance_dir):
            os.makedirs(instance_dir)
        file_path = os.path.join(instance_dir, 'instance_data.json')
        with open(file_path, 'w') as f:
            json.dump(instance, f, indent=4)
        print(f"{file_path} saved successfully")

    def construct_volunteer_transition_matrix(self, N, K, q, context_prob, p, n_actions=2):
        """
        Constructs the transition matrix for the Volunteer RMAB problem.

        Args:
            N (int): Number of arms.
            K (int): Number of contexts.
            q (numpy.ndarray): Array of shape (N,) representing the probability of success without action.
            context_prob (numpy.ndarray): Array of shape (K,) representing context probabilities.
            p (numpy.ndarray): Array of shape (N, K) representing the probability of success with action in each context.
            n_actions (int, optional): Number of actions. Defaults to 2.

        Returns:
            numpy.ndarray: The constructed transition matrix of shape (N, n_states, n_actions, n_states).
        """
        n_states = K * 2
        all_transitions = np.zeros((N, n_states, n_actions, n_states))
        for k in range(K):
            for i in range(N):
                # When state s = 0
                # Action a = 0
                all_transitions[i, 2 * k, 0, 1::2] = q[i] * context_prob
                all_transitions[i, 2 * k, 0, 0::2] = (1 - q[i]) * context_prob
                # Action a = 1
                all_transitions[i, 2 * k, 1, 1::2] = q[i] * context_prob
                all_transitions[i, 2 * k, 1, 0::2] = (1 - q[i]) * context_prob

                # When state s = 1
                # Action a = 0
                all_transitions[i, 2 * k + 1, 0, 1::2] = context_prob
                # Action a = 1
                all_transitions[i, 2 * k + 1, 1, 0::2] = p[i, k] * context_prob
                all_transitions[i, 2 * k + 1, 1, 1::2] = (1 - p[i, k]) * context_prob

        # Check if transition probabilities sum to 1
        assert np.allclose(np.sum(all_transitions, axis=-1), 1, atol=1e-6), \
            "Transition probabilities do not sum to 1."
        return all_transitions

    def randomly_generate_transitions(self, N, K, homogeneous=True):
        """
        Randomly generates transition probabilities for the instances.

        Args:
            N (int): Number of arms.
            K (int): Number of contexts.
            homogeneous (bool, optional): If True, all arms have the same parameters. Defaults to True.

        Returns:
            tuple:
                - all_transitions (numpy.ndarray): Transition probability matrices.
                - context_prob (numpy.ndarray): Context probabilities.
        """
        if homogeneous:
            q = np.ones(N) * np.random.rand()
            p = np.ones((N, K)) * np.random.rand(K)
        else:
            q = np.random.rand(N)
            p = np.random.rand(N, K)

        context_prob = np.random.rand(K)
        context_prob /= np.sum(context_prob, keepdims=True)

        all_transitions = self.construct_volunteer_transition_matrix(
            N=N, K=K, q=q, context_prob=context_prob, p=p
        )
        return all_transitions, context_prob

    def load_instance(self, instance_type=None, identifier=None):
        if instance_type is None:
            instance_type = self.instance_type
        if identifier is None:
            identifier = self.identifier
        
        # Load the appropriate instance from the file system
        instance_dir = os.path.join(self.data_dir, instance_type, identifier)
        file_path = os.path.join(instance_dir, 'instance_data.json')
        with open(file_path, 'r') as f:
            instance = json.load(f)
            instance['transitions'] = np.array(instance['transitions'])
            instance['context_prob'] = np.array(instance['context_prob'])
        return instance

    def print_instance(self, all_transitions, context_prob):
        """
        Prints the parameters of the instance for preview.

        Args:
            all_transitions (numpy.ndarray): Transition probability matrices.
            context_prob (numpy.ndarray): Context probabilities.
        """
        p = np.zeros((self.N, self.K))
        q = np.zeros(self.N)

        for i in range(self.N):
            q[i] = np.sum(all_transitions[i, 0, 0, 1::2])
            for k in range(self.K):
                p[i, k] = np.sum(all_transitions[i, k * 2 + 1, 1, ::2])

        if np.all(all_transitions[0] == all_transitions[1]):
            print(f"p_k:\n{np.round(p, 2)[0]}")
            print(f"q = {np.round(q, 2)[0]}")
            print(f"f_k = {np.round(context_prob, 2)}")

    def get_original_vectors(self, all_transitions, context_prob):
        """
        Extracts the original vectors p, q, and context probabilities.

        Args:
            all_transitions (numpy.ndarray): Transition probability matrices.
            context_prob (numpy.ndarray): Context probabilities.

        Returns:
            tuple:
                - p (numpy.ndarray): Probability of success with action.
                - q (numpy.ndarray): Probability of success without action.
                - context_prob (numpy.ndarray): Context probabilities.
        """
        p = np.zeros((self.N, self.K))
        q = np.zeros(self.N)

        for i in range(self.N):
            q[i] = np.sum(all_transitions[i, 0, 0, 1::2])
            for k in range(self.K):
                p[i, k] = np.sum(all_transitions[i, k * 2 + 1, 1, ::2])

        return p, q, context_prob

    def generate_instance_given_probs(self, p, q, context_prob, instance_type="given_p_q", identifier='identifier_placeholder'):
        """
        Generates instances with given probabilities and saves them.

        Args:
            p (numpy.ndarray): Probability of success with action.
            q (numpy.ndarray): Probability of success without action.
            context_prob (numpy.ndarray): Context probabilities.
            instance_type (str, optional): Instance Type. Defaults to "real_data_based".
            instance_identifier (str, optimal): Identifier for the specific instance
        """
        # Check for mismatches between lengths of p, q, and context_prob vs. self.N and self.K
        if len(p) != self.N:
            logging.warning(f"Length of p ({len(p)}) does not match self.N ({self.N}). Adjusting self.N to {len(p)}.")
            self.N = len(p)  # Adjust self.N to match the length of p
        
        if len(q) != self.N:
            logging.warning(f"Length of q ({len(q)}) does not match self.N ({self.N}). Adjusting self.N to {len(q)}.")
            self.N = len(q)  # Adjust self.N to match the length of q

        if len(context_prob) != self.K:
            logging.warning(f"Length of context_prob ({len(context_prob)}) does not match self.K ({self.K}). Adjusting self.K to {len(context_prob)}.")
            self.K = len(context_prob)  # Adjust self.K to match the length of context_prob
    
        # Create directory based on instance type
        instance_dir = os.path.join(
            self.data_dir,
            instance_type,
            identifier
        )
        if not os.path.exists(instance_dir):
            os.makedirs(instance_dir)

        # Generate and save M instances
        transitions = self.construct_volunteer_transition_matrix(N=self.N, K=self.K, q=q, context_prob=context_prob, p=p)

        # Create the instance data
        instance = {
            'transitions': transitions.tolist(),
            'context_prob': context_prob.tolist()
        }
        # Save the instance
        self.save_instance(instance, instance_dir)

def initial_state_generator(n_epochs, N, K, seed, context_prob):
    """
    A class to generate and manage initial states for the Volunteer RMAB problem.

    Args:
        n_epochs (int): this number of states
        K (int): Number of contexts.
        N (int): Number of arms.
        seed (int): Random seed for reproducibility.

    Return:
        a list initial_states_list: each element is (state, context) pair
        each state is [N]-size numpy array (int)
            states[i] = 2*k + 1 (active), 2*k (inactive)
        each context is an int k

    """
    np.random.seed(seed)
    random.seed(seed)

    state_list = []
    for i in range(n_epochs):
        context = np.random.choice(a=K, p=context_prob)
        states = np.random.choice(a=2, size = N, p = [0.5, 0.5])
        states += 2*context
        state_list.append((states, context))

    return state_list

def f_q(his):
    """
    Computes q value (recovery_rate) based on the historical jobs completed by a volunteer.

    Args:
        his (int): Number of historical jobs completed by the volunteer.

    Returns:
        float: Probability of success without action (q).
    """
    # Convex function for q increasing with historical jobs completed
    return min(1, 1 - np.exp(-0.005 * his))

def p_his(his):
    """
    Computes sensitivity to distance based on historical jobs completed by a volunteer.

    Args:
        his (int): Number of historical jobs completed by the volunteer.

    Returns:
        float: Sensitivity to distance.
    """
    # Sensitivity function, decreases with more historical jobs
    return -0.001 * (50 - his)

class Region:
    def __init__(self, x, y, favourability):
        """
        Initializes a region with x and y coordinates.

        Args:
            x (float): X coordinate of the region.
            y (float): Y coordinate of the region.
            favourability (float): Favourability score of the region.
        """
        self.x = x
        self.y = y
        self.favourability = favourability

class Volunteer:
    def __init__(self, his, x, y):
        """
        Initializes a volunteer with historical jobs completed and x, y coordinates.

        Args:
            his (int): Number of historical jobs completed by the volunteer.
            x (float): X coordinate of the volunteer.
            y (float): Y coordinate of the volunteer.
        """
        self.his = his
        self.x = x
        self.y = y

class RealRegionalInstanceGenerator(InstanceGenerator):
    def __init__(self, N, K, seed=66, data_dir='data', homogeneous=False, f_q=f_q, p_his=p_his, identifier = None):
        """
        Initializes the RealRegionalInstanceGenerator with given parameters and functions for q and p computation.

        Args:
            N (int): Number of volunteers.
            K (int): Number of regions.
            seed (int): Random seed for reproducibility.
            data_dir (str): Directory to save generated instances.
            homogeneous (bool): Indicates if the instances are homogeneous.
            f_q (function): Function to compute q value based on historical jobs.
            p_his (function): Function to compute sensitivity to distance based on historical jobs.
        """
        super().__init__(N, K, seed, data_dir, homogeneous)
        self.regions = []
        self.volunteers = []
        self.f_q = f_q
        self.p_his = p_his


        # default fixed
        self.instance_type = 'frictional_real_data_based'
        
        if identifier is None:
            self.identifier = f'N_{self.N}_K_{self.K}_homo_{self.homogeneous}_seed_{self.seed}'
        else:
            self.identifier = identifier

    def generate_frictionalized_data(self):
        """
        Generates frictionalized data for regions and volunteers based on the seed.
        """
        # Set random seed
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        # Generate K regions with random locations and favourability scores
        self.regions = [Region(random.uniform(0, 100), random.uniform(0, 100), random.uniform(0.5, 1.5)) for _ in range(self.K)]

        # Generate N volunteers with random historical jobs completed and locations
        self.volunteers = [Volunteer(random.randint(1, 50), random.uniform(0, 100), random.uniform(0, 100)) for _ in range(self.N)]

        self.context_prob = np.random.rand(self.K)
        self.context_prob /= np.sum(self.context_prob)
    
    def compute_distance(self, volunteer, region):
        """
        Computes Euclidean distance between a volunteer and a region.

        Args:
            volunteer (Volunteer): The volunteer object.
            region (Region): The region object.

        Returns:
            float: Euclidean distance.
        """
        return np.sqrt((volunteer.x - region.x) ** 2 + (volunteer.y - region.y) ** 2)
    
    def compute_p(self):
        """
        Computes the probability of success with action for each volunteer-region pair.

        Returns:
            numpy.ndarray: Probability matrix p of shape (N, K).
        """
        p_matrix = np.zeros((self.N, self.K))
        favourabilities = [region.favourability for region in self.regions]
        for i, volunteer in enumerate(self.volunteers):
            distances = [self.compute_distance(volunteer, region) for region in self.regions]
            sensitivity = self.p_his(volunteer.his)
            exp_values = [np.exp(sensitivity * d) * fav for d, fav in zip(distances, favourabilities)]
            exp_sum = sum(exp_values)
            for k in range(self.K):
                p_matrix[i][k] = exp_values[k] / exp_sum
                # ensure 0 <= p <= 1
                p_matrix[i][k] = min(1, max(0, p_matrix[i][k]))
        return p_matrix
    
    def generate_instances(self):
        """
        Generates real-data-based instances using the frictionalized data.
        """
        # Generate frictionalized data for regions and volunteers
        self.generate_frictionalized_data()

        # Compute q values for each volunteer
        self.q = np.array([self.f_q(volunteer.his) for volunteer in self.volunteers])

        # Compute p values for each volunteer-region pair
        self.p = self.compute_p()

        self.instance_type = "frictional_real_data_based"
        self.identifier = f'N_{self.N}_K_{self.K}_homo_{self.homogeneous}_seed_{self.seed}'
        self.generate_instance_given_probs(
            self.p, self.q, self.context_prob,
            instance_type=self.instance_type,
            identifier=self.identifier
        )

    def save_two_description_plots(self, dir = None):
        self.visualize_f_q_and_p_his(dir=dir)
        self.visualize_volunteers_and_regions(dir=dir)

    def visualize_volunteers_and_regions(self, dir = None):
        """
        Plots the allocation of budget to volunteers and regions, incorporating solved results from the budget solver.

        Args:
            real_instance_generator (RealRegionalInstanceGenerator): Instance generator with regions and volunteers.
            budget_solver (BudgetSolver): Solver to allocate budget to volunteers and regions.
            theta (float): Fairness lower bound parameter.
            args: Additional arguments including time and experiment name for saving the plot.
        """

        colors = plt.colormaps['tab10']

        plt.figure(figsize=(14, 14))

        x_min, x_max = -20, 120
        y_min, y_max = -20, 120
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        for i, volunteer in enumerate(self.volunteers):
            volunteer_x, volunteer_y = volunteer.x, volunteer.y
            # Determine the color of the volunteer based on the region with highest p[i][k]
            dominant_region = np.argmin([self.compute_distance(volunteer, region) for region in self.regions])
            volunteer_color = colors(dominant_region)
            # Determine the size of the volunteer based on (sum_k p[i][k]) * q[i]
            volunteer_size = volunteer.his * 3.6
            plt.scatter(volunteer_x, volunteer_y, c=[volunteer_color], marker='o', s=volunteer_size, edgecolor=volunteer_color, linewidth=1, label='Volunteers' if i == 0 else "", zorder=3)

            # Plot regions with budget allocation
        for k, region in enumerate(self.regions):
            region_x, region_y = region.x, region.y

            plt.scatter(region_x, region_y, c=[colors(k)], marker='*', s=360 * region.favourability, alpha=0.8, edgecolor=[colors(k)], linewidth=1.5, label=f'Region {k}', zorder = 3)

        # Set labels and title
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Volunteers and Regions on a Plane')
        plt.legend(loc='upper right')
        plt.grid(True)

        if dir is None:
            save_dir = os.path.join(
            self.data_dir, 
            self.instance_type, 
            self.identifier, 
            'f_q_p_his_plot.png'
            )
        else:
            save_dir = os.path.join(dir, 'f_q_p_his_plot.png')
        
        plt.savefig(save_dir)


    # def visualize_volunteers_and_regions(self, dir = None):
    #     """
    #     Visualizes the locations of volunteers and regions on a 2D plane.
    #     """
    #     plt.figure(figsize=(10, 8))
    #     # Plot regions
    #     region_x = [region.x for region in self.regions]
    #     region_y = [region.y for region in self.regions]
    #     region_fav = [region.favourability for region in self.regions]
    #     plt.scatter(region_x, region_y, c='red', marker='^', s=[100 * fav for fav in region_fav], label='Regions (scaled by favourability)')
    #     # Plot volunteers
    #     volunteer_x = [volunteer.x for volunteer in self.volunteers]
    #     volunteer_y = [volunteer.y for volunteer in self.volunteers]
    #     plt.scatter(volunteer_x, volunteer_y, c='blue', marker='o', s=50, label='Volunteers')

    #     plt.xlabel('X Coordinate')
    #     plt.ylabel('Y Coordinate')
    #     plt.title('Volunteers and Regions on a Plane')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(os.path.join(
    #         self.data_dir, 
    #         self.instance_type, 
    #         self.identifier, 
    #         'volunteers_regions_plot.png'
    #         ))
        
    #     if dir is None:
    #         save_dir = os.path.join(
    #         self.data_dir, 
    #         self.instance_type, 
    #         self.identifier, 
    #         'volunteers_regions_plot.png'
    #         )
    #     else:
    #         save_dir = os.path.join(dir, 'volunteers_regions_plot.png')
        
    #     plt.savefig(save_dir)
    #     # plt.show()

    def visualize_f_q_and_p_his(self, dir = None):
        """
        Visualizes the functions f_q and p_his with respect to the historical jobs completed (his).
        """
        his_values = np.arange(1, 51)
        f_q_values = [self.f_q(his) for his in his_values]
        p_his_values = [self.p_his(his) for his in his_values]

        plt.figure(figsize=(12, 5))
        # Plot f_q
        plt.subplot(1, 2, 1)
        plt.plot(his_values, f_q_values, label='f_q(his)', color='green')
        plt.xlabel('Historical Jobs Completed (his)')
        plt.ylabel('Recover Rate (q[i])')
        plt.title('Recover Rate vs Historical Jobs Completed')
        plt.grid(True)
        plt.legend()

        # Plot p_his
        plt.subplot(1, 2, 2)
        plt.plot(his_values, p_his_values, label='p_his(his)', color='purple')
        plt.xlabel('Historical Jobs Completed (his)')
        plt.ylabel('Sensitivity to Distance (p_his)')
        plt.title('p_his vs Historical Jobs Completed')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        if dir is None:
            save_dir = os.path.join(
            self.data_dir, 
            self.instance_type, 
            self.identifier, 
            'f_q_p_his_plot.png'
            )
        else:
            save_dir = os.path.join(dir, 'f_q_p_his_plot.png')
        
        plt.savefig(save_dir)
        # plt.show()

if __name__ == '__main__':
    # Sanity check for generated instances
    logging.info("---- Sanity check for instance generation ----")
    N = 100
    seed = 43
    B = 5
    K = 5
    generator = InstanceGenerator(N=N, K=K, seed=seed)
    generator.generate_instances()
    instance = generator.load_instance()
    all_transitions, context_prob = instance['transitions'], instance['context_prob']
    p, q, _ = generator.get_original_vectors(all_transitions, context_prob)
    print(f"p[0] = {p[0]}, q[0] = {q[0]}")
    print(f"context_prob = {context_prob}")

    T = 100
    reward_vector = np.ones(K)
    budget = B
    simulator = Volunteer_RMABSimulator(
        N=N, K=K, T=T, context_prob=context_prob,
        all_transitions=all_transitions, budget=budget, reward_vector=reward_vector
    )
    total_reward = 0

    for t in range(T):
        action = np.zeros(N)
        selection_idx = np.random.choice(a=N, size=budget, replace=False)
        action[selection_idx] = 1
        action = action.astype(int)

        states, reward, done, _ = simulator.step(action)
        total_reward += reward

    logging.info(f'Total reward:{total_reward}')
    logging.info(f'final states:{states}')

    logging.info("---- Sanity check for instance generation on given p, q: ----")
    generator = InstanceGenerator(N=N, K=K)
    context_prob = np.ones(K)/K
    p = np.ones((N, K))*0.2
    q = np.ones(N)*0.5

    generator.generate_instance_given_probs(p=p, q=q, context_prob=context_prob)
    instance = generator.load_instance(instance_type="real_data_based", identifier='identifier_placeholder')

    all_transitions, context_prob = instance['transitions'], instance['context_prob']
    p, q, _ = generator.get_original_vectors(all_transitions, context_prob)
    print(f"p[0] = {p[0]}, q[0] = {q[0]}")
    print(f"context_prob = {context_prob}")

    T = 100
    reward_vector = np.ones(K)
    budget = B
    simulator = Volunteer_RMABSimulator(
        N=N, K=K, T=T, context_prob=context_prob,
        all_transitions=all_transitions, budget=budget, reward_vector=reward_vector
    )
    total_reward = 0

    for t in range(T):
        action = np.zeros(N)
        selection_idx = np.random.choice(a=N, size=budget, replace=False)
        action[selection_idx] = 1
        action = action.astype(int)

        states, reward, done, _ = simulator.step(action)
        total_reward += reward

    logging.info(f'Total reward:{total_reward}')
    logging.info(f'final states:{states}')

    logging.info("---- Sanity check for real generator on given p, q: ----")
    real_generator = RealRegionalInstanceGenerator(N=100, K=5, seed=43)
    real_generator.generate_instances()
    real_generator.visualize_volunteers_and_regions()
    real_generator.visualize_f_q_and_p_his()

    logging.debug(f'p:\n{real_generator.p.round(3)}')
    logging.debug(f'q:\n{real_generator.q.round(3)}')