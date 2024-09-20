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


    if args.data == 'local_generated':
        if args.data_name == None:
            print('using locally generated data w.r.t. a seed that fixing everything')
            # generate local data using a same seed—if N, K be the same, generated probability transitions should also be the same
            generator = InstanceGenerator(N=args.N, K=args.K, seed=args.seed)
            # let's say we only support one instance for now
            num_instances = 1
            generator.generate_instances(num_instances, homogeneous=args.homogeneous)
            instance = generator.load_instance()
            all_transitions, context_prob = instance['transitions'], instance['context_prob']
        
            reward_vector = np.ones(args.K)
        else:
            print(f'loading data named {args.data_name}')
            generator = InstanceGenerator(N=args.N, K=args.K, seed=args.seed)
            num_instances = 1
            instance = generator.load_instance(name=args.data_name)
            all_transitions, context_prob = instance['transitions'], instance['context_prob']
            reward_vector = np.ones(args.K)

    else:
        raise Exception(f'dataset {args.data} not implemented')

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


class InstanceGenerator:
    """
    A class to generate and manage instances for the Volunteer RMAB problem.

    Attributes:
        N (int): Number of arms.
        K (int): Number of contexts.
        seed (int): Random seed for reproducibility.
        data_dir (str): Directory to save generated instances.
    """

    def __init__(self, N, K, seed=66, data_dir='data'):
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
        self.data_dir = os.path.join(data_dir, f'N_{N}_K_{K}_seed_{seed}')
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def generate_instances(self, M=1, homogeneous=True):
        """
        Generates M instances and saves them to disk.

        Args:
            M (int, optional): Number of instances to generate. Defaults to 1.
            homogeneous (bool, optional): If True, generates homogeneous instances. Defaults to True.
        """
        for m in range(M):
            transitions, context_prob = self.randomly_generate_transitions(
                N=self.N, K=self.K, homogeneous=homogeneous
            )
            instance_dir = os.path.join(
                self.data_dir, f'N_{self.N}_K_{self.K}_seed_{self.seed}_num_{m}'
            )
            if not os.path.exists(instance_dir):
                os.makedirs(instance_dir)

            instance = {
                'transitions': transitions.tolist(),
                'context_prob': context_prob.tolist()
            }
            self.save_instance(instance, instance_dir)

    def save_instance(self, instance, instance_dir):
        """
        Saves an instance to the specified directory.

        Args:
            instance (dict): The instance data containing transitions and context probabilities.
            instance_dir (str): The directory to save the instance.
        """
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

    def load_instance(self, name=None):
        """
        Loads an instance from disk.

        Args:
            name (str, optional): Name identifier for the instance. Defaults to None.

        Returns:
            dict: The loaded instance containing transitions and context probabilities.
        """
        pattern = f'N_{self.N}_K_{self.K}_'
        if name is None:
            pattern += f"seed_{self.seed}_"
            directories = [d for d in os.listdir(self.data_dir) if d.startswith(pattern)]
        else:
            pattern += f"NAME_{name}_"
            directories = [d for d in os.listdir(self.data_dir) if d.startswith(pattern)]
        pattern += f"num_"
        if not directories:
            assert name is None, f"Directory starting with {pattern} doesn't exist"
            print("No existing instances found, generating new instance...")
            self.generate_instances(1)
            directories = [d for d in os.listdir(self.data_dir) if d.startswith(pattern)]

        instance_dir = os.path.join(self.data_dir, directories[0])
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

    def generate_instance_given_probs(self, p, q, context_prob, M=1, name="placeholder"):
        """
        Generates instances with given probabilities and saves them.

        Args:
            p (numpy.ndarray): Probability of success with action.
            q (numpy.ndarray): Probability of success without action.
            context_prob (numpy.ndarray): Context probabilities.
            M (int, optional): Number of instances to generate. Defaults to 1.
            name (str, optional): Name identifier for the instance. Defaults to "placeholder".
        """
        N = self.N
        K = self.K
        seed = self.seed
        instances = []
        self.data_dir = self.data_dir.removeprefix(f'data/N_{N}_K_{K}_seed_{seed}')
        self.data_dir = os.path.join(f'data/N_{N}_K_{K}_NAME_{name}', self.data_dir)
        for m in range(M):
            transitions = self.construct_volunteer_transition_matrix(
                N=self.N, K=self.K, q=q, context_prob=context_prob, p=p
            )
            instance_dir = os.path.join(
                self.data_dir, f'N_{N}_K_{K}_NAME_{name}_num_{m}'
            )
            if not os.path.exists(instance_dir):
                os.makedirs(instance_dir)

            instance = {
                'transitions': transitions.tolist(),
                'context_prob': context_prob.tolist()
            }
            instances.append(instance)
            self.save_instance(instance, instance_dir)

if __name__ == '__main__':
    # Sanity check for generated instances
    print("---- Sanity check for instance generation: if q is correct ----")
    N = 25
    seed = 43
    B = 5
    K = 3
    generator = InstanceGenerator(N=N, K=K, seed=seed)
    generator.generate_instances()
    instance = generator.load_instance()
    all_transitions, context_prob = instance['transitions'], instance['context_prob']
    p, q, _ = generator.get_original_vectors(all_transitions, context_prob)
    print(f"p[0] = {p[0]}, q[0] = {q[0]}")

    # Sanity check for random method
    print("---- Sanity check for random method ----")
    K = 6
    N = 60
    seed = 66
    generator = InstanceGenerator(N=N, K=K, seed=seed)
    instance = generator.load_instance()
    print(f"Loaded Instance:\n{instance}")

    # Simulation using loaded instance
    all_transitions, context_prob = instance['transitions'], instance['context_prob']
    T = 100
    budget = 20
    reward_vector = np.ones(K)
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

    print('Total reward:', total_reward)
    print('Final states:', states)

    # Test: Generate almost-trivial instance
    print("\n\n---- Sanity check for load-data method ----")
    seed = 43
    N = 25
    K = 2
    generator = InstanceGenerator(N=N, K=K, seed=seed)
    p = np.array([[0.999, 0] for _ in range(N)])
    context_prob = np.array([0.5, 0.5])
    q = np.ones(N) * 0.2
    generator.generate_instance_given_probs(p=p, q=q, context_prob=context_prob, name="simpleK2")
    instance = generator.load_instance(name="simpleK2")
    print(f"Loaded Instance:\n{instance['transitions'][0]}, {instance['context_prob'][0]}")

    # Simulation using loaded instance
    all_transitions, context_prob = instance['transitions'], instance['context_prob']
    p, q, _ = generator.get_original_vectors(all_transitions, context_prob)
    print("p, q = ", p, q)
    T = 100
    budget = 5
    reward_vector = np.ones(K)
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

    print('Total reward:', total_reward)
    print('Final states:', states)


# # instance_generator.py
# # both instance generator and loader

# import json
# import numpy as np
# import os
# import random
# from volunteer_simulator import Volunteer_RMABSimulator

# class InstanceGenerator:
#     def __init__(self, N, K, seed=66, data_dir='data'):
#         self.N = N
#         self.K = K
#         self.seed = seed
#         self.data_dir = os.path.join(data_dir, f'N_{N}_K_{K}_seed_{seed}')
#         if seed is not None:
#             np.random.seed(seed)
#             random.seed(seed)
#         if not os.path.exists(self.data_dir):
#             os.makedirs(self.data_dir)

#     def generate_instances(self, M = 1, homogeneous = True):
#         instances = []
#         for m in range(M):
#             transitions, context_prob = self.randomly_generate_transitions(N=self.N, K=self.K, homogeneous=homogeneous)
#             instance_dir = os.path.join(self.data_dir, f'N_{self.N}_K_{self.K}_seed_{self.seed}_num_{m}')
#             if not os.path.exists(instance_dir):
#                 os.makedirs(instance_dir)
            
#             instance = {
#                 'transitions': transitions.tolist(),
#                 'context_prob': context_prob.tolist()
#             }
#             instances.append(instance)
#             self.save_instance(instance, instance_dir)


#     def save_instance(self, instance, instance_dir):
#         file_path = os.path.join(instance_dir, 'instance_data.json')
#         with open(file_path, 'w') as f:
#             json.dump(instance, f, indent=4)
#         print(f"{file_path} saved successfully")

#     def construct_volunteer_transition_matrix(self, N, K, q, context_prob, p, n_actions = 2):
#         """
#         given parameters, construct transition_matrix
#         Paramters:
#             q: shape N
#             p: shape N*K
#             context_prob: shape K
#             -> n_states = 2*K
#         Return
#             transition_matrix: shape N*n_states*2*n_states
#         """
#         n_states = K*2
#         all_transitions = np.zeros((N, n_states, n_actions, n_states))
#         for k in range(K):
#             for i in range(N):
#                 # when s = 0
#                 # action = 0
#                 all_transitions[i, 2*k, 0, 1::2] = q[i]*context_prob # s = 0 -> s = 1 w.p. q
#                 all_transitions[i, 2*k, 0, 0::2] = (1 - q[i])*context_prob # s = 0 -> s = 0 w.p. 1 - q
#                 # action = 1
#                 all_transitions[i, 2*k, 1, 1::2] = q[i]*context_prob # s = 0 -> s = 1 w.p. q
#                 all_transitions[i, 2*k, 1, 0::2] = (1 - q[i])*context_prob # s = 0 -> s = 0 w.p. 1 - q

#                 # when s = 1
#                 # action = 0
#                 all_transitions[i, 2*k + 1, 0, 1::2] = context_prob # s = 1, a = 0 -> s = 1 w.p. 1
#                 # action = 1
#                 all_transitions[i, 2*k + 1, 1, 0::2] = p[i, k]*context_prob # s = 1, a = 1 -> s = 0 w.p. p[i][k]
#                 all_transitions[i, 2*k + 1, 1, 1::2] = (1 - p[i, k])*context_prob # s = 1, a = 1 -> s = 1 w.p. 1 - p[i][k]
#         # print(context_prob)
#         # print(np.sum(all_transitions, axis=-1)[np.sum(all_transitions, axis=-1) != 1])
#         assert (np.sum(all_transitions, axis=-1) <= 1 + 1e-6).all() and (np.sum(all_transitions, axis=-1) >= 1 - 1e-6).all(), "sum_{s'} P[s'|s, a] ≠ 1, wrong!"

#         return all_transitions

#     def randomly_generate_transitions(self, N, K, homogeneous = True):
#         if homogeneous:
#             q = np.ones(N)*np.random.rand()
#             p = np.ones((N, K))*np.random.rand(K)

#         else:
#             q = np.random.rand(N)
#             p = np.random.rand(N, K)
        
#         context_prob = np.random.rand(K)
#         context_prob /= np.sum(context_prob, keepdims=True)

#         all_transitions = self.construct_volunteer_transition_matrix(N = N, K = K, q = q, context_prob=context_prob, p = p)
#         return all_transitions, context_prob
    
#     def load_instance(self, name = None):
#         pattern = f'N_{self.N}_K_{self.K}_'
#         if name == None:
#             pattern += f"seed_{self.seed}_"
#             directories = [d for d in os.listdir(self.data_dir) if d.startswith(pattern)]
#         else:
#             pattern += f"NAME_{name}_"
#             directories = [d for d in os.listdir(self.data_dir) if d.startswith(pattern)]
#         pattern += f"num_"
#         if not directories:
#             assert name is None, f"directory starting with {pattern} doesn't exist"
#             print("No existing instances found, generating new instance...")
#             self.generate_instances(1)  # Generate one instance
#             directories = [d for d in os.listdir(self.data_dir) if d.startswith(pattern)]
        
#         instance_dir = os.path.join(self.data_dir, directories[0])
#         file_path = os.path.join(instance_dir, 'instance_data.json')
#         with open(file_path, 'r') as f:
#             instance = json.load(f)
#             instance['transitions'] = np.array(instance['transitions'])
#             instance['context_prob'] = np.array(instance['context_prob'])
#         return instance
    
#     def print_instance(self, all_transitions, context_prob):
#         """
#         input: all prob. matrices
#         job: get p, q, f and preview the parameters
#         """
#         p = np.zeros((self.N, self.K))
#         q = np.zeros(self.N)

#         for i in range(self.N):
#             q[i] = np.sum(all_transitions[i, 0, 0, 1::2])
#             for k in range(self.K):
#                 p[i, k] = np.sum(all_transitions[i, k*2 + 1, 1, ::2])
        
#         if (all_transitions[0] == all_transitions[1]).all():
#             print(f"p_k:\n{np.round(p, 2)[0]}")
#             print(f"q = {np.round(q, 2)[0]}")
#             print(f"f_k = {np.round(context_prob, 2)}")

#     def get_original_vectors(self, all_transitions, context_prob):
#         """
#         input: all prob. matrices
#         job: get p, q, f and preview the parameters
#         """
#         p = np.zeros((self.N, self.K))
#         q = np.zeros(self.N)

#         for i in range(self.N):
#             # wrong version:
#             # q[i] = np.sum(all_transitions[i, 0, :, 1::2])
#             q[i] = np.sum(all_transitions[i, 0, 0, 1::2])
#             for k in range(self.K):
#                 p[i, k] = np.sum(all_transitions[i, k*2 + 1, 1, ::2])

#         return p, q, context_prob
    
#     def generate_instance_given_probs(self, p, q, context_prob, M = 1, name="placeholder"):
#         """
#         given parameters, construct transition_matrix
#         Paramters:
#             q: shape N
#             p: shape N*K
#             context_prob: shape K
#             -> n_states = 2*K
#         Return
#         """
#         N = self.N
#         K = self.K
#         seed = self.seed
#         instances = []
#         self.data_dir = self.data_dir.removeprefix(f'data/N_{N}_K_{K}_seed_{seed}')
#         self.data_dir = os.path.join(f'data/N_{N}_K_{K}_NAME_{name}', self.data_dir)
#         print(f"in generate_instance_given_probs() data_dir = {self.data_dir}")
#         for m in range(M):
#             transitions = self.construct_volunteer_transition_matrix(N=self.N, K=self.K, q = q, context_prob=context_prob, p = p)
#             instance_dir = os.path.join(self.data_dir, f'N_{N}_K_{K}_NAME_{name}_num_{m}')
#             print(f"in generate_instance_given_probs() instance-dir = {instance_dir}")
#             if not os.path.exists(instance_dir):
#                 os.makedirs(instance_dir)
            
#             instance = {
#                 'transitions': transitions.tolist(),
#                 'context_prob': context_prob.tolist()
#             }
#             instances.append(instance)
#             self.save_instance(instance, instance_dir)


# if __name__ == '__main__':

#     # sanity check for generated instances:
#     print("---- sanity check for instance generation: if q is correct ----")
#     N = 25
#     seed = 43
#     B = 5
#     K = 3
#     generator = InstanceGenerator(N=N, K=K, seed=seed)
#     generator.generate_instances()
#     instance = generator.load_instance()
#     all_transitions, context_prob = instance['transitions'], instance['context_prob']
#     p, q, _ = generator.get_original_vectors(all_transitions, context_prob)
#     print(f"p[0] = {p[0]}, q[0] = {q[0]}")

#     # this part works
#     print("---- sanity check for random method ----")
#     K = 6
#     N = 60
#     seed = 66
#     generator = InstanceGenerator(N=N, K=K, seed=seed)
#     instance = generator.load_instance()
#     print(f"Loaded Instance:\n{instance}")

#     # Simulation part using loaded instance
#     all_transitions, context_prob = instance['transitions'], instance['context_prob']
#     T = 100
#     budget = 20
#     reward_vector = np.ones(K)
#     simulator = Volunteer_RMABSimulator(N=N, K=K, T=T, context_prob=context_prob, all_transitions=all_transitions, budget=budget, reward_vector=reward_vector)
#     total_reward = 0

#     for t in range(T):
#         action = np.zeros(N)
#         selection_idx = np.random.choice(a=N, size=budget, replace=False)
#         action[selection_idx] = 1
#         action = action.astype(int)

#         states, reward, done, _ = simulator.step(action)
#         total_reward += reward

#     print('Total reward:', total_reward)
#     print('Final states:', states)

#     # test no. 2: generate almost-trivial instance:
#     # K = 2, N = 10
#     # p_i = [0.999, 0]
#     # q_i = 0.2
#     # context_prob = [0.5, 0.5]
#     print("\n\n---- sanity check for load-data method ----")
#     seed = 43
#     N = 25
#     K = 2
#     generator = InstanceGenerator(N=N, K=K, seed=seed)
#     p = np.array([[0.999, 0] for i in range(N)])
#     context_prob = np.array([0.5, 0.5])
#     q = np.ones(N)*0.2
#     generator.generate_instance_given_probs(p = p, q = q, context_prob=context_prob, name="simpleK2")
#     instance = generator.load_instance(name="simpleK2")
#     print(f"Loaded Instance:\n{instance['transitions'][0], instance['context_prob'][0]}")

#     # Simulation part using loaded instance
#     all_transitions, context_prob = instance['transitions'], instance['context_prob']
#     p, q, _ = generator.get_original_vectors(all_transitions, context_prob)
#     print("p, q = ", p, q)
#     T = 100
#     budget = 5
#     reward_vector = np.ones(K)
#     simulator = Volunteer_RMABSimulator(N=N, K=K, T=T, context_prob=context_prob, all_transitions=all_transitions, budget=budget, reward_vector=reward_vector)
#     total_reward = 0

#     for t in range(T):
#         action = np.zeros(N)
#         selection_idx = np.random.choice(a=N, size=budget, replace=False)
#         action[selection_idx] = 1
#         action = action.astype(int)

#         states, reward, done, _ = simulator.step(action)
#         total_reward += reward

#     print('Total reward:', total_reward)
#     print('Final states:', states)




    
