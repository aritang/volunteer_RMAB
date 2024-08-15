# instance_generator.py
# both instance generator and loader

import json
import numpy as np
import os
import random
from volunteer_simulator import Volunteer_RMABSimulator

class InstanceGenerator:
    def __init__(self, N, K, seed=66, data_dir = 'data'):
        """
        Initialize the InstanceGenerator
        with the number of arms (N), 
        context size (K), 
        and an optional random seed.
        """
        self.N = N
        self.K = K
        self.seed = seed
        self.data_dir = data_dir
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def generate_instances(self, M):
        """
        Generate M instances with transitions and context probabilities, storing them in JSON files.
        """
        instances = []
        for _ in range(M):
            transitions, context_prob = self.randomly_generate_transitions(N = self.N, K = self.K)
            instance = {
                'transitions': transitions.tolist(),
                'context_prob': context_prob.tolist()
            }
            instances.append(instance)

        # Saving instances to JSON files
        for i, instance in enumerate(instances):
            file_path = os.path.join(self.data_dir, f'transition_instance_{i}.json')
            with open(file_path, 'w') as f:
                json.dump(instance, f, indent=4)

    def load_instances(self, M):
        """
        Load M instances from JSON files located in a specific directory into memory.
        """
        instances = []
        for i in range(M):
            file_path = os.path.join(self.data_dir, f'transition_instance_{i}.json')
            with open(file_path, 'r') as f:
                instance = json.load(f)
                instance['transitions'] = np.array(instance['transitions'])
                instance['context_prob'] = np.array(instance['context_prob'])
                instances.append(instance)
        return instances

    def construct_volunteer_transition_matrix(self, N, K, q, context_prob, p, n_actions = 2):
        """
        given parameters, construct transition_matrix
        Paramters:
            q: shape N
            p: shape N*K
            context_prob: shape K
            -> n_states = 2*K
        Return
            transition_matrix: shape N*n_states*2*n_states
        """
        n_states = K*2
        all_transitions = np.zeros((N, n_states, n_actions, n_states))
        for k in range(K):
            for i in range(N):
                # when s = 0
                # action = 0
                all_transitions[i, 2*k, 0, 1::2] = q[i]*context_prob # s = 0 -> s = 1 w.p. q
                all_transitions[i, 2*k, 0, 0::2] = (1 - q[i])*context_prob # s = 0 -> s = 0 w.p. 1 - q
                # action = 1
                all_transitions[i, 2*k, 1, 1::2] = q[i]*context_prob # s = 0 -> s = 1 w.p. q
                all_transitions[i, 2*k, 1, 0::2] = (1 - q[i])*context_prob # s = 0 -> s = 0 w.p. 1 - q

                # when s = 1
                # action = 0
                all_transitions[i, 2*k + 1, 0, 1::2] = context_prob # s = 1, a = 0 -> s = 1 w.p. 1
                # action = 1
                all_transitions[i, 2*k + 1, 1, 0::2] = p[i, k]*context_prob # s = 1, a = 1 -> s = 0 w.p. p[i][k]
                all_transitions[i, 2*k + 1, 1, 1::2] = (1 - p[i, k])*context_prob # s = 1, a = 1 -> s = 1 w.p. 1 - p[i][k]
        # print(context_prob)
        # print(np.sum(all_transitions, axis=-1)[np.sum(all_transitions, axis=-1) != 1])
        assert (np.sum(all_transitions, axis=-1) <= 1 + 1e-6).all() and (np.sum(all_transitions, axis=-1) >= 1 - 1e-6).all(), "sum_{s'} P[s'|s, a] ≠ 1, wrong!"

        return all_transitions

    def randomly_generate_transitions(self, N, K, homogeneous = True):
        if homogeneous:
            q = np.ones(N)*np.random.rand()
            p = np.ones((N, K))*np.random.rand(K)

        else:
            q = np.random.rand(N)
            p = np.random.rand(N, K)
        
        context_prob = np.random.rand(K)
        context_prob /= np.sum(context_prob, keepdims=True)

        all_transitions = self.construct_volunteer_transition_matrix(N = N, K = K, q = q, context_prob=context_prob, p = p)
        return all_transitions, context_prob

if __name__ == '__main__':
    K = 6
    N = 60
    generator = InstanceGenerator(N=N, K=K, seed=66)
    generator.generate_instances(1)
    instances = generator.load_instances(1)
    print(f"created and loaded {len(instances)} Instances:")
    
    # use the first instance to construct an environment
    all_transitions, context_prob = instances[0]['transitions'], instances[0]['context_prob']
    
    T = 100
    budget = 20
    reward_vector = np.ones(K)
    simulator = Volunteer_RMABSimulator(N = N, K = K, T = T, context_prob=context_prob, all_transitions=all_transitions, budget=budget, reward_vector=np.ones(K))
    total_reward = 0

    for t in range(T):
        action = np.zeros(N)
        selection_idx = np.random.choice(a=N, size=budget, replace=False)
        action[selection_idx] = 1
        action = action.astype(int)

        states, reward, done, _ = simulator.step(action)
        total_reward += reward

    print('total reward: {}'.format(total_reward))
    print(states)

    
