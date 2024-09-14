# instance_generator.py
# both instance generator and loader

import json
import numpy as np
import os
import random
from volunteer_simulator import Volunteer_RMABSimulator

class InstanceGenerator:
    def __init__(self, N, K, seed=66, data_dir='data'):
        self.N = N
        self.K = K
        self.seed = seed
        self.data_dir = os.path.join(data_dir, f'N_{N}_K_{K}_seed_{seed}')
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def generate_instances(self, M = 1, homogeneous = True):
        instances = []
        for m in range(M):
            transitions, context_prob = self.randomly_generate_transitions(N=self.N, K=self.K, homogeneous=homogeneous)
            instance_dir = os.path.join(self.data_dir, f'N_{self.N}_K_{self.K}_seed_{self.seed}_num_{m}')
            if not os.path.exists(instance_dir):
                os.makedirs(instance_dir)
            
            instance = {
                'transitions': transitions.tolist(),
                'context_prob': context_prob.tolist()
            }
            instances.append(instance)
            self.save_instance(instance, instance_dir)


    def save_instance(self, instance, instance_dir):
        file_path = os.path.join(instance_dir, 'instance_data.json')
        with open(file_path, 'w') as f:
            json.dump(instance, f, indent=4)
        print(f"{file_path} saved successfully")

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
    
    def load_instance(self, name = None):
        pattern = f'N_{self.N}_K_{self.K}_'
        if name == None:
            pattern += f"seed_{self.seed}_"
            directories = [d for d in os.listdir(self.data_dir) if d.startswith(pattern)]
        else:
            pattern += f"NAME_{name}_"
            directories = [d for d in os.listdir(self.data_dir) if d.startswith(pattern)]
        pattern += f"num_"
        if not directories:
            assert name is None, f"directory starting with {pattern} doesn't exist"
            print("No existing instances found, generating new instance...")
            self.generate_instances(1)  # Generate one instance
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
        input: all prob. matrices
        job: get p, q, f and preview the parameters
        """
        p = np.zeros((self.N, self.K))
        q = np.zeros(self.N)

        for i in range(self.N):
            q[i] = np.sum(all_transitions[i, 0, 0, 1::2])
            for k in range(self.K):
                p[i, k] = np.sum(all_transitions[i, k*2 + 1, 1, ::2])
        
        if (all_transitions[0] == all_transitions[1]).all():
            print(f"p_k:\n{np.round(p, 2)[0]}")
            print(f"q = {np.round(q, 2)[0]}")
            print(f"f_k = {np.round(context_prob, 2)}")

    def get_original_vectors(self, all_transitions, context_prob):
        """
        input: all prob. matrices
        job: get p, q, f and preview the parameters
        """
        p = np.zeros((self.N, self.K))
        q = np.zeros(self.N)

        for i in range(self.N):
            # wrong version:
            # q[i] = np.sum(all_transitions[i, 0, :, 1::2])
            q[i] = np.sum(all_transitions[i, 0, 0, 1::2])
            for k in range(self.K):
                p[i, k] = np.sum(all_transitions[i, k*2 + 1, 1, ::2])

        return p, q, context_prob
    
    def generate_instance_given_probs(self, p, q, context_prob, M = 1, name="placeholder"):
        """
        given parameters, construct transition_matrix
        Paramters:
            q: shape N
            p: shape N*K
            context_prob: shape K
            -> n_states = 2*K
        Return
        """
        N = self.N
        K = self.K
        seed = self.seed
        instances = []
        self.data_dir = self.data_dir.removeprefix(f'data/N_{N}_K_{K}_seed_{seed}')
        self.data_dir = os.path.join(f'data/N_{N}_K_{K}_NAME_{name}', self.data_dir)
        print(f"in generate_instance_given_probs() data_dir = {self.data_dir}")
        for m in range(M):
            transitions = self.construct_volunteer_transition_matrix(N=self.N, K=self.K, q = q, context_prob=context_prob, p = p)
            instance_dir = os.path.join(self.data_dir, f'N_{N}_K_{K}_NAME_{name}_num_{m}')
            print(f"in generate_instance_given_probs() instance-dir = {instance_dir}")
            if not os.path.exists(instance_dir):
                os.makedirs(instance_dir)
            
            instance = {
                'transitions': transitions.tolist(),
                'context_prob': context_prob.tolist()
            }
            instances.append(instance)
            self.save_instance(instance, instance_dir)


if __name__ == '__main__':

    # sanity check for generated instances:
    print("---- sanity check for instance generation: if q is correct ----")
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

    # this part works
    print("---- sanity check for random method ----")
    K = 6
    N = 60
    seed = 66
    generator = InstanceGenerator(N=N, K=K, seed=seed)
    instance = generator.load_instance()
    print(f"Loaded Instance:\n{instance}")

    # Simulation part using loaded instance
    all_transitions, context_prob = instance['transitions'], instance['context_prob']
    T = 100
    budget = 20
    reward_vector = np.ones(K)
    simulator = Volunteer_RMABSimulator(N=N, K=K, T=T, context_prob=context_prob, all_transitions=all_transitions, budget=budget, reward_vector=reward_vector)
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

    # test no. 2: generate almost-trivial instance:
    # K = 2, N = 10
    # p_i = [0.999, 0]
    # q_i = 0.2
    # context_prob = [0.5, 0.5]
    print("\n\n---- sanity check for load-data method ----")
    seed = 43
    N = 25
    K = 2
    generator = InstanceGenerator(N=N, K=K, seed=seed)
    p = np.array([[0.999, 0] for i in range(N)])
    context_prob = np.array([0.5, 0.5])
    q = np.ones(N)*0.2
    generator.generate_instance_given_probs(p = p, q = q, context_prob=context_prob, name="simpleK2")
    instance = generator.load_instance(name="simpleK2")
    print(f"Loaded Instance:\n{instance['transitions'][0], instance['context_prob'][0]}")

    # Simulation part using loaded instance
    all_transitions, context_prob = instance['transitions'], instance['context_prob']
    p, q, _ = generator.get_original_vectors(all_transitions, context_prob)
    print("p, q = ", p, q)
    T = 100
    budget = 5
    reward_vector = np.ones(K)
    simulator = Volunteer_RMABSimulator(N=N, K=K, T=T, context_prob=context_prob, all_transitions=all_transitions, budget=budget, reward_vector=reward_vector)
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




    
