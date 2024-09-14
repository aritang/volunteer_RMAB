import gym
import numpy as np

class Volunteer_RMABSimulator(gym.Env):
    '''
    This simulator simulates the interaction with a set of arms (volunteers) with known transition probabilities
    
    This setup is aligned with restless multi-armed bandit problems.

    Attributes:
        N: the total number of arms in the entire population
        T: the total number of time steps per episode iteration
        K (K): the number of context
        budget: the number of arms that can be pulled in a time step

        all_transitions: context-wise transformation matrix. this is a numpy array with shape (K, N, K*2, 2, K*2)
                        state (NE-0, E-1)*[K], action (NI-0, I-1), next state (NE-0, E-1)*[K]

        reward_vector: numpy array shape(K)

        context (int): the real context at every period
        context_prob: shape-K numpy vector of context-occuring probabilities

    This simulation now supports:
        - context-wise, potentially heterogeneous volunteers

    
    '''

    def __init__(self, 
                 N, # num of arms 
                 K, # num of  context
                 T, # length of episode
                 context_prob, # shape = (K)
                 all_transitions, # shape = (N, K*2, 2, K*2)
                 budget, # B
                 reward_vector, # shape = (K)
                 initial_state = None,
                 constraint_type = "hard" # or, soft
                 ):
        '''
        Initialization

        Parameters:
            N, # num of arms 
            K, # num of  context
            T, # T
            context_prob, # shape = (K)
            all_transitions, # shape = (N, K*2, 2, K*2)
            initial_state, # numpy array of length N and 
            budget, # B
            reward_vector # shape = (K)
        '''

        self.N               = N
        self.all_transitions = all_transitions
        self.budget          = budget
        self.number_states   = 2*K
        self.T               = T
        self.initial_state   = initial_state # initial state function written later
        self.reward_vector   = reward_vector
        
        self.K = K

        # check validity of context_prob
        assert len(context_prob) == K, f"len(context_prob) ({len(context_prob)}) ≠ K ({K})"
        assert sum(context_prob) <= 1 + 1e-5 and sum(context_prob) >= 1 - 1e-5, f"sum of context_prob (f{np.round(context_prob, 2)}) ≠ 1"
        self.context_prob = context_prob

        self.context = 0
        self.timestep = 0

        self.constraint_type = constraint_type

        # assert_valid_transition(all_transitions)

        if self.initial_state is None:
            self.randomly_initialize_state()
        else:
            self.set_initial_state(self.initial_state)

    def randomly_initialize_state(self):
        self.context = np.random.choice(a=self.K, p=self.context_prob)
        self.states = np.random.choice(a=2, size = self.N, p = [0.5, 0.5])
        self.states += 2*self.context
        self.initial_state = self.states
        self.reward = 0
        self.total_budget = 0

    def set_initial_state(self, initial_state):
        # check validity of initial_state. check later
        self.states = initial_state
        self.initial_state = initial_state
        self.reward = 0
        self.total_budget = 0

    def reset_all(self, initial_state = None):
        return self.reset_instance(initial_state)

    def reset_instance(self, initial_state = None):
        # current state initialization
        self.timestep    = 0
        if initial_state:
            self.set_initial_state(initial_state)
        else:
            self.randomly_initialize_state()
        return self.observe()

    def reset(self):
        self.timestep = 0
        self.reward = 0
        self.total_budget = 0
        self.states = self.initial_state
        return self.observe()

    def is_terminal(self):
        if self.timestep >= self.T:
            return True
        else:
            return False

    def observe(self):
        return self.states

    def validify_action(self, action):
        if self.constraint_type == 'hard':
            assert len(action) == self.N
            # clip action if exceeds budget
            # should raise warning?
            if np.sum(action) > self.budget:
                # Find indices where action is 1
                active_indices = np.where(action == 1)[0]
                # If there are more active actions than the budget, choose `self.budget` indices randomly
                if len(active_indices) > self.budget:
                    # Randomly select `self.budget` indices to keep
                    keep_indices = np.random.choice(active_indices, self.budget, replace=False)
                    # Set all actions to zero
                    action[:] = 0
                    # Set the selected indices back to 1
                    action[keep_indices] = 1
        return action
    
    def record_budget(self, action):
        self.total_budget += np.sum(action)

    def report_avg_budget(self):
        if self.timestep == 0:
            return 0
        else:
            return self.total_budget/self.timestep

    def step(self, action):

        action = self.validify_action(action)
        # problem (trick of the transition)
        # when state (context) is global, in actual simulation, first the transition happens non-globally, every arm may transition to a different context
        # then we set context to be the same (by randomly drawing another lottery)
        next_states = np.zeros(self.N)
        # this iteration can be written in one line but whatever
        for i in range(self.N):
            prob = self.all_transitions[i, self.states[i], action[i], :]
            next_state = np.random.choice(a=self.number_states, p=prob)
            next_states[i] = next_state
        # global context
        self.record_budget(action)
        self.timestep += 1
        
        reward = self.get_reward(next_states, action)
        self.reward = reward

        self.states = next_states.astype(int)
        self.context = np.random.choice(a = self.K, p = self.context_prob)
        self.states = self.states%2 + 2*self.context
        done = self.is_terminal()

        # print(f'  action {action}, sum {action.sum()}, reward {reward}')

        return self.observe(), reward, done, {}
    
    def get_reward_external(self):
        return self.reward
    
    def get_original_vectors(self):
        """
        input: all prob. matrices

        return: p, q, f and preview the parameters
        """
        p = np.zeros((self.N, self.K))
        q = np.zeros(self.N)

        for i in range(self.N):
            q[i] = np.sum(self.all_transitions[i, 0, 0, 1::2])
            for k in range(self.K):
                p[i, k] = np.sum(self.all_transitions[i, k*2 + 1, 1, ::2])
        return p, q, self.context_prob

    def get_reward(self, next_states, action):
        """
        Computes the reward based on the current states, actions, and next states.
        
        Parameters:
            next_states (np.array): The states of the arms in the next time step.
            action (np.array): The actions taken at the current time step.
        
        Returns:
            float: The total reward computed.
        """
        # Initialize the reward
        total_reward = 0
        activeness_now = self.states%2
        activeness_later = next_states%2

        # Loop over each arm/index
        for i in range(len(action)):
            # Check if the conditions for receiving a reward are met
            if activeness_now[i] == 1 and action[i] == 1 and activeness_later[i] == 0:
                total_reward += self.reward_vector[int(self.context)]
        return total_reward


def random_transition(K, N, n_actions):
    n_states = K*2
    all_transitions = np.random.random((N, n_states, n_actions, n_states))
    all_transitions = all_transitions / np.sum(all_transitions, axis=-1, keepdims=True)
    return all_transitions


def construct_volunteer_transition_matrix(N, K, q, context_prob, p, n_actions = 2):
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

def randomly_generate_transitions(N, K, homogeneous = True):
    if homogeneous:
        q = np.ones(N)*np.random.rand()
        p = np.ones((N, K))*np.random.rand(K)

    else:
        q = np.random.rand(N)
        p = np.random.rand(N, K)
    
    context_prob = np.random.rand(K)
    context_prob /= np.sum(context_prob, keepdims=True)

    all_transitions = construct_volunteer_transition_matrix(N, K, q = q, context_prob=context_prob, p = p)
    return all_transitions, context_prob


'''
Testing the functionality of the simulator
'''
if __name__ == '__main__':
    N  = 100
    K = 3
    all_transitions, context_prob = randomly_generate_transitions(N, K, homogeneous = True)

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

