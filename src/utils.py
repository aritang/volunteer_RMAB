import numpy as np
import argparse
import random
import time, datetime
import sys, os

class Memoizer:
    """ improve performance of memoizing solutions (to QP and WI value iteration) """
    def __init__(self, method):
        self.method = method
        self.solved_p_vals = {}

    def to_key(self, input1, input2):
        """ convert inputs to a key

        QP: inputs: LCB and UCB transition probabilities
        UCB and extreme: inputs - estimated transition probabilities and initial state s0 """
        if self.method in ['lcb_ucb', 'QP', 'QP-min']:
            lcb, ucb = input1, input2
            p_key = (np.round(lcb, 4).tobytes(), np.round(ucb, 4).tobytes())
        elif self.method in ['p_s', 'optimal', 'UCB', 'extreme', 'ucw_value']:
            transitions, state = input1, input2
            p_key = (np.round(transitions, 4).tobytes(), state)
        elif self.method in ['lcb_ucb_s_lamb']:
            lcb, ucb = input1
            s, lamb_val = input2
            p_key = (np.round(lcb, 4).tobytes(), np.round(ucb, 4).tobytes(), s, lamb_val)
        else:
            raise Exception(f'method {self.method} not implemented')

        return p_key

    def check_set(self, input1, input2):
        p_key = self.to_key(input1, input2)
        if p_key in self.solved_p_vals:
            return self.solved_p_vals[p_key]
        return -1

    def add_set(self, input1, input2, wi):
        p_key = self.to_key(input1, input2)
        self.solved_p_vals[p_key] = wi

def parse_arguments():
    """
    Parses command-line arguments provided to the script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_arms',         '-N', help='num beneficiaries (arms)', type=int, default=25)
    parser.add_argument('--budget',         '-B', help='budget', type=int, default=5)
    parser.add_argument('--num_context',    '-K', help='context_size', type=int, default='3')

    parser.add_argument('--episode_len',    '-T', help='episode length', type=int, default=600)
    parser.add_argument('--n_episodes',     '-H', help='num episodes', type=int, default=6)
    parser.add_argument('--data',           '-D', help='dataset to use {synthetic, real, local_generated}', type=str, default='local_generated')

    parser.add_argument('--data_name',           '-DN', help='name of specified dataset', type=str, default=None)

    parser.add_argument('--n_epochs',       '-E', help='number of epochs (num_repeats)', type=int, default=20)
    parser.add_argument('--discount',       '-d', help='discount factor', type=float, default=0.98)

    # doesn't seem necessary
    parser.add_argument('--alpha',          '-a', help='alpha: for conf radius', type=float, default=3)

    # parser.add_argument('--n_states',       '-S', help='num states', type=int, default=2)
    parser.add_argument('--n_actions',      '-A', help='num actions', type=int, default=2)
    # special treatment
    parser.add_argument('--homogeneous', '-HOMO', help='if homogenous', type=str, default='True')
    parser.add_argument('--seed',           '-s', help='random seed', type=int, default=43)
    parser.add_argument('--verbose',        '-V', type=bool, help='if True, then verbose output (default False)', default=False)
    parser.add_argument('--local',          '-L', help='if True, running locally (default False)', action='store_true')
    parser.add_argument('--prefix',         '-p', help='prefix for file writing', type=str, default='')
    parser.add_argument('--name',         '-NAME', help='experiment name', type=str, default='')

    args = parser.parse_args()
    # special treatement
    args.homogeneous = args.homogeneous.lower() in ['true', '1', 'yes']


    args.exp_name_out = f'{args.data}_N{args.n_arms}_B{args.budget}_K{args.num_context}_T{args.episode_len*args.n_episodes}_epochs{args.n_epochs}'

    args.str_time = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')

    args.N = args.n_arms
    args.K = args.num_context

    if not os.path.exists(f'results/{args.data}'):
        os.makedirs(f'results/{args.data}')
    return args
