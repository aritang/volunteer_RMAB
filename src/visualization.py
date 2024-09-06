import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

colors   = {'whittle': 'purple', 'ucw_value': 'b', 'ucw_qp': 'c', 'ucw_qp_min': 'goldenrod', 'ucw_ucb': 'darkorange',
                'ucw_extreme': 'r', 'wiql': 'limegreen', 'random': 'brown', 'type_specific' : 'goldenrod'}

def smooth(rewards, weight=0.7):
    """ smoothed exponential moving average """
    prev = rewards[0]
    smoothed = np.zeros(len(rewards))
    for i, val in enumerate(rewards):
        smoothed_val = prev * weight + (1 - weight) * val
        smoothed[i] = smoothed_val
        prev = smoothed_val

    return smoothed

def get_cum_sum(reward):
    cum_sum = reward.cumsum(axis=1).mean(axis=0)
    cum_sum = cum_sum / (np.arange(len(cum_sum)) + 1)
    return smooth(cum_sum)

def plot_rewards(rewards, use_algos, args, colors = colors):
    # plot average cumulative reward
    this_path = f'./results/{args.str_time}'

    if not os.path.exists(f'./results/{args.str_time}'):
        os.makedirs(f'./results/{args.str_time}')
    
    plt.figure()
    for algo in use_algos:
        plt.plot(get_cum_sum(rewards[algo]), c=colors[algo], label=algo)
    plt.legend()
    plt.xlabel(f'timestep $t$ ({args.n_episodes} episodes of length {args.episode_len})')
    plt.ylabel('average cumulative reward')
    plt.title(f'{args.data} - N={args.n_arms}, B={args.budget}, discount={args.discount}, {args.n_epochs} epochs')

    plt.savefig(this_path + f'/cum_reward_{args.exp_name_out}.pdf')

    # plot average reward
    plt.figure()
    for algo in use_algos:
        plt.plot(smooth(rewards[algo].mean(axis=0)), c=colors[algo], label=algo)
    plt.legend()
    plt.xlabel(f'timestep $t$ ({args.n_episodes} episodes of length {args.episode_len})')
    plt.ylabel('average reward')
    plt.title(f'{args.data} - N={args.n_arms}, budget={args.budget}, discount={args.discount}, {args.n_epochs} epochs')
    # plt.savefig(f'results/{args.data}/avg_reward_{args.exp_name_out}_{args.str_time}.pdf')
    plt.savefig(this_path + f'/avg_reward_{args.exp_name_out}.pdf')

import numpy as np
import matplotlib.pyplot as plt

def plot_type_tuple(reward, context_prob, args):
    """
    Plots a series of trade-off graphs between budget allocations for different contexts.

    Args:
        reward (dict): A dictionary mapping tuples of budget allocations to rewards.
        context_prob (list): A list of probabilities associated with each context.
        B (float): Total budget available.

    Returns:
        None, but displays a grid of contour plots showing the trade-offs between pairs of contexts.
    """
    n = len(context_prob)  # Number of contexts
    fig, axes = plt.subplots(n, n, figsize=(15, 15))
    this_path = f'./results/{args.str_time}'

    if not os.path.exists(f'./results/{args.str_time}'):
        os.makedirs(f'./results/{args.str_time}')

    for i in range(n):
        for j in range(n):
            if i != j:
                max_i = int(args.budget / context_prob[i])
                max_j = int(args.budget / context_prob[j])
                reward_to_plot = (-1)*np.ones((max_i, max_j))

                # Populate reward_to_plot matrix
                for B_i in range(max_i):
                    for B_j in range(max_j):
                        if context_prob[i] * B_i + context_prob[j] * B_j <= args.budget:
                            # Finding the maximum reward for the current (B_i, B_j)
                            max_reward = 0
                            for key, val in reward.items():
                                if len(key) > i and len(key) > j and int(key[i]) == B_i and int(key[j]) == B_j:
                                    if val > max_reward:
                                        max_reward = val
                            reward_to_plot[B_i, B_j] = max_reward

                # Plot the contour graph for context i vs context j
                ax = axes[i, j]
                CS = ax.contourf(reward_to_plot, levels=50, cmap='viridis')
                ax.set_title(f'Context {i+1} vs Context {j+1}')
                ax.set_xlabel(f'Budget for context {i+1}')
                ax.set_ylabel(f'Budget for context {j+1}')
                fig.colorbar(CS, ax=ax, orientation='vertical')

    for i in range(n):
        max_i = int(args.budget / context_prob[i])
        rewards = np.full(max_i, -1.0)
        for B_i in range(max_i):
            max_reward = 0
            for key, val in reward.items():
                if len(key) > i and int(key[i]) == B_i:
                    if val > max_reward:
                        max_reward = val
            rewards[B_i] = max_reward
        ax = axes[i, i]
        ax.plot(range(max_i), rewards, '-o')
        ax.set_title(f'Rewards for context {i+1}')
        ax.set_xlabel(f'Budget for context {i+1}')
        ax.set_ylabel('Reward')

    plt.tight_layout()
    plt.savefig(this_path + f'/budget_rewards-{args.exp_name_out}.pdf')
    


if __name__ == "__main__":
    filename = 'results/local_generated/reward_local_generated_n100_b20_s4_a2_H357_L6_epochs6_type_specific.csv'
    rewards = pd.read_csv(filename).to_numpy()

