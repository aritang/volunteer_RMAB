import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random

colors = {'whittle': 'purple', 'global_context_SIMULATION-uniform_budget_allocation': 'b', 'global_context_LDS': 'c', 'soft_budget_FIXED_occupancy_measure': 'goldenrod', 
          'independent_context_SIMULATION': 'darkorange', 'ucw_extreme': 'r', 'soft_budget_occupancy_measure': 'limegreen', 
          'random': 'brown', 'type_specific' : 'goldenrod'}

def random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

line_styles = ['-', '-.', '--', ':', (0, (3, 1, 1, 1)), (0, (5, 10)), (0, (5, 1)), (0, (3, 5, 1, 5)), (0, (1, 10))]

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

def plot_rewards(rewards, use_algos, args, colors=colors, line_styles=line_styles):
    # plot average cumulative reward
    this_path = f'./results/{args.str_time}'

    if not os.path.exists(this_path):
        os.makedirs(this_path)
    
    # Plot average cumulative reward
    plt.figure(figsize=(16, 12))
    for idx, algo in enumerate(use_algos):
        color = colors.get(algo, random_color())  # Use defined color or generate a random one
        plt.plot(get_cum_sum(rewards[algo]), color=color, linestyle=line_styles[idx % len(line_styles)], label=algo, linewidth=2)
    plt.legend()
    plt.xlabel(f'Timestep $t$ ({args.n_episodes} episodes of length {args.episode_len})')
    plt.ylabel('Average cumulative reward')
    plt.title(f'{args.data} - N={args.n_arms}, B={args.budget}, discount={args.discount}, {args.n_epochs} epochs')

    plt.savefig(this_path + f'/cum_reward_{args.exp_name_out}.pdf')

    # Plot average reward
    plt.figure(figsize=(16, 12))
    for idx, algo in enumerate(use_algos):
        color = colors.get(algo, random_color())  # Use defined color or generate a random one
        plt.plot(smooth(rewards[algo].mean(axis=0)), color=color, linestyle=line_styles[idx % len(line_styles)], label=algo, linewidth=2)
    plt.legend()
    plt.xlabel(f'Timestep $t$ ({args.n_episodes} episodes of length {args.episode_len})')
    plt.ylabel('Average reward')
    plt.title(f'{args.data} - N={args.n_arms}, budget={args.budget}, discount={args.discount}, {args.n_epochs} epochs')

    plt.savefig(this_path + f'/avg_reward_{args.exp_name_out}.pdf')

def plot_rewards_over_N(rewards, use_algos, args, colors=colors, line_styles=line_styles):
    # plot average cumulative reward
    this_path = f'./results/{args.str_time}'

    if not os.path.exists(this_path):
        os.makedirs(this_path)
    
    plt.figure(figsize=(16, 12))
    for idx, algo in enumerate(use_algos):
        color = colors.get(algo, random_color())
        plt.plot(rewards[algo], color=color, linestyle=line_styles[idx % len(line_styles)], label=algo, linewidth=2)
    
    plt.legend()
    plt.xlabel(f'Num of arms $N$ ({args.N0} times {args.size})')
    plt.ylabel('Average cumulative reward (over N and T)')
    plt.title(f'{args.data} - N={args.n_arms}, B={args.budget}, {args.n_epochs} epochs')

    plt.savefig(this_path + f'/cum_reward_{args.exp_name_out}.pdf')

def plot_pareto_frontier(theta_list, total_rewards, args, colors=None, line_styles=None):
    """
    Plots the Pareto frontier of theta (fairness lower bound) vs. total reward.

    Args:
        theta_list (list or numpy.ndarray): List of theta values.
        total_rewards (list or numpy.ndarray): Corresponding total rewards.
        args (argparse.Namespace): Parsed command-line arguments containing experiment configurations.
        colors (dict, optional): Dictionary mapping algorithms to colors.
        line_styles (list, optional): List of line styles for plotting.
    """
    # Ensure colors and line_styles are provided
    if colors is None:
        colors = {}
    if line_styles is None:
        line_styles = ['-']

    # Filter out NaN values for plotting
    theta_list = np.array(theta_list)
    total_rewards = np.array(total_rewards)
    valid_indices = ~np.isnan(total_rewards)
    theta_list_valid = theta_list[valid_indices]
    total_rewards_valid = total_rewards[valid_indices]

    # Plot the Pareto frontier
    plt.figure(figsize=(8, 6))
    color = colors.get('pareto_frontier', 'pink')  # Default color is blue
    linestyle = line_styles[0]  # Use the first line style by default
    plt.plot(theta_list_valid, total_rewards_valid, color=color, linestyle=linestyle,
             marker='o', linewidth=2, label='Pareto Frontier')

    # Set labels and title
    plt.xlabel('Theta (Fairness Lower Bound)', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title(f'Pareto Frontier - N={args.N}, B={args.budget}, K={args.K}', fontsize=14)
    plt.legend()
    plt.grid(True)

    
    # Create the results directory if it doesn't exist
    this_path = f'./results/{args.str_time}'
    if not os.path.exists(this_path):
        os.makedirs(this_path)
    # Save the plot
    plot_filename = os.path.join(this_path, f'pareto_frontier_{args.exp_name_out}.pdf')
    plt.savefig(plot_filename)
    plt.close()  # Close the figure to free memory

    print(f"Pareto frontier plot saved to {plot_filename}")


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
    
def MIP_n_SIM_plot_result(MIP_rewards, SIM_rewards, args, p, q, context_prob, result_name = ""):

    n = len(context_prob)  # Number of contexts
    fig, axes = plt.subplots(1, n, figsize=(n*5, 5))
    this_path = f'./results/{args.str_time}' + result_name

    # MIP reward:
    reward = MIP_rewards
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
        ax = axes[i]
        ax.plot(range(max_i), rewards, '-o', label = "MIP results")
        ax.set_title(f'Max Rewards for context {i+1}')
        ax.set_xlabel(f'Budget for context {i+1}')
        ax.set_ylabel('Reward')

    # MIP reward:
    reward = SIM_rewards
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
        ax = axes[i]
        ax.plot(range(max_i), rewards, '-o', label = "SIM results")
        ax.set_title(f'Max Rewards for context {i+1}')
        ax.set_xlabel(f'Budget for context {i+1}')
        ax.set_ylabel('Reward')
    
    plt.tight_layout()
    plt.savefig(this_path + f'/budget_rewards-{args.exp_name_out}.pdf')

    
def LDS_n_SIM_plot_result(LDS_rewards, SIM_rewards, args, p, q, context_prob, result_name = ""):

    n = len(context_prob)  # Number of contexts
    fig, axes = plt.subplots(1, n, figsize=(n*5, 5))
    this_path = f'./results/{args.str_time}' + result_name

    # MIP reward:
    reward = LDS_rewards
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
        ax = axes[i]
        ax.plot(range(max_i), rewards, '-o', label = "LDS results")
        ax.set_title(f'Total Rewards for context {i+1}')
        ax.set_xlabel(f'Budget for context {i+1}')
        ax.set_ylabel('Reward')

    # MIP reward:
    reward = SIM_rewards
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
        ax = axes[i]
        ax.plot(range(max_i), rewards, '-', label = "SIM results")
        ax.set_title(f'Total Rewards for context {i+1}')
        ax.set_xlabel(f'Budget for context {i+1}')
        ax.set_ylabel('Reward')
    
    plt.tight_layout()
    plt.legend()
    plt.savefig(this_path + f'/SIMvsLDS_budget_rewards-{args.exp_name_out}.pdf')


if __name__ == "__main__":
    filename = 'results/local_generated/reward_local_generated_n100_b20_s4_a2_H357_L6_epochs6_type_specific.csv'
    rewards = pd.read_csv(filename).to_numpy()

