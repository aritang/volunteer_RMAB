import numpy as np
import pandas as pd
import random
import time, datetime
import sys, os
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

def save_rewards(rewards, use_algos, args, this_path):
    file_list = []
    file_desciptions = []
    for algo in use_algos:    
        x_vals = np.arange(args.n_episodes * args.episode_len + 1)

        data_df = pd.DataFrame(data=rewards[algo], columns=x_vals)

        out_df = data_df
        data_df = pd.DataFrame(data=rewards[algo], columns=x_vals)

        filename = this_path + f'/reward_{args.exp_name_out}_{algo}.csv'
        file_list.append(filename)
        file_desciptions.append(algo + f" reward at each time point")

        with open(filename, 'a') as f:
            # write header, if file doesn't exist
            if f.tell() == 0:
                print(f'creating file {filename}')
                out_df.to_csv(f)

            # write results (appending) and no header
            else:
                print(f'appending to file {filename}')
                out_df.to_csv(f, mode='a', header=False)

    return file_list, file_desciptions


def write_result(rewards, use_algos, args, transition_probabilities, context_prob, p, q, rewards_to_write, best_allocation = None):
    # do two jobs: 
    # first, create a folder to store all the results
    # folder is just experiment time
    if not os.path.exists(f'./results/{args.str_time}'):
        os.makedirs(f'./results/{args.str_time}')

    # everything will be saved under this directory
    this_path = f'./results/{args.str_time}'

    # then
    # save any kinds of results (call functions that do the job), for the saved results, need two parameters:
    # (1) file location
    # (2) description

    # lastly, save the json file of all the parameters 
    if rewards == None:
        args_dict = vars(args)
    else:
        file_list, file_descriptions = save_rewards(rewards = rewards, use_algos = use_algos, args = args, this_path=this_path)
        args_dict = vars(args)
        for file_name, file_description in zip(file_list, file_descriptions):
            args_dict[file_name] = file_description
    
    if args.homogeneous == True:
        args_dict["p"] = p[0].tolist()
        args_dict["q"] = q[0].tolist()
    else:
        args_dict["p"] = p.tolist()
        args_dict["q"] = q.tolist()
    args_dict["context_prob"] = context_prob.tolist()
    args_dict["rewards"] = rewards_to_write

    args_dict["transition_probabilities"] = transition_probabilities.tolist()
    args_dict["best_allocation"] = np.array(best_allocation).tolist()

    # Serialize dictionary to JSON
    json_filename = this_path + '/param_settings.json'
    with open(json_filename, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

