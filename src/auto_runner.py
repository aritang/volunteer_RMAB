import os
import json
import subprocess

"""
auto_runner.py

This script automates the execution of the 'budget_allocation_solver.py' script across multiple experimental setups.
It scans the './results' directory for subdirectories containing 'param_settings.json' files,
extracts necessary parameters from each JSON file, and runs the solver script with those parameters.

Usage:
    python auto_runner.py
"""

def main():
    """
    Automate running the budget allocation solver for multiple experiment configurations.

    Process:
    - Traverse the './results' directory recursively to find all 'param_settings.json' files.
    - For each JSON file found:
        - Load experiment parameters such as number of arms, budget, number of contexts, seed, and homogeneity.
        - Construct a command to execute 'budget_allocation_solver.py' with the extracted parameters.
        - Run the constructed command as a subprocess.
    
    Inputs:
        None

    Outputs:
        None

    Remarks:
        - Executes 'budget_allocation_solver.py' multiple times with different parameters.
        - Prints progress and debugging information to the console.
        - May raise exceptions if subprocess execution fails.
    """
    # Path to the results directory
    results_path = "./results"
    
    # Walk through each directory in the results directory
    for root, dirs, files in os.walk(results_path):
        # Debug: print the list of files in the current directory
        print(f"Checking directory: {root}")
        print(f"Files: {files}\n")
        
        # Check if 'param_settings.json' exists in the current directory
        if 'param_settings.json' in files:
            # Construct the full path to 'param_settings.json'
            json_path = os.path.join(root, 'param_settings.json')
            
            # Read the JSON file
            with open(json_path, 'r') as f:
                params = json.load(f)
            
            try:
                # Extract and validate the necessary parameters
                n_arms = params['n_arms']
                budget = params['budget']
                num_context = params['num_context']
                seed = params['seed']
                homogeneous = params['homogeneous']
            except KeyError as e:
                print(f"Missing parameter {e} in {json_path}")
                continue  # Skip this iteration if a parameter is missing
            
            # Convert boolean to command line argument format
            homo_arg = 'true' if homogeneous else 'false'
            
            # Command to run
            command = [
                'python', 'budget_allocation_solver.py',
                '-N', str(n_arms),
                '-B', str(budget),
                '-K', str(num_context),
                '--seed', str(seed),
                '-HOMO', homo_arg
            ]
            
            # Print the command for debugging purposes
            print("Running command:", " ".join(command))
            
            # Execute the command
            subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
