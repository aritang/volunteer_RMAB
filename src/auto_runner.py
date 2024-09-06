import os
import json
import subprocess

def main():
    # Path to the results directory
    results_path = "./results"
    
    # Walk through each directory in the results directory
    for root, dirs, files in os.walk(results_path):
        print(files)
        if 'param_settings.json' in files:
            # Construct the full path to param_settings.json
            json_path = os.path.join(root, 'param_settings.json')
            
            # Read the JSON file
            with open(json_path, 'r') as f:
                params = json.load(f)
            
            # Extract the necessary parameters
            n_arms = params.get('n_arms', '')
            budget = params.get('budget', '')
            num_context = params.get('num_context', '')
            seed = params.get('seed', '')
            homogeneous = params.get('homogeneous', '')
            
            # Convert boolean to command line argument format
            homo_arg = 'true' if homogeneous else 'false'
            
            # Command to run
            command = [
                'python', 'bugdet_allocation_solver.py',
                '-N', str(n_arms),
                '-B', str(budget),
                '-K', str(num_context),
                '--seed', str(seed),
                '-HOMO', homo_arg
            ]
            
            # Print the command for debugging purposes
            print("Running command:", " ".join(command))
            
            # Execute the command
            subprocess.run(command)

if __name__ == "__main__":
    main()
