# Online Learning for Restless Bandits

This code implements and evaluates algorithms for our global-context volunteer RMAB model.

Results are stored in folders in `results` folder. Every date-time folder is a run of experiment on a certain instance. All parameters and run results (rewards for different policies) are stored in `param_whatever.json` files. Plottings are in `figures` folder.

## Files

- `main.py` - main driver. Run

    ```bash
     python main.py --seed 313 -B 4 -N 20 -K 3 -HOMO true
    ```

- `load_to_run.py`. Load local `.json` parameter file to run an instance.

    ```bash
    python load_to_run.py results/19-08-2024_12:53:31/param_settings.json
    ```

- (auxiliary) `instance_generator.py` - initialize with `N`, `K` and generate transition probability and task-occuring probabilities with a fix seed. (for experiment consistency). Data stored in file 'data'.

- (environment) `volunteer_simulator.py` - simulator based on OpenAI gym, implement the environement.

    support 

    - hard budget constraint (would clip action if exceeds budget)

    - soft budget constraint (record average used budget)

- (**algorithm**)`brute_search_budget_allocation.py` - search for optimal budget, given input parameters.

    f[k] probability of context

    sum_k f[k]B[k] <= B

- (**algorithm**) `volunteer_algorithms.py` - implementing algorithms that interact with environment 

    `whittle`: (under hard budget constraint) choose top-B arms.

    `random`: (under hard budget constraint) randomly choose $B$ arms to pull.

    `whittle_policy_type_specific()`, take `type_specific_budget` as input and choose top-B[k] arms.

- (*auxiliary*)`volunteer_compute_whittle.py` - compute Whittle index for Whittle index threshold policy

- (*auxiliary*)`utils.py` - utilities across all files, almost unchanged.

    ***Problem***: hadn't specified rules for storing for different instances. Everytime used, instance is **regenerated**.

- (*auxiliary*) `result_recorder.py` - store all experiment parameters, experiment results and pointer to plotted figures.

- (*auxiiliary*) `visualization.py` - visualization funcions 


## Usage

To execute the code and run experiments, comparing baselines (`whittle` and `random`) and (brutally searched optimal) `type-specific`, run

```sh
 python main.py --seed 313 -B 4 -N 20 -K 3 -HOMO true
```

`N`, `K`, `B` are arm-num/context-size of the RMAB instance. `B` is budget.

Other configurations:

- `--verbose 1` print transition probabilities and the search process
- `--episode_len` or `-H` - corresponds to the time horizon. Default: T = 357.
- `--n_episodes` - for every instance run 6 times
- `--n_epochs` - number of instances

### reference

[Optimistic Whittle Index Policy: Online Learning for Restless Bandits](https://arxiv.org/abs/2205.15372). Kai Wang*, Lily Xu*, Aparna Taneja, Milind Tambe
