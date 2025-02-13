## Distributed Q Learning algorithm (DQL)

#### Short description of the algorithm
The DQL algorithm is a distributed version of the popular Q-Learning algorithm [...]
The original update of state-value function of an agent according to the Q-learning algorithm is:

$$
Q(s,a) = (1-\alpha) Q(s,a) + \alpha(r + \gamma\max_{a'} Q(s,a'))
$$

The distributed version instead has the following update for each agent:

$$
Q(s,a) = (1-\alpha) Q(s,a) + \alpha(r + \gamma\max_{a'} Q_{\text{next}}(s,a'))
$$

where $Q_{\text{next}}$ is the state-value function of the successor agent, i.e., the successor node in the graph.


#### Overview of code structure
:open_file_folder: **distributed_q_learning**
├── :open_file_folder: flatland tools

│   └── ...

├── :open_file_folder: training _utils

│   └── ...

├── :open_file_folder: plot

│   └── ...

├── train.py

├── eval.py

The folder *flatland_tools* contains ..., the folder *training_utils* contains, the folder *plot* contains.
The python scripts *train.py*, *eval.py* can be used to train and evaluate the DQL algorithm.


#### Installation guide
Create a virtual environment, activate it and install all the requirements.

```commandline
python -m venv venv_dql
source venv_venv_dql/bin/activate
pip install -r requirements.txt
```

The main dependencies are `flatland-rl`, `numpy`, `scikit-learn`, `scipy`.


#### Input
In the file *train.py* edit the following two parts.

1) Inside the function *generate_env* set the following return options
    - `env_width`: the width of the flatland grid
    - `env_height`: the height of the flatland grid
    - `env_n_cities`: the number of cities in the flatland grid
    - `env_n_trains`: the number of trains in the flatland grid
    - `seed`: the seed used to generate the environment

2) Before the starting of the main, edit the following variables
    - `malf_configs`: the rate and the [min,max] interval defining the malfunctions in flatland
    - `hp_configs` : the hyperparameters configuration of the DQL algorithm, i.e., 'epsilon', 'epsilon decay', 'alpha', 'alpha decay', 'n_episodes'.
    - `out_dir`: the path of the output directory
    - `n_workers`: the number of parallel workers used to execute the code
    - `master_seed`: the master seed that generates all the randomness
    - `log_every`: frequency of the logging
    - `save_every`: frequency of DQL model savings
    - `n_evals_calc`: number of evaluations
    - `eval_batch_size`: batch size used to compute evaluations


#### Output
The output is generated in the `out_dir` path specified in the *train.py* script.
It contains:
- one folder for each hyperparameter configuration, containing the saved models, the configuration parameters of the experiments, the cumulative rewards obtained during the training phase, the log file and the seeds.
- a global log file containing the computation time and the configurations of each experiment.


#### Reproduce experiments
Inside the function `generate_env` set the following return argument

```commandline
    return ModifiedEnv(
        env_width=40,
        env_height=40,
        env_n_cities=7,
        env_n_trains=5,
        seed=13,
        destination_bonus=200,
        deadlock_penalty=-200,
        delay_threshold=0.2,
        malfunction_rate=malf_rate,
        malfunction_min_duration=malf_min,
        malfunction_max_duration=malf_max,
        malfunction_seed=malf_seed
    )
```

Before the main, edit the variables as follows:

```commandline
# Malfunction configurations
malf_configs = pd.DataFrame([
    [0., 0, 0],
    # [1e-3, 5, 15],
    # [1e-3, 15, 30],
    # [5e-3, 5, 15],
    # [5e-3, 15, 30]
], columns=['rate', 'min', 'max'])

# Hyperparameters
hp_configs = pd.DataFrame([
    # Best hps from exp 12
    [1.0, 0.99997, 0.1, 0.99999, int(4e5)],  # blue
    [1.0, 0.999965, 0.1, 0.99999, int(4e5)],  # orange
    [1.0, 0.99997, 0.01, 1., int(4e5)],  # green
    [1.0, 0.999965, 0.01, 1.00000, int(4e5)],  # red
    [1.0, 0.999975, 0.1, 0.99999, int(4e5)],  # purple
    [1.0, 0.999975, 0.01, 1, int(4e5)]  # brown
], columns=['epsilon', 'epsilon decay', 'alpha', 'alpha decay', 'n_episodes'])

# Other parameters
out_dir = 'experiments/test'
n_workers = multiprocessing.cpu_count()
master_seed = 666
log_every = 10
save_every = 10
n_evals_calc = lambda _: 1 # lambda malf_rate: int(300 / malf_rate)
eval_batch_size = 10_000
```

Use the plot function to obtain the following training curve

<img alt="Training curve" src="https://gitlab.inesctec.pt/cpes/european-projects/ai4realnet/politecnico-di-milano/beta_release/-/blob/main/distributed_q_learning/plot/training.png" height=60px width=140px>
