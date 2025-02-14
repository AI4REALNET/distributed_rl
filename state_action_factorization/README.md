# Distributed Reinforcement Learning for Power Grid Operations


The underlying idea is to decompose the problem by splitting the MDP into sub-problems by estimating Mutual Information between pairs of state and action variables, that are then clustered so that variables that have high correlation with the same ones are grouped together. The code for this part is collected in the `clustering` folder. We made an experiment on a custom made MDP and one on a simple Grid2op environment.


### Short description of the algorithm


### Overview of code structure


### Installation guide

Create a virtual environment, activate it and install the requirements.

```commandline
python3 -m venv venv_clustering
source venv_clustering/bin/activate
pip install -r requirements.txt

```

Replace `grid2op_patch/EpisodeData.py` in `.../venv_clustering/lib/python3.11/site-packages/grid2op/Episode/EpisodeData.py`

Replace `grid2op_patch/aux_fun.py` in `.../venv_clustering/lib/python3.11/site-packages/grid2op/Runner/aux_fun.py`

### Input

Run experiment on synthetic data
```commandline
python clustering/synthetic_data.py
```

Run experiment on power grids
```commandline
python clustering/power_grid.py
```
One can modify the two variables `n_episodes` and `n_samples`. The first one is the number of time series that are used in the simulation for collecting the data, the second is the number of samples that are used in the computation of the Mutual Information estimator. The total number of samples collected depends on the survival of the agent in the simulation, if it lower than `n_samples`, the MI is computed on all available samples. 


### Output


### Reproduce experiments
To reproduce the results provided in the thesis, use the following settings:

```commandline
SEED = 29
n_episodes = 1000
n_samples = 50000
```

Tested on Ubuntu 18.04.6 LTS | RAM 8GB | Intel® Core™ i7-8750H CPU @ 2.20GHz × 12 
Running time: ~1h on 6 cores
