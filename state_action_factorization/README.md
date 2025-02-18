# Distributed Reinforcement Learning for Power Grid Operations


The underlying idea is to decompose the problem by splitting the MDP into sub-problems by estimating Mutual Information between pairs of state and action variables, that are then clustered so that variables that have high correlation with the same ones are grouped together. The code for this part is collected in the `clustering` folder. We made an experiment on a custom made MDP and one on a simple Grid2op environment.


### Short description of the algorithm

The main idea is to define a fully connected graph $\mathcal{G}=(V,E)$ in which:

- $V$ is the set of nodes containing all the state and action components from $\mathbf{s}, \mathbf{a}$ and all the next state components from $\mathbf{s'}$

- $E$ is the set of edges representing the interactions among components
    
    $$
    E = \{(x_i,s'_j) \,|\, x_i,s'_j\in V\; \text{and}\; c(x_i,s'_j)\geq \delta\},
    $$
    
    where $c(x_i,s'_j)$ is a metric that measures how much a variable $x_i$ (state or action component) is important to predict the variable $s'_j$ (next state component), with $\delta$ being a suitable threshold.

The metric that is used in this algorithm is the mutual information, i.e., the amount of information (or, equivalently, reduction in uncertainty) that knowing either variable provides about the other.

A dataset is collected from the environment and the adjacency matrix of the graph $\mathcal{G}$ is computed. The matrix is then filtered with a suitable theshold and the diagonal blocks are used to define a factorization of the original Markov Decision Process, $\big( \widehat{\mathcal{S}}_k, \widehat{\mathcal{A}}_k  \big)_{k=1}^{\widehat{K}}$ that ideally matches the true factorization $\big(\mathcal{S}_k, \mathcal{A}_k  \big)_{k=1}^{K}$.


#### Overview of code structure

:open_file_folder: **state_action_factorization**

├── :open_file_folder: cluster

│   └── ...

├── :open_file_folder: extract_data

│   └── ...

├── :open_file_folder: grid2op_patch

│   └── ...

├── :open_file_folder: mutual_information

│   └── ...

├── main.py


The folder *cluster* contains the block diagonalization procedure, the folder *extract_data* contains the functions for exctracting data from Grid2Op, the folder *grid2op_patch* contains two python scripts that need to be replaced in the Grid2Op library, the folder *mutual_information* contains the functions for the computation of the mutual information. The python scripts *main.py* can be used to launch the factorization algorithm on Grid2Op.


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
