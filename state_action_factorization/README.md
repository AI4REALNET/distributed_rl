# State and Action factorization (SAF)

![Short description of the algorithm](https://drive.google.com/uc?export=view&id=1-IpbwXPdFQk2DpWta_Prtt6qTXA2-XV1)


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

The main dependencies are `Grid2Op`, `gymnasium`, `numba` (with python >= 3.6).


### Input

In the file *main.py* you can modify the two variables `n_episodes` and `n_samples`. The first one is the number of time series that are used in the simulation for collecting the data, the second is the number of samples that are used in the computation of the Mutual Information estimator. The total number of samples collected depends on the survival of the agent in the simulation - if lower than `n_samples`, the MI is computed on all available samples.

The variable `env_name` is used to specify the environment of Grid2Op. The variable `quant_list` is used to specify the list of threshold considered that is applied to the adjacency matrix.


To run experiment on Grid2Op run
```commandline
python main.py
```


### Output
A folder `data` is created containing three different versions of the adjacency matrix, i.e., the original one, the shuffled and the unbiased one (format .npy). The unbiased matrix is used to get the final diagonal matrix that is plotted in the subfolder `diagonalizations`. Each block corresponds to an independent Markov Decision Process. Additionally, a subfolder for each substation of the grid is created containing the data extracted for that substation (format .npz) from the environment Grid2Op.


### Reproduce experiments
To reproduce the results provided in the paper [State and Action factorization in Power Grids](https://arxiv.org/abs/2409.04467), use the following settings:

```commandline
SEED = 29
n_episodes = 1000
n_samples = 50000

env_name = 'l2rpn_case14_sandbox'
quant_list = [0.7]
```

Tested on Ubuntu 18.04.6 LTS | RAM 8GB | Intel® Core™ i7-8750H CPU @ 2.20GHz × 12 
Running time: ~1h on 6 cores
