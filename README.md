# Deliverable D2.1 | Politecnico di Milano

Authors: Gianvito Losapio, Marco Mussi, Alberto Maria Metelli, Marcello Restelli

## Description
The code is composed of two different folders, each containing a specific algorithm implementation.

- :open_file_folder: **distributed_q_learning** contains the code for the Distributed Q Learning algorithm (DQL)
- :open_file_folder: **state_action_factorization** contains the code for the State and Action Factorization algorithm (SAF)

In the following, the code of the two algorithms will be presented separately.

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


#### Installation guide
Create a virtual environment, activate it and install all the requirements.

```commandline
python -m venv venv_dql
source venv_venv_dql/bin/activate
pip install -r requirements.txt
```

The main dependencies are `flatland-rl`, `numpy`, `scikit-learn`, `scipy`.


#### Input

#### Output

#### Reproduce experiments


## State and Action Factorization algorithm (SAF)

