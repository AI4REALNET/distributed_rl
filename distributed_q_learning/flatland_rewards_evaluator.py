from flatland_tools.env import Environment, Node
from flatland_tools.agent import TQLearningAgent
from typing import Dict, Tuple
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env import TrainState
import numpy as np
import pandas as pd
import os
import concurrent.futures

class ModifiedEnv(Environment):
    """
    This environment produces a slightly different observation:
    - station_id: same as before
    - node_id: same as before
    - delay: same as before
    - semaphore_edge_0: 
        - 0: no train coming towards me in edge 0
        - 1: train coming towards me in edge 0
        - 2: no train coming towards me, but there is a malfunctioning train
    - semaphore_edge_1:
        same as semaphore_edge_0 but for edge 1
    We also introduce a way to save total flatland rewards and total number of malfunction steps
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, step_until_action_required=False, reset_malfunctions=False):
        self.flatland_cum_rew = 0.0
        self.total_malfunction_steps = 0
        self.terminated_by_deadlock = False
        super().reset(step_until_action_required, reset_malfunctions)

    def _step_low_level(self, actions: Dict[int, RailEnvActions], step_until_action_required: bool = False) -> bool:
        _, flatland_rewards, _, _ = self.rail_env.step(actions)
        self.flatland_cum_rew += sum(flatland_rewards.values())
        self.total_malfunction_steps += sum([1 for train in self.rail_env.agents if train.state == TrainState.MALFUNCTION])
        self._update_rewards()
        if self._check_deadlocks():
            return True
        if step_until_action_required:
            while len(self.graph.trainsThatAreOnNodes(self.rail_env.agents)) == 0 and not self.rail_env.dones['__all__']:
                self.rail_env.step({tid: RailEnvActions.MOVE_FORWARD for tid in range(self.rail_env.get_num_agents())})
                self._update_rewards()
                if self._check_deadlocks():
                    return True
        return self.rail_env.dones['__all__']
    
    def _check_deadlocks(self) -> bool:
        self.terminated_by_deadlock = super()._check_deadlocks()
        return self.terminated_by_deadlock

    def observe(self):
        """
        Returns [(train_id, node_id, obs, old_reward)]
        """
        observations = []
        for train_id, node in self.graph.trainsThatAreOnNodes(self.rail_env.agents).items():
            node_id = self.node_to_id[node]
            obs = self.observe_node_v2(node, train_id)
            self.last_obs[node_id] = obs
            old_reward = self._get_reward_for_node(node)
            observations.append((train_id, node_id, obs, old_reward))
        return observations

    def observe_node_v2(self, node: Node, train_id: int) -> Tuple[int, int, int, int, int]:
        """
        Preconditions: train_id exists and is on a node
        Returns (station_id, node_id, delay, semaphore edge 0, sempahore edge 1)
        """
        train = self.rail_env.agents[train_id]

        # Station identifier
        station_id = self.stations[train.target]

        # Train delay
        delay = self._discretize_delay(train_id, self._compute_delay(train_id))

        # Semaphores
        sem_0 = self.compute_semaphore_v2(node, 0)
        sem_1 = self.compute_semaphore_v2(node, 1)

        return station_id, self.node_to_id[node], delay, sem_0, sem_1
    
    def compute_semaphore_v2(self, node: Node, edge_idx: int) -> int:
        _, edge_data = self.graph.fw_star(node)[edge_idx]
        malf_present = False
        for train in self.rail_env.agents:
            row, col = train.position if train.position is not None else train.initial_position
            if row == node.row and col == node.col:
                continue
            # if train is currently on this edge and its direction is opposite to the one that is available on this edge starting from the given node
            # or if train is malfunctioning
            # semaphore is activated (True)
            train_on_edge = edge_data.direction_mask[row, col] != -1
            opposite_dir = train.direction != edge_data.direction_mask[row, col]
            malfunction = train.state == TrainState.MALFUNCTION
            if train_on_edge and opposite_dir:
                return 1
            elif malfunction:
                malf_present = True
        return 2 if malf_present else 0
    
def apply_silent_deadlock_prevention(actions, env: Environment):
    new_actions = []
    from_to = {}
    for train_id, node_id, action in actions:
        if action != 2:
            next_node = env.graph.fw_star(env.nodes[node_id])[action][0]
            if next_node in from_to:
                new_actions.append((train_id, node_id, 2)) # Stop it instead
            else:
                from_to[env.nodes[node_id]] = next_node
        new_actions.append((train_id, node_id, action))
    return new_actions
    
def generate_env(malf_rate, malf_min, malf_max, malf_seed) -> ModifiedEnv:
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

def eval_once(
    env: Environment,
    agent: TQLearningAgent,
    malf_seed: int
):
    env.rail_env.malfunction_generator.seed = malf_seed
    env.reset(step_until_action_required=True, reset_malfunctions=True)
    cumulative_reward = 0.0
    train_to_obs_action_node_id = {}
    while True:
        actions = []
        for train_id, node_id, obs, old_reward in env.observe():
            if train_id in train_to_obs_action_node_id:
                last_obs, last_action, old_node_id = train_to_obs_action_node_id[train_id]
            action = agent.max_action(obs)
            cumulative_reward += old_reward # Log reward
            actions.append((train_id, node_id, action))
            train_to_obs_action_node_id[train_id] = (obs, action, node_id)
        # PREVENT SILENT DEADLOCKS
        actions = apply_silent_deadlock_prevention(actions, env)
        if env.step(actions, step_until_action_required=True):
            break
    last_rewards = env.end_episode()
    for train_id, (obs, action, node_id) in train_to_obs_action_node_id.items():
        r = last_rewards[node_id]
        cumulative_reward += r
    
    n_trains = env.rail_env.get_num_agents()
    delays = [None] * n_trains
    arrived = [None] * n_trains
    for train_id, (_, final_delay) in env.train_to_last_node.items():
        arrived[train_id] = env.trains_arrived[train_id]
        if env.trains_arrived[train_id]:
            delays[train_id] = final_delay
    flatland_normalized_reward = 1 + (
        env.flatland_cum_rew / (
            env.rail_env._max_episode_steps * env.rail_env.get_num_agents()
        ))
    return \
        malf_seed, \
        cumulative_reward, \
        *delays, \
        *arrived, \
        env.flatland_cum_rew, \
        flatland_normalized_reward, \
        env.total_malfunction_steps, \
        env.terminated_by_deadlock

def eval_batch(
    env: Environment,
    agent: TQLearningAgent,
    malf_seeds: np.ndarray[int],
    first_eval_id: int,
    exp_id: int,
    episode: int,
    out_dir: str,
    chunk_id: int
):
    results = []
    for i, malf_seed in enumerate(malf_seeds):
        result = eval_once(env, agent, malf_seed)
        results.append((exp_id, episode, first_eval_id + i, *result))
    pd.DataFrame(results, columns=
        [
            'Experiment id',
            'Episode',
            'Evaluation id',
            'Malfunction seed',
            'Cumulative reward'
        ] + 
        [
            f'Train {i} delay' for i in range(env.rail_env.get_num_agents())
        ] + 
        [
            f'Train {i} arrived' for i in range(env.rail_env.get_num_agents())
        ] +
        [
            'Flatland cumulative reward',
            'Flatland normalized reward',
            '# malfunction steps',
            'Terminated by deadlock'
        ]
    ).to_parquet(os.path.join(out_dir, f'{chunk_id:07d}.parquet'))

# Execution parameters
master_seed = 666
results_dir = 'results_exp_12'
out_dir = 'flatland_eval_results'
n_workers = 8
get_n_evals = 1 #lambda malf_rate: 1 + int(100 / malf_rate)
batch_size = 5000

if __name__ == '__main__':
    os.makedirs(out_dir, exist_ok=True)
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers)
    tasks = []
    chunk_id = 0
    for exp_dir in os.listdir(results_dir):
        exp_dir_path = os.path.join(results_dir, exp_dir)
        if exp_dir == 'eval_results' or not os.path.isdir(exp_dir_path):
            continue
        exp_id = int(exp_dir)
        config = pd.read_csv(os.path.join(exp_dir_path, f'{exp_id}_config.csv'))
        config.set_index(config.columns[0], inplace=True)
        n_evals = 1 # get_n_evals(config.loc['rate'].item())
        malf_seeds = np.random.RandomState(master_seed).randint(0, 2**32, n_evals)
        qtables_path = os.path.join(exp_dir_path, f'qtables')
        for qtable_filename in os.listdir(qtables_path):
            episode = int(qtable_filename.split('.')[0].split('_')[-1])
            for first_eval_id in range(0, n_evals, batch_size):
                task = executor.submit(
                    eval_batch,
                    generate_env(
                        config.loc['rate'].item(),
                        config.loc['min'].item(),
                        config.loc['max'].item(),
                        malf_seeds[first_eval_id:first_eval_id + batch_size]
                    ),
                    TQLearningAgent.load(os.path.join(qtables_path, qtable_filename)),
                    malf_seeds[first_eval_id:first_eval_id + batch_size],
                    first_eval_id,
                    exp_id,
                    episode,
                    out_dir,
                    chunk_id
                )
                chunk_id += 1
                tasks.append(task)
    concurrent.futures.wait(tasks)
    executor.shutdown()
    print('Done')

