import numpy as np
import pandas as pd
import pickle

class TQLearningAgent:
    def __init__(self, gamma = 1., default_q = 0.):
        self.n_actions = 3
        self.gamma = gamma
        self.default = [default_q] * self.n_actions
        self.q_table = {}

    def __check_entry(self, state):
        if state not in self.q_table:
            self.q_table[state] = self.default.copy()

    def eval(self, state, action):
        self.__check_entry(state)
        return self.q_table[state][action]
        
    def update(self, lr, state, action, reward, next_state = None):
        self.__check_entry(state)
        self.q_table[state][action] = \
            (1 - lr) * self.q_table[state][action] + \
            lr * (reward + self.gamma * self.max_q(next_state))
        
    def max_q(self, state):
        self.__check_entry(state)
        return max(self.q_table[state]) if state is not None else 0.
    
    def max_action(self, state):
        self.__check_entry(state)
        return np.argmax(self.q_table[state])

    def dump(self, filename: str, mode: str = 'pickle'):
        """
        Dumps the agent to a file.
        :param filename: The name of the file.
        :param mode: The mode of the dump. Can be 'pickle', 'csv', 'parquet'.
        """
        if mode == 'pickle':
            self.__dump_pickle(filename)
        elif mode == 'csv':
            self.__dump_csv(filename)
        elif mode == 'parquet':
            self.__dump_parquet(filename)

    def load(filename: str):
        """
        Loads the agent from a file (pickle).
        :param filename: The name of the file.
        """
        return TQLearningAgent.__load_pickle(filename)

    def __dump_pickle(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def __load_pickle(filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)