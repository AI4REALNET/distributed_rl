# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:58:35 2024

@author: david
"""

import numpy as np

import extract_data.runner as runner
import extract_data.fetch_data as fd

import mutual_information.cmi_computation as cmi

import grid2op
import os
import time

import cluster.block_diag as bd

SEED = 29
np.random.seed(SEED)

if __name__=='__main__':

    # Inputs: modify as will
    env_name = 'l2rpn_idf_2023'
    n_episodes = 100
    n_samples = 50

    quant_list = [0.7]

    # --------------------------------------------

    t_tot = time.time()
    
    path = f"./data/{env_name}_{n_episodes}"
    env = grid2op.make(env_name)

    n = env.observation_space.n_line
    m = env.observation_space.n_sub

    connections = env.action_space.sub_info
    #collect history for each substation
    mi = np.zeros((n,m))
    shuffled_mi = np.zeros((n,m))

    for sub in range(m):
        if connections[sub] > 3:
            sub_path = os.path.join(path, f'sub{sub}')
            os.makedirs(sub_path, exist_ok=True)

            st_r = time.time()
            runner.run(sub, env, n_episodes, SEED, sub_path)
            end_r = time.time()
            print(f'sub {sub}: runner time {round(end_r-st_r,2)}')

            fd.fetch(env, n_samples, sub_path)

            history = np.load(os.path.join(sub_path, "hist.npz"))["data"]
            st_m = time.time()
            mi_vector, eta = cmi.compute_mi_matrix_parallel(n, m, sub, history)
            end_m = time.time()
            mi[:,sub] = mi_vector[:,n+sub]
            print(f'sub {sub}: matrix time {round(end_m-st_m,2)}')

            shuffled_history = history.copy()
            np.random.shuffle(shuffled_history[:,:n])
            shuffled_vector, seta = cmi.compute_mi_matrix_parallel(n, m, sub, shuffled_history)
            shuffled_mi[:,sub] = shuffled_vector[:,n+sub]
        
    unbiased_mi = mi - shuffled_mi

    with open(os.path.join(path, 'mi.npy'), 'wb') as f:
        np.save(f, mi)

    with open(os.path.join(path, 'shuffled_mi.npy'), 'wb') as f:
        np.save(f, shuffled_mi)

    with open(os.path.join(path, 'unbiased_mi.npy'), 'wb') as f:
        np.save(f, unbiased_mi)

    # with open(os.path.join(path, 'unbiased_mi.npy'), 'rb') as f:
    #     unbiased_mi = np.load(f)   

    bd.diagonalize(unbiased_mi, os.path.join(path, 'diagonalizations'), quant_list, env_name)

    t_tot_end = time.time()
    print(f'DONE! Elapsed time: {round(t_tot_end-t_tot,2)}')


    
        



    
        
