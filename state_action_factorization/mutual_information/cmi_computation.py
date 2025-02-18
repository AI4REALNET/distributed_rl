import numpy as np
import multiprocessing as mp
import mutual_information.mixed as mixed
import os
from tqdm import tqdm

import time

def get_relative_indices(state_action, ns, iv, dim_state, dim_action):
  """
  Get the relative indices of the variables in the history matrix

  Parameters
  ----------
  state_action : str
    The type of variable to consider ('state' or 'action')
  ns : int
    The next state index
  iv : int
    The input variable index
  dim_state : int
    The number of states
  dim_action : int
    The number of actions

  Returns
  -------
  ns : int
    The next state index
  iv_idx : int
    The input variable index
  k_idx : list
    The list of indices of the remainder variables
  """
  if state_action == 'state':
    iv_idx = dim_state + iv
  if state_action == 'action':
    iv_idx = 2 * dim_state + iv

  k_idx = [x for x in np.arange(dim_state, 2*dim_state+dim_action) if x != iv_idx]

  return ns, iv_idx, k_idx

def compute_MI_entry(iv_label, ns_idx, iv_idx, n, m, history):
  """
  Compute the mutual information between the next state and the input variable

  Parameters
  ----------
  iv_label : str
    The type of variable to consider ('state' or 'action')
  ns_idx : int
    The next state index
  iv_idx : int
    The input variable index
  n : int
    The number of states
  m : int
    The number of actions
  history : numpy array
    The history matrix

  Returns
  -------
  mi_ns_iv : float
    The mutual information between the next state and the input variable

  """
  ns, iv, k_idx = get_relative_indices(iv_label, ns_idx, iv_idx, n, m)
  
  ns_vector = history[:, ns].reshape((len(history),1))
  iv_vector = history[:, iv].reshape((len(history),1))
  
  print(f"[{os.getpid()}] : starting Mixed_KSG", flush=True)
  st_time = time.time()
  mi_ns_iv = mixed.Mixed_KSG(ns_vector, iv_vector, k=int(len(history)/20))
  end_time = time.time()
  print(f'[{os.getpid()}] : ETA {round(end_time-st_time,2)}. Next state {ns}/{n}. Input variable: {iv_idx}', flush=True)  

  return mi_ns_iv

def compute_cmi_matrix(n, m, history):
    """
    Compute the mutual information matrix

    Parameters
    ----------
    n : int
      The number of states
    m : int
      The number of actions
    history : numpy array
      The history matrix

    Returns
    -------
    MI : numpy array
      The mutual information matrix
    """
    MI = np.zeros((n, n+m))
    history = np.asarray(history)

    st = time.time()
    for ns in range(n):
      print()
      print('--------------------')
      print(f'Next state {ns}/{n}')
      iv_label = 'state'
      for cs in range(n):
        sti = time.time()  
        print(f'Input variable: state {cs}/{n}')  
       
        MI[ns][cs] = compute_MI_entry(iv_label, ns, cs, n, m, history)
        print(f'Computed probabilities. Elapsed time: {round(time.time()-sti, 2)} s')
        
      iv_label = 'action'
      for a in range(m):
        sti = time.time() 
        print(f'Input variable: action {a}/{m}')  
        
        MI[ns][n+a] = compute_MI_entry(iv_label, ns, a, n, m, history)
        print(f'Computed probabilities. Elapsed time: {round(time.time()-sti, 2)} s')
     
    print('-----------------------------------------')    
    print(f'Total time: {round(time.time() - st, 2)} s')
    return MI

def compute_MI_entry_wrapper(args):
    return compute_MI_entry(*args)


def compute_mi_matrix_parallel(n, m, sub, history):
    """
    Compute the mutual information matrix in parallel

    Parameters
    ----------
    n : int
      The number of states
    m : int
      The number of actions
    sub : int
      The substation index
    history : numpy array
      The history matrix

    Returns
    -------
    MI : numpy array
      The mutual information matrix
    t : float
      The elapsed time

    """
    MI = np.zeros((n, n+m))

    history = np.asanyarray(history)

    st = time.time()

    pool = mp.Pool(int(mp.cpu_count()/2))

    args_list = []
    for ns in range(n):
        iv_label = 'action'
        for a in range(m):
            if a == sub:
                args_list.append((iv_label, ns, a, n, m, history))

    results = []
    for result in tqdm(pool.imap(compute_MI_entry_wrapper, args_list), total=len(args_list)):
        results.append(result)

    i = 0
    for ns in range(n):
        for a in range(m):
            if a == sub:
                MI[ns][n+a] = results[i]
                i += 1

    print('-----------------------------------------')
    print(f'Total time: {round(time.time() - st, 2)} s')

    t = round(time.time() - st, 2)
    return MI, t   




