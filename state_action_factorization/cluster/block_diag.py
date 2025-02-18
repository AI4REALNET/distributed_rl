import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.patches import Rectangle
import pandas as pd
import os
import grid2op

def dfs(adj_matrix, node, targets, variables, visited_r, visited_c, component, rc):
  """
  Depth First Search algorithm to find connected components in a graph

  Parameters
    ----------
    adj_matrix : numpy array
      The adjacency matrix of the graph
    node : int
        The node from which we start the search
    targets : list
        The list of targets
    variables : list   
        The list of variables
    visited_r : list
        The list of visited targets
    visited_c : list
        The list of visited variables
    component : list
        The list of the component we are updating
    rc : str
        The type of node we are considering ('r' for row, 'c' for column)
  """
  #the argument 'rc' is checked because we need to act in different ways if we are considering a row or a column
  if rc == 'r':
    #keep track that the node is visited and append the variable it to the component we are updating
    visited_r[node] = True
    component.append(targets[node])

    #we iterate on the columns to check for possible neighbours
    for neighbour in range(adj_matrix.shape[1]):
      #if a non visited neighbour is found we call the recursive function on it
      if adj_matrix[node][neighbour] == True and not visited_c[neighbour]:
        dfs(adj_matrix, neighbour, targets, variables, visited_r, visited_c, component, rc='c')
  else:
    #same as before but keep in mind that we were passed a column
    visited_c[node] = True
    component.append(variables[node])

    for neighbour in range(adj_matrix.shape[0]):
      if adj_matrix[neighbour][node] == True and not visited_r[neighbour]:
        dfs(adj_matrix, neighbour, targets, variables, visited_r, visited_c, component, rc='r')
        

def find_connected_components(adj_matrix, targets, variables):
  """
  Apply DFS to find connected components in a graph

  Parameters
    ----------
    adj_matrix : numpy array
      The adjacency matrix of the graph
    targets : list
        The list of targets
    variables : list   
        The list of variables

    Returns
    -------
    components : list
        The list of connected components
  """
  n = adj_matrix.shape[0]
  m = adj_matrix.shape[1]
  components = []

  #define a visited flag for each target and for each variable
  visited_r = [False] * n
  visited_c = [False] * m

  for row in range(n):
    if not visited_r[row]:

      #init the component for the actual row as empty
      component = []

      #call the recursive function indicating that we are considering an element taken from rows (a target)
      dfs(adj_matrix, row, targets, variables, visited_r, visited_c, component, rc='r')
      components.append(component)

  return components


def block_diagonalization(matrix, targets, variables, thres):
    """
    Get a block diagonal matrix from a given matrix with a given threshold

    Parameters
    ----------
    matrix : numpy array
      The input matrix
    targets : list
        The list of targets
    variables : list
        The list of variables
    thres : float
        The threshold to apply

    Returns
    -------
    block_df : pandas dataframe
        The block diagonal matrix
    matrix_bin : numpy array
        The binarized matrix
    components : list
        The list of connected components
    rearranged_targets : list
        The list of rearranged targets
    rearranged_variables : list
        The list of rearranged variables
    """
    
    matrix_bin = matrix.copy()
    matrix_bin[matrix > thres] = 1
    matrix_bin[matrix <= thres] = 0
    
    components = find_connected_components(matrix_bin, targets, variables)
    
    rearranged_targets = []
    rearranged_variables = []
    for component in components:
        for n in component:
            if n in targets:
                rearranged_targets.append(n)
            else:
                rearranged_variables.append(n)
    df = pd.DataFrame(data=matrix_bin, index=targets, columns=variables)
                
    block_df = df.loc[rearranged_targets,rearranged_variables]
    
    return block_df, block_df.to_numpy(), components, rearranged_targets, rearranged_variables

    
def plot_results(bin, df, out_folder, quant):
    """
    Plot the results of the block diagonalization

    Parameters
    ----------
    bin : numpy array
      The binarized matrix
    df : pandas dataframe
        The block diagonal matrix
    out_folder : str
        The output folder
    quant : float
        The threshold quantile
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sn.heatmap(data = bin, annot=True, cbar=False, ax=ax1)

    sn.heatmap(data = df, annot=True, cbar=False, ax=ax2)

    #plt.suptitle(f'Quant = {quant}, score = {round(total_score,2)}', size=24)
    plt.suptitle(f'Quant = {quant}', size=24)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(out_folder, f'cliques_{quant}.png'), dpi=200)
    plt.close()

def diagonalize(a, path, quant_list, env_name='l2rpn_case14_sandbox'):
  """
  Diagonalize a matrix and plot the results

    Parameters
    ----------
    a : numpy array
      The input matrix
    path : str
        The output folder
    quant_list : list
        The list of quantiles
    env_name : str
        The environment name
  """
  os.makedirs(path, exist_ok=True)
  
  env = grid2op.make(env_name)
  idx = []
  n = env.observation_space.n_line
  m = env.observation_space.n_sub

  for sub in range(m):
      if env.observation_space.sub_info[sub] > 3:
          idx.append(sub)
  bin = np.zeros((n,m))

  targets = [f's{line}' for line in range(n)]
  variables = [f'sub{s}' for s in range(m)]

  for quant in quant_list:

    print()
    print(f'Threshold quantile: {quant}')

    for sub in idx:
        thresh = np.quantile(a[:,sub].flatten(), quant)
        bin[:,sub] = a[:,sub]>thresh

    bdf, bm, _, _, _ = block_diagonalization(bin, targets, variables, 0.75)
      
    #blocks_idx = find_cliques(bm)
    #total_score = compute_total_score(blocks_idx, bm)

    plot_results(bin, bdf, path, quant)

    #print(f'Score: {round(total_score,2)}')
    print()
