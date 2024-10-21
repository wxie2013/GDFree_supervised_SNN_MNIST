import math, time, os, subprocess, pickle, re, sys
import pandas as pd
from collections import defaultdict  
from typing import Dict, List
from io import StringIO
from datetime import datetime
import numpy as np
from ray.data.datasource import FilenameProvider
import ray.data as rd
import ray
from ray import tune
import torch.distributed as dist
from Brian2.get_brian_dir_path_head import *
input_param = importlib.import_module(path_head+'.jobs_sub.ray_tune.input_param')

# if there are running jobs, sleep for a certain time
def check_running_jobs(sleep_time, job_name = None):
    job_count = 1
    while job_count != 0:
        # check the number of jobs running in the queue
        if job_name != None:  # for jobs with specific names
            procs = subprocess.check_output(['squeue', '-u', 'wxie', '-n', job_name]).splitlines()
        else: # for all of my jobs
            procs = subprocess.check_output(['squeue', '-u', 'wxie']).splitlines()

        list_of_jobs = [proc for proc in procs if b'wxie' in proc] # b means: use a bytes object 
        job_count = len(list_of_jobs)

        if job_count != 0:
            current_time = datetime.now()
            print(job_count, " jobs with name: ", job_name, " in the queue, sleep for ", sleep_time/60, 
                    'minutes, current time: ', current_time.strftime("%d/%m/%Y %H:%M:%S"))
            time.sleep(sleep_time)

    # clear cython cache from previous runs, otherwise import brian2 can freeze
    # don't use clear_cache() from brian2. import brian2 somehow overwite many things and crash jobs script
    #subprocess.run(["rm", "-r", "/home/wxie/.cython/brian_extensions"])

# Function to Check if the path and file from specific rank  exist
def isNotEmpty(rank, base_path, common_name=str()): 
    list_dirs = []
    for dirpath, dirnames, filenames in os.walk(base_path):
        for dir in dirnames:
            if common_name in dir:
                list_dirs.append(os.path.join(dirpath, dir))

    for path in list_dirs:
        for dirpath, dirnames, filenames in os.walk(path):
            if any(f'rank{rank}.{input_param.data_format}' == fname for fname in filenames):
              return True

    return False

# obtain matrix from a a synapse
# https://brian2.readthedocs.io/en/stable/user/synapses.html#synaptic-connection-weight-matrices
# use matrix operation to speed it up. Only works when each target has a fixed number of source.
# https://gitter.im/brian-team/brian2?utm_source=notification&utm_medium=email&utm_campaign=unread-notifications
def get_matrix_from_synapse_and_normalize(syn, gmax, norm_scale):
    matrix = np.full((syn.N_pre, syn.N_post), np.nan)
    matrix[syn.i[:], syn.j[:]] = syn.w[:]

    n_pres = np.sum(~np.isnan(matrix), axis = 0) # count the number of pre-neuron to each post in the marix
    tot_weight = np.nansum(matrix, axis=0) # total weight for each post-neuron

    # scale factor when normalizing the weight of, e.g. input2e(Diehl's paper used 78).
    # norm_scale*n_pres = 0.1*784 = 78.4, where input_param.norm_scale = 0.1
    # It keep the mean of all synapses to a target neuron to 0.1. For example, the weight intial weight
    # is randomized in 0-1, i.e. average is 0.5. The mean of the total weight is 0.5*784. The weight normalization
    # factor is 78.4/(0.5*784) = 0.2. After the 1st normalization, the mean weight: 0.2*0.5 = 0.1. Starting from
    # from the 2nd normalization, the scale is 78.4/78.4=1 if there's no STDP.
    ratio = np.where(tot_weight==0, 0, np.divide(n_pres, tot_weight)) # sometime tot_weight is 0 for some synapses
    scale = gmax*norm_scale*ratio

    matrix *= scale
    flattened = matrix.flatten()

    return flattened[~np.isnan(flattened)]

# merge ROOT files in a directory and remove the original ROOT file
def merge_root_file(dir_file_list: dict):
    if len(dir_file_list) == 0:
        return -1

    src_list = None
    dir_name = None
    for path, f in dir_file_list.items():
        src_list = f
        dir_name = path

        outfile = src_list[0][:src_list[0].index('_rank')] + '.root'

        command = ['hadd']
        command.append('-k') # Skip corrupt or non-existent files, do not exit
        command.append('-v') # Explicitly set the verbosity level:
        command.append('0') # 0 request no output
        command.append(os.path.join(dir_name, outfile))
        for src in src_list:
            command.append(os.path.join(dir_name, src)) 

        merge = subprocess.run(command)

        # when failed, it still leave a file on disk
        if merge.returncode != 0:
            subprocess.run(['rm', outfile]) 
        else: # remove the original ROOT files after successful merge
            for src in src_list:
                subprocess.run(['rm',  os.path.join(dir_name, src)]) 

    return  merge.returncode  # 0 if success


# based on brian2 example to visualize synaptic connections
def visualise_connectivity(axs, S, s_name):
    Ns = len(S.x_pre)
    Nt = len(S.x_post)
    if Ns >0:
        max_s = S.x_max_pre[0]
        min_s = S.x_min_pre[0]
    else:
        max_s = 0
        min_s = 0
    if Nt>0:
        max_t = S.x_max_post[0]
        min_t = S.x_min_post[0]
    else:
        max_t = 0
        min_t = 0

    count=0
    for i, j in zip(S.x_pre, S.x_post):
        axs.plot([0, 1], [i, j], '-k')
    axs.set_xticks([0, 1], ['Source', 'Target'])
    axs.set_ylabel('Neuron X-coordinates')
    axs.set_xlabel(s_name)
    axs.set_xlim(-0.1, 1.1)
    axs.set_ylim(min(min_s, min_t), max(max_s, max_t))

# translate slurm wall time into the time in seconds
def get_time_in_seconds_from_slurm_time(slurm_time):
    tmp = re.findall(r'\d+', slurm_time)
    N_num = len(tmp)

    day = hour = minute = second = 0
    if N_num == 4:  # include day, hour, minute, second
        day, hour, minute, second = list(map(int, tmp)) # map string to integer
    elif N_num == 3:  # include hour, minute, second
        hour, minute, second = list(map(int, tmp))
    elif N_num == 2:  # include minute, second
        minute, second = list(map(int, tmp))
    else:
        sys.exit('!!! max walltime for this quene is too small !!!')

    return day*24*3600 + hour*3600 + minute*60 + second

# convert a dictionary of hyperparams to scalar or list
def info_hyperparams(name, hyperparam, unit=None):
    '''
    name: variable or list name used in the value
    hyperparam: dictionary of hyperparameters
    unit: brian2 unite, e.g. brian2.mV
    '''
    group = {key: value for key, value in hyperparam.items() if name in key}
    new_list = [None]*len(group)
    for key, value in group.items():
        layer = key[len(name):]
        if not layer:
            return value if not unit else value*unit # this is a scalar

        if layer.isnumeric():
            new_list[int(layer)] = value

    new_list = [new_list[i] for i in range(len(new_list)) if new_list[i] is not None] # remove none

    return new_list if not unit else new_list*unit

### assign each element to a random group (help from chatGPT)
def assign_element_of_array_to_random_groups(Nelement, Ngroup):
    '''
    Nelement: number of elements, e.g. neurons in a layer
    Ngroup: number of groups, e.g. 10 digit 
    '''

    if Nelement < Ngroup:
        sys.exit('!!! Nelement < Ngroup, exit !!!')

    # note: the root of group_size is garanteed to be an integer already
    group_size = int(Nelement/Ngroup)
    if((math.sqrt(group_size))**2 != group_size):
        print('math.sqrt(group_size))**2: ', math.sqrt(group_size), 'group_size: ', group_size)
        sys.exit('!!! root of group_size is not an integer, exit !!!')

    grp = np.full(Nelement, -1)
    for ig in range(Ngroup):
        indices = np.where(grp == -1)[0]
        selected = np.random.choice(indices, size=group_size, replace=False)
        grp[selected] = ig

    # now assign x/y coordinates to this group
    x_coord = np.full(Nelement, -1)
    y_coord = np.full(Nelement, -1)
    for ig in range(Ngroup):
        this_group = [idx for idx, m in enumerate(grp) if m==ig]
        for j, idx in enumerate(this_group):
            x_coord[idx] = j%math.sqrt(group_size)
            y_coord[idx] = math.sqrt(group_size) - j//math.sqrt(group_size) - 1

    x_max = math.sqrt(group_size)-1
    y_max = x_max

    return grp, x_coord, y_coord, x_max, y_max

## expand a dictionary, specifically for copying 1st layer hyperpar to other hidden layers
def expand_dictionary(d, replace_str, num_copies):
    new_d = dict(d)
    for key in d:
        if key[-1] == replace_str:
            for i in range(num_copies):
                new_d[key.replace(replace_str, str(i+1))] = d[key]
    return new_d

# one of the argument for write_csv, etc in ray.data
class FilenameProvider(FilenameProvider):
    def __init__(self, rank):
        self.rank = rank

    def get_filename_for_block(self, block, task_index, worker_index):
        return f"rank{self.rank}.{input_param.data_format}"


# save_syn or spike data in specified directory and data format
def write_data(df, dir_name, rank, mode='w'):

    if mode == 'a': # append to existing data
        existing_data_path = os.path.join(dir_name, f"rank{rank}.{input_param.data_format}")
        if os.path.exists(existing_data_path):
            if input_param.data_format == 'csv':
                existing_df = pd.read_csv(existing_data_path)
            elif input_param.data_format == 'parquet':
                existing_df = pd.read_parquet(existing_data_path)
        else:
            existing_df = pd.DataFrame()

        df = pd.concat([existing_df, df], ignore_index=True)

    #
    df_ray_data = rd.from_pandas(df)

    if df_ray_data.count() == 0: # don't write empty data
        return

    if input_param.data_format == 'csv':
        rd_write = df_ray_data.write_csv
    elif input_param.data_format == 'parquet':
        rd_write = df_ray_data.write_parquet
    

    rd_write(dir_name, filename_provider=FilenameProvider(rank))

# read data from above write_data function
def read_data(dir_name, worker_index=0, data_format=None, single_value=False):
    if data_format is not None:
        actual_format = data_format
    else:
        actual_format = input_param.data_format

    if actual_format == 'csv':
        rd_read = rd.read_csv
    elif actual_format == 'parquet':
        rd_read = rd.read_parquet
    elif actual_format == 'json':
        rd_read = rd.read_json

    if not os.path.exists(dir_name):
        print(f'warning: {dir_name} does not exist, ignore if done intentionally')
        return None

    start_time = time.time()
    filename = os.path.join(dir_name, f"rank{worker_index}.{actual_format}")
    while not os.path.exists(filename): # wait for other this ranks from last round to finish
        time.sleep(input_param.sleep_time)
        elapsed_time = time.time() - start_time
        print(f'warning: {filename} does not exist yet, waited for {elapsed_time} seconds')
        if elapsed_time > input_param.max_wait_time:
            sys.exit('!!!!!! {filename} does not exist after {elapsed_time} seconds, exit !!!!')

    df = rd_read(filename).to_pandas() # panda dataframe
    df_dict = df.to_dict(orient='list') # convert to dictionary

    # return value as numpy array
    if single_value==False:
        return {key: np.array(value) for key, value in df_dict.items()} 
    else:
        return df.to_dict(orient='records')[0] 

# check if a directory contains files with name partially match a string
def find_files_with_partial_names(directory: str, partial_name: str) -> Dict[str, List[str]]:
  """
  Finds all files with names containing the provided partial name in the given directory and its subdirectories.

  Args:
      directory: The directory to search in (as a string path).
      partial_name: The partial name to search for (case-insensitive).

  Returns:
      A dictionary where keys are subdirectory paths (strings) and values are lists of filenames (strings) containing the partial name.
  """

  matching_files = defaultdict(list)  # Automatically creates empty lists for new subdirectories
  for root, _, files in os.walk(directory):
    for filename in files:
      if partial_name.lower() in filename.lower():
        matching_files[root].append(filename)

  return matching_files

# check ia file is in any of the multi-layer directories
def find_file(filename, start_path):

  """
  Searches for a file in a directory and its subdirectories.
  Args:
      filename: The name of the file to search for.
      start_path: The starting directory to search from.
  Returns:
      The full path to the file if found, None otherwise.
  """

  for dirpath, dirnames, filenames in os.walk(start_path):
    if filename in filenames:
      return os.path.join(dirpath, filename)

  return None

def find_directory(root_dir, target_dir_name):

  """Searches for a directory with the given name within a directory tree.
  Args:
    root_dir: The root directory to start the search from.
    target_dir_name: The name of the directory to find.

  Returns:
    The full path to the found directory, or None if not found.
  """

  dir_list = []
  for root, directories, files in os.walk(root_dir):
      for directory in directories:
          if target_dir_name in directory:
              dir_list.append(os.path.join(root, directory)) 

  return dir_list

# get rank and number of workers
def get_rank_and_num_workers():
    try:
        rank = dist.get_rank() # the rank in the distributed training
        num_workers = dist.get_world_size() # the number of workers
    except:
        rank = 0
        num_workers = 1

    return rank, num_workers

#Concatenates arrays with the same key from a list of dictionaries.
def concatenate_dict_arrays(list_of_dicts):
  """
  Args:
      list_of_dicts (list): A list of dictionaries.
  Returns:
      dict: A new dictionary with concatenated arrays for shared keys.
  """
  concatenated_dict = {}

  if len(list_of_dicts) == 1:
      concatenated_dict =  list_of_dicts[0]
  else:
    for key in set.union(*[set(d.keys()) for d in list_of_dicts]):
      arrays = []
      for d in list_of_dicts:
        if key in d:
          arrays.append(d[key])

      if len(arrays) > 1:
        concatenated_array = np.concatenate(arrays)
        concatenated_dict[key] = concatenated_array

  return concatenated_dict

# sort multiple list to march the order of the first list, i.e. nsample, which in ascending order 
def sort_lists_by_key(nsample, *other_lists):
    """Sorts multiple lists based on the order of a key list.
    Args:
      nsample: The key list to sort.
      *other_lists: Other lists to be sorted according to the new order of nsample.
    Returns:
      A tuple of sorted lists, with nsample as the first element.
    """
    try:
        # Zip the lists together
        zipped_lists = zip(nsample, *other_lists)

        # Sort the zipped lists based on the first element (nsample)
        sorted_zipped_lists = sorted(zipped_lists)

        # Unzip the sorted lists
        nsample_sorted, *other_lists_sorted = zip(*sorted_zipped_lists)

        return nsample_sorted, *other_lists_sorted
    except:   
        print('!!! sort failed: lists are likely empty !!!')
        return nsample, *other_lists

#
def merge_and_replace(list_of_dicts, keys_to_expand):

    """Merges multiple dictionaries and replaces values with random choices, avoiding duplicates.
    Args:
        list_of_dicts: A list of dictionaries to merge.
    Returns:
        The merged dictionary with replaced values.
    """
    merged_dict = {}
    for dict_ in list_of_dicts:
        for key, value in dict_.items():
            if key not in merged_dict:
                if key[-1].isdigit() or key in keys_to_expand:
                    merged_dict[key] = []
                else:
                    merged_dict[key] = None

            if key[-1].isdigit() or key in keys_to_expand:
                if value not in merged_dict[key]:
                    merged_dict[key].append(value)
            elif merged_dict[key] is None:
                merged_dict[key] = value

    for key, values in merged_dict.items():
        if type(values) is list:
            merged_dict[key] = tune.choice(values)


    print(merged_dict)
    return merged_dict
