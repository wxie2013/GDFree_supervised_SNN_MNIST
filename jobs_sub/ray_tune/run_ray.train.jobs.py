import subprocess, platform
from Brian2.tools.general_tools import *

def run_commands():
  """Runs the specified Python commands with varying parameters."""

  base_cmd = f"python auto_sub.py --task ray.train --torchvision_data MNIST --sqrt_grp_size '{sqrt_grp_size}' --cluster {cluster}  --train {train} --test {test}"

  sleep_time = 60 # second
  for test_option in test_option_values:
      for idx_end, num_workers, n_epoch in zip(idx_end_values, num_workers_values, n_epoch_values):
        check_running_jobs(sleep_time)
        if idx_end >= 50000 and num_workers <5:
            account = 'physics'
        else:
            account = 'standby'

        full_cmd = f"{base_cmd} --account {account} --idx_range_train '[0, {idx_end}]' --idx_range_test '[0, 10000]'  --num_workers {num_workers}"
        full_cmd += f" --n_epoch {n_epoch} --test_option {test_option}"

        print(f"Running: {full_cmd}")
        subprocess.run(full_cmd, shell=True, check=True)

if __name__ == "__main__":
    if "negishi" in platform.node():
        cluster = "negishi"
    elif "bell" in platform.node():
        cluster = "bell"
    else:
        cluster = None

    #test_option_values = [None]
    test_option_values = ["add_more_neuron", "add_more_neuron_div"]
    train = True
    test = True
    if all(item is None for item in test_option_values): # test_option = None
        sqrt_grp_list = [[1], [2], [3], [3], [5]]
        #sqrt_grp_list = [[1,1], [2,1], [3,1], [3, 3], [5, 5]]

        idx_end_values =     [1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000, 50000, 50000, 60000, 50000, 42000, 50000, 60000, 60000, 50000]
        num_workers_values = [100,  100,  100,  100,  100,   100,   100,   100,   50,    25,    20,    10,    6,     5,     3,     2,     1]
        n_epoch_values =     [1]*len(idx_end_values)

    elif all(test_option_values): # test_options other than None
        sqrt_grp_list = [[1,1]]
        idx_end_values =     [250, 500, 750, 1250, 2500, 5000, 12500, 25000, 50000, 50000, 50000, 50000, 50000, 50000, 50000] 
        n_epoch_values =     [1,   1,   1,   1,    1,    1,    1,     1,     1,     2,     4,     5,     8,     10,    15   ] # vary n_epoch to make sure each worker sample enough statistics, e.g. idx_end_values = 50000, n_epoch = 5, num_workers = 25, means each worker samples 50000*5/25 = 10000 samples. Note: make sure fixed_seed = True in fetch_torchvision_data so that for a worker, each epoch sample different data
        num_workers_values = [25]*len(idx_end_values)

    for sqrt_grp_size in sqrt_grp_list:
        run_commands()
