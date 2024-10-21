import argparse, subprocess, importlib, json
from Brian2.get_brian_dir_path_head import *
input_param = importlib.import_module(path_head+'.jobs_sub.ray_tune.input_param')
general_tools = importlib.import_module(path_head+'.tools.general_tools')
tools_names = [x for x in general_tools.__dict__ if not x.startswith("_")]
globals().update({k: getattr(general_tools, k) for k in tools_names})
from distutils.util import strtobool

#-------------------------------------------------------
# total slurm wall time from clusters
def get_total_wall_time_slurm(args):
  if args.account == 'standby':
      total_wall_time_slurm = '04:00:00'
  elif args.account == 'physics':
      if args.cluster == 'bell' or args.cluster == 'negishi':
          total_wall_time_slurm = '14-00:00:00'
      elif args.cluster == 'brown':
          total_wall_time_slurm = '8-00:00:00'

  return total_wall_time_slurm

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='basic input')
  parser.add_argument('--torchvision_data', required=True, type=str, help='name of the torchvision dataset, e.g. "MNIST"')
  parser.add_argument('--sqrt_grp_size', required=True, type=json.loads, 
                      help='sqrt size of network, e.g. "[3, 1]", means 1st/2nd hidden layer has 9 and 1 neurons in each group, respectively') 
  parser.add_argument('--cluster', required=True, type=str, help='name of ITAP community clusters')
  parser.add_argument('--account', required=True, type=str, help='name of account')
  parser.add_argument('--search_algo', type=str, required=False, default="HyperOptSearch",
            help = "options: SkOptSearch, HyperOptSearch, BayesOptSearch, AxSearch, HEBOSearch, OptunaSearch")
  parser.add_argument('--note', required=False, type=str, default='', help='brief_description') 
  parser.add_argument('--idx_range_train', required=False, type=json.loads, default=[0, 1000], 
                              help='index range for training samples, e.g. "[0, 1000]"')
  parser.add_argument('--idx_range_valid', required=False, type=json.loads, default=[0, 1000], 
                              help='index range for validation samples, e.g. "[0, 1000]"')
  parser.add_argument('--idx_range_test', required=False, type=json.loads, default=[0, 1000], 
                              help='index range for testing samples, e.g. "[0, 1000]"')
  parser.add_argument('--task', required=True, type=str, help='option: ray.train, ray.tune')
  parser.add_argument('--mem', required=False, type=str, default='40GB', help='option: ray.train, ray.tune')
  parser.add_argument('--num_workers', required=False, type=int, default=None, help='number of workers')
  parser.add_argument('--n_epoch', required=False, default=1, type=int, help='number of epoch')
  parser.add_argument('--test_option',  required=False, default=None, type=str, help='option: None, add_more_neuron,  add_more_syn, use_rank0_info')
  parser.add_argument('--train',  required=False, default=True, type=lambda x: bool(strtobool(x)), help='options:  True or False, i.e. train or not')
  parser.add_argument('--test',  required=False, default=True, type=lambda x: bool(strtobool(x)), help='options:  True or False, i.e. test or not')
  parser.add_argument('--fixed_seed',  required=False, default=False, type=lambda x: bool(strtobool(x)), help='options:  True or False, i.e. each worker process the same image or not')

  args = parser.parse_args()
  
  n_hlayer = len(args.sqrt_grp_size) # 1: 1-hlayer, 2: 2-hlayer
  job_name = args.task+ '_'
  job_name += str(n_hlayer)+'h_sqrtgrp'+'-'.join([str(x) for x in args.sqrt_grp_size])
  job_name += '_idxtrain'+'-'.join([str(x) for x in args.idx_range_train])
  if args.task == 'ray.tune':
    job_name += '_idxvalid'+'-'.join([str(x) for x in args.idx_range_valid])
  job_name += '_idxtest'+'-'.join([str(x) for x in args.idx_range_test])
  if args.task == 'ray.train':
      job_name += f'_num_workers-{args.num_workers}_test_option-{args.test_option}-n_epoch-{args.n_epoch}'
  job_name += f'_note-{args.note}'
  
  sleep_time = 300 # check every 300 seconds
  cpus_per_task = 25 # number of CPUs per task
  gpus_per_task = 0 # number of GPUs per task
  if args.task=='ray.tune':
      n_concurrent = input_param.n_initial_points
      n_iter = 1000 # number of iteration of submitting batches of jobs
  elif args.task=='ray.train':
      n_concurrent = args.num_workers
      n_iter = 1 # number of iteration of submitting batches of jobs

  # if n_concurrent 8 and cpus_per_task = 10, then 8//10 = 0, so num_nodes = 8//10 + 1
  num_nodes = int(n_concurrent//cpus_per_task) 
  if int(n_concurrent/cpus_per_task) > int(n_concurrent//cpus_per_task):
      num_nodes += 1

  base_dir = "/scratch/"+args.cluster+"/wxie/"+job_name
  ray_log_dir = base_dir + '/ray_log/'
  
  total_wall_time_slurm = get_total_wall_time_slurm(args) # total wall time
  
  for i in range(n_iter):
      check_running_jobs(sleep_time, job_name)
      run_command = [
          "python", "submit_ray_tune.py",  
          "--base_dir", base_dir,  
          "--job_name", job_name, 
          "--log_out", job_name, 
          "--err_out", job_name, 
          "--num_nodes", str(num_nodes), 
          "--gpus_per_task", str(gpus_per_task), 
          "--mem", args.mem,  
          "--total_wall_time_slurm", total_wall_time_slurm, 
          "--account", args.account, 
          "--torchvision_data", args.torchvision_data, 
          "--sqrt_grp_size", json.dumps(args.sqrt_grp_size),
          "--idx_range_train", json.dumps(args.idx_range_train),
          "--idx_range_test", json.dumps(args.idx_range_test),
          "--task", args.task
          ]

      if args.task=='ray.tune':
          run_command.extend(["--cpus_per_task", str(cpus_per_task)]) 
          run_command.extend(["--search_algo", args.search_algo])
          run_command.extend(["--idx_range_valid", json.dumps(args.idx_range_valid)])
      elif args.task=='ray.train':
          run_command.extend(["--cpus_per_task", str(cpus_per_task+1)]) # need to add 1, not sure why 
          run_command.extend(["--num_workers", str(args.num_workers)])
          run_command.extend(["--n_epoch", str(args.n_epoch)])
          run_command.extend(["--test_option", str(args.test_option)])
          run_command.extend(["--train", str(args.train)])
          run_command.extend(["--test", str(args.test)])
          run_command.extend(["--fixed_seed", str(args.fixed_seed)])

      subprocess.run(run_command)
  
      # pause to let the jobs status show up in the queue
      time.sleep(10)
