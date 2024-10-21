import argparse, subprocess, importlib, json
from Brian2.get_brian_dir_path_head import *
general_tools = importlib.import_module(path_head+'.tools.general_tools')
tools_names = [x for x in general_tools.__dict__ if not x.startswith("_")]
globals().update({k: getattr(general_tools, k) for k in tools_names})
from distutils.util import strtobool

#
BRIAN_DIR = "${BRIAN_DIR}"
JOB_NAME = "${JOB_NAME}"
OUTPUT = "${OUTPUT}"
ERROR = "${ERROR}"
NUM_NODES = "${NUM_NODES}"
GPUS_PER_TASK = "${GPUS_PER_TASK}"
CPUS_PER_TASK = "${CPUS_PER_TASK}"
MEM= "${MEM}"
TOTAL_WALL_TIME_SLURM = "${TOTAL_WALL_TIME_SLURM}"
TOTAL_WALL_TIME_RAY = "${TOTAL_WALL_TIME_RAY}"
SEARCH_ALGO = "${SEARCH_ALGO}"
TORCHVISION_DATA = "${TORCHVISION_DATA}"
SQRT_GROUP_SIZE = "${SQRT_GROUP_SIZE}"
IDX_RANGE_TRAIN = "${IDX_RANGE_TRAIN}"
IDX_RANGE_VALID = "${IDX_RANGE_VALID}"
IDX_RANGE_TEST = "${IDX_RANGE_TEST}"
TASK = "${TASK}"
NUM_WORKERS = "${NUM_WORKERS}"
N_EPOCH = "${N_EPOCH}"
FIXED_SEED = "${FIXED_SEED}"
TEST_OPTION = "${TEST_OPTION}"
TRAIN= "${TRAIN}"
TEST= "${TEST}"

# basic input
parser = argparse.ArgumentParser(description='basic input')
parser.add_argument('--base_dir',  required=True, type=str)
parser.add_argument('--job_name',  required=True, type=str)
parser.add_argument('--log_out',  required=True, type=str)
parser.add_argument('--err_out',  required=True, type=str)
parser.add_argument('--num_nodes',  required=True, type=int)
parser.add_argument('--gpus_per_task',  required=True, type=int)
parser.add_argument('--cpus_per_task',  required=True, type=int)
parser.add_argument('--mem',  required=True, type=str)
parser.add_argument('--total_wall_time_slurm',  required=True, type=str)
parser.add_argument('--account',  required=True, type=str)
parser.add_argument('--torchvision_data', required=True, type=str, help='name of the torchvision dataset, e.g. MNIST')
parser.add_argument('--search_algo', type=str, required=False, default="HyperOptSearch",  
                    help = "options: SkOptSearch, HyperOptSearch, BayesOptSearch, AxSearch, HEBOSearch, OptunaSearch")
parser.add_argument('--sqrt_grp_size', required=True, type=json.loads, 
                    help='sqrt size of network, e.g. "[3, 1]", means 1st/2nd hidden layer has 9 and 1 neurons in each group, respectively')
parser.add_argument('--idx_range_train', required=True, type=json.loads, 
                    help='index range for training samples, e.g. "[0, 1000]"')
parser.add_argument('--idx_range_valid', required=False, type=json.loads, default=None, 
                    help='index range for validation samples, e.g. "[0, 1000]"')
parser.add_argument('--idx_range_test', required=True, type=json.loads, 
                    help='index range for testing samples, e.g. "[0, 1000]"')
parser.add_argument('--task', required=True, type=str, help='option: ray.train, ray.tune')
parser.add_argument('--num_workers', required=False, type=int, default=1, help='number of workers')
parser.add_argument('--n_epoch', required=False, default=1, type=int, help='number of epoch')
parser.add_argument('--test_option',  required=False, default=None, type=str, help='option: None, add_more_neuron,  add_more_syn, use_rank0_info')
parser.add_argument('--train',  required=False, default=True, type=lambda x: bool(strtobool(x)), help='options:  True or False, i.e. train or not')
parser.add_argument('--test',  required=False, default=True, type=lambda x: bool(strtobool(x)), help='options:  True or False, i.e. test or not')
parser.add_argument('--fixed_seed',  required=False, default=False, type=lambda x: bool(strtobool(x)), help='options:  True or False, i.e. each worker process the same image or not')

args = parser.parse_args()

job_name = args.job_name
log_out = "{}_{}".format(args.log_out, time.strftime("%m%d-%H%M", time.localtime()))
err_out = "{}_{}".format(args.err_out, time.strftime("%m%d-%H%M", time.localtime()))

# translate slurm wall time to the time in seconds
total_wall_time_ray = str(get_time_in_seconds_from_slurm_time(args.total_wall_time_slurm))

# read the template and replace keywords
with open('ray_cluster_script.template', "r") as f:
    text = f.read()

text = text.replace(BRIAN_DIR, brian_dir)
text = text.replace(JOB_NAME, job_name)
text = text.replace(OUTPUT, log_out)
text = text.replace(ERROR, err_out)
text = text.replace(NUM_NODES, str(args.num_nodes))
text = text.replace(GPUS_PER_TASK, str(args.gpus_per_task))
text = text.replace(CPUS_PER_TASK, str(args.cpus_per_task))
text = text.replace(MEM, args.mem)
text = text.replace(TOTAL_WALL_TIME_SLURM, args.total_wall_time_slurm)
text = text.replace(TOTAL_WALL_TIME_RAY, total_wall_time_ray)
text = text.replace(SEARCH_ALGO, args.search_algo)
text = text.replace(TORCHVISION_DATA, args.torchvision_data)
text = text.replace(SQRT_GROUP_SIZE, "'"+json.dumps(args.sqrt_grp_size)+"'")
text = text.replace(IDX_RANGE_TRAIN, "'"+json.dumps(args.idx_range_train)+"'")
text = text.replace(IDX_RANGE_VALID, "'"+json.dumps(args.idx_range_valid)+"'")
text = text.replace(IDX_RANGE_TEST, "'"+json.dumps(args.idx_range_test)+"'")
text = text.replace(NUM_WORKERS, str(args.num_workers))
text = text.replace(N_EPOCH, str(args.n_epoch))
text = text.replace(FIXED_SEED, str(args.fixed_seed))
text = text.replace(TEST_OPTION, str(args.test_option))
text = text.replace(TRAIN, str(args.train))
text = text.replace(TEST, str(args.test))
text = text.replace(TASK, args.task)

if os.path.exists(args.base_dir) == False:
    os.makedirs(args.base_dir)
os.chdir(args.base_dir) # go to the base_dir

# save slurm job script
script_file = "{}.sh".format(job_name)
with open(script_file, "w") as f:
    f.write(text)

# submit
subprocess.Popen(["sbatch", "-A", args.account, script_file])
print("Job submitted! Directory: {}. Script:{}. Log: {}".format(os.getcwd(), script_file, "{}.log".format(job_name)))
