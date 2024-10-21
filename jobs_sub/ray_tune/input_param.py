import brian2, math

###################################################          
# parameter for model training, validation and test 
###################################################
num_classes = 10 # 10 digit
Num_input_neuron = 784 # mnist data 28X28
input_rows = input_cols = math.sqrt(Num_input_neuron)

max_n_hlayer = 5  # maximum number of hidden layers, beyond which one need to add more in the hyperparam.py

sim_time = 0.5*brian2.second

############################
#  parameters for Ray Train
############################
sleep_time = 1 # seconds
max_wait_time = 300 # wait for read_data of a worker to finish writing

############################
#  parameters for Ray Tune
############################
n_searchs = 2 # number of searchs before ending the run
n_initial_points = 2 # below this, it's a random search instead of baysian
if n_searchs < n_initial_points:
    n_searchs = n_initial_points

max_concurrent = n_initial_points # set it to 1.0 will run num_samples sequentially

# how many CPUs are needed for each brian2 run in parallel. This was intended to be 
# the value of prefs.devices.cpp_standalone.openmp_threads in brian2. For small net, 
# enabling it leads to longer running time. Use this cautiously for large network.
num_cpu_per_job = 1 
if num_cpu_per_job > 1:
    brian2.prefs.devices.cpp_standalone.openmp_threads = num_cpu_per_job

data_format = 'csv'
#data_format = 'parquet'
############################
#   device to be used 
############################
device_name = 'cpp_standalone'
#device_name = 'cuda_standalone'

report_period = 600*brian2.second # how frequently to report the progress when debug==False
############################
# fixed parameters for model
############################
defaultclock_dt = 0.1*brian2.ms # Diehl paper used 0.5ms. Brian2 default is 0.1*ms.
scale_img = 0.25 #starting point of the scale

# criteria for running condition
n_spike_sample_min = 5  # each layer has at least n_spike_sample_min of spikes
d_scale_img =  0.25 # multiplicative increment segment for input neuron rate, i.e. 256/4 = 64*Hz increase in maximum rate
max_scale_img = 1 # set a maximum to avoid infinit loop in case the model parameter will produce no spikes
max_repetitions = max_scale_img/d_scale_img # max number of repetition of the same sample
