from Brian2.import_common_module import *

#--------------------------------------------------------------------------
def logging_setup_func():
    if args.debug == False:
        level = logging.INFO
        logging.getLogger("ray.data").setLevel(level)
        logging.getLogger("ray.data").setLevel(level)
        logging.getLogger("ray.tune").setLevel(level)
        logging.getLogger("ray.rllib").setLevel(level)
        logging.getLogger("ray.train").setLevel(level)
        logging.getLogger("ray.serve").setLevel(level)
        logging.getLogger("ray").setLevel(level)

def init_ray(args):
    tmp_dir = '/tmp/ray'
    port = os.getenv("port")
    head_node = os.getenv("head_node")
    if head_node != None:
        #delete_remote_files(head_node, int(port), tmp_dir)
        print(f'--- cleaning {tmp_dir} in {head_node} ---')
        os.system(f'ssh {head_node}; rm -r {tmp_dir}; exit')
        print('--- done ---')

        ray.init(log_to_driver=False, runtime_env={"worker_process_setup_hook": logging_setup_func}, _temp_dir=tmp_dir)
    else:
        shutil.rmtree(tmp_dir)
        print('-- running without a cluster ---')

# define search space
def search_space(sqrt_grp_size):
    return hyperparam.hpar(sqrt_grp_size).get_param()

# trainable function
def trainable(hyperparams, idx_start_train, idx_end_train, sqrt_grp_size, n_epoch, data_train=None, data_valid=None, data_test=None, debug = None, test_option=None):
    if idx_end_train > idx_start_train + len(data_train['label']):
        idx_end_train = idx_start_train + len(data_train['label'])

    task_list = ['train', 'valid', 'test']

    nsample_train_per_grp = idx_end_train - idx_start_train

    nt_content = "eff_train:eff_mult_match_train:"
    nt_content += "eff_valid:eff_mult_match_valid:"
    nt_content += "eff_test:eff_mult_match_test:"
    nt_content += "epoch"
    for epoch in range(n_epoch):
        # open efficiency summary file
        if os.path.isdir("root_file") == False:
            os.mkdir("root_file")
        eff_outfile = ROOT.TFile("root_file/eff_summary.root", "update")
        nt_last_layer = eff_outfile.Get("nt_last_layer")
        if not nt_last_layer:
            nt_last_layer = ROOT.TNtuple("nt_last_layer", "efficiency from last layer spikes", nt_content)

        # w/o delete it before each epoch, compilation will crash in ray tune
        if os.path.isdir(input_param.device_name) == True:
            subprocess.run(['rm', '-r', input_param.device_name]) 

        # process the data
        seg_name = f'seg_{idx_start_train}_{idx_end_train}'
        input_data = {'train':data_train, 'valid': data_valid, 'test': data_test}
        all_results = {'epoch':epoch}
        for task in task_list:
            model = Spike_MNIST_Nlayer(
                task = task,
                idx_start_train = idx_start_train,
                idx_start = 0,
                idx_end = len(input_data[task]['label']),
                simulation_duration = input_param.sim_time,
                epoch = epoch,
                previous_seg_name = seg_name,
                sqrt_grp_size = sqrt_grp_size,
                test_option = test_option,
                hyperParams = hyperparams,
                debug = debug,
                activate_input_spikemon = False,
                root_out = True
            )

            _, eff_last_layer, eff_mult_match_last_layer = model.run(input_data[task])
            all_results.update({f'eff_{task}':eff_last_layer, f'eff_mult_match_{task}':eff_mult_match_last_layer})
            # clean up before running the next task to avoid overflowing
            brian2.device.delete(force=True) 


        all_results.update({ 'eff_net':all_results['eff_valid']-all_results['eff_mult_match_valid']})

        print(all_results)

        nt_last_layer.Fill(all_results['eff_train'], all_results['eff_mult_match_train'], 
                           all_results['eff_valid'], all_results['eff_mult_match_valid'], 
                           all_results['eff_test'], all_results['eff_mult_match_test'], 
                           epoch)

        # save efficiency summary of this group in this epoch
        eff_outfile.cd()
        nt_last_layer.Write()
        eff_outfile.Close()

        # report the results
        train.report(all_results)


# define a tuner
def define_tuner(raw_log_dir, raw_log_name, trainable, data_train, data_valid, data_test, time_budget_s, algorithm, scheduler, debug, sqrt_grp_size, idx_start_train, idx_end_train, n_epoch, search_space = None):
    trainable_with_resources = tune.with_resources(trainable, {"cpu": input_param.num_cpu_per_job})
    tuner = tune.Tuner(
            tune.with_parameters(trainable_with_resources, 
                                 idx_start_train=idx_start_train, 
                                 idx_end_train=idx_end_train,
                                 sqrt_grp_size=sqrt_grp_size, 
                                 n_epoch=n_epoch,
                                 data_train = data_train,
                                 data_valid = data_valid,
                                 data_test = data_test,
                                 debug=debug), 
            tune_config = tune.TuneConfig(
                num_samples = input_param.n_searchs, # number of tries. Note: Brian2 is expensive
                time_budget_s = time_budget_s, # max time for the entire run
                search_alg=algorithm, 
                scheduler=scheduler
                ),
            param_space = search_space,
            # where to save the log which will be loaded later
            run_config = train.RunConfig(name=raw_log_name, storage_path=raw_log_dir, verbose=0) 
            )
    return tuner

# obtain initial parameters with promising result
def get_initial_params():
    try:
        initial_params_file = os.path.join(brian_dir, 'jobs_sub/ray_tune/initial_params.json')
        with open(initial_params_file,  'r') as f:
            initial_params = json.load(f)

        print('--- initial parameters loaded from ', initial_params_file)

    except:
        print('--- initial parameters not found ---')
        initial_params = None

    return initial_params


# define an algorithm
def define_algorithm(option, is_1st_run):
    if is_1st_run == True:
        points_to_evaluate = get_initial_params()
    else:
        points_to_evaluate=None

    if option == 'SkOptSearch':
        print('--- using SkOptSearch ---')
        algorithm = SkOptSearch(metric="eff_net", mode="max", 
                                points_to_evaluate=points_to_evaluate)
    elif option == 'BayesOptSearch':
        print('--- using BayesOptSearch ---')
        algorithm = BayesOptSearch(metric="eff_net", mode="max", 
                                   points_to_evaluate=points_to_evaluate)
    elif option == 'HyperOptSearch':
        print('--- using HyperOptSearch ---')
        algorithm = HyperOptSearch(metric="eff_net", mode="max", 
                                   n_initial_points=input_param.n_initial_points,
                                   points_to_evaluate=points_to_evaluate)
    elif option == 'AxSearch':
        print('--- using AxSearch ---')
        algorithm = AxSearch(metric="eff_net", mode="max", 
                             points_to_evaluate=points_to_evaluate)
    elif option == 'HEBOSearch':
        print('--- using HEBOSearch ---')
        algorithm = HEBOSearch(metric="eff_net", mode="max", 
                               points_to_evaluate=points_to_evaluate)
    elif option == 'OptunaSearch':
        print('--- using OptunaSearch ---')
        algorithm = OptunaSearch(metric="eff_net", mode="max", 
                                 points_to_evaluate=points_to_evaluate)
    else:
        sys.exit("Unknown algorithm: {}. can onlt be 'SkOptSearch', 'BayesOptSearch', 'HyperOptSearch', 'AxSearch', 'HEBOSearch' or 'OptunaSearch".format(option))

    algorithm = ConcurrencyLimiter(algorithm, max_concurrent=input_param.max_concurrent)

    return algorithm

# function to implement tuning
def run_tune(args, scheduler):
    # log files and check if it's the 1st time or a continuation of a tune
    raw_log_dir = os.path.join(os.getcwd(), "ray_log")
    raw_log_name = "tune"

    # input data for tuning
    torchvision_data = fetch_torchvision_data(
            args.idx_range_train[0], args.idx_range_train[1],
            args.idx_range_valid[0], args.idx_range_valid[1],
            args.idx_range_test[0], args.idx_range_test[1], 
            args.torchvision_data)
    data_train, data_valid, data_test = torchvision_data.get_data_numpy()

    #Start a Tune run and print the best result
    if os.path.exists(os.path.join(raw_log_dir, raw_log_name)) == False:
        print('--- 1st time run ----')
        algorithm = define_algorithm(args.search_algo, True)
        tuner = define_tuner(raw_log_dir, 
                             raw_log_name, 
                             trainable, 
                             data_train,
                             data_valid,
                             data_test,
                             args.time_budget_s, 
                             algorithm, 
                             scheduler, 
                             args.debug, 
                             args.sqrt_grp_size, 
                             args.idx_range_train[0], 
                             args.idx_range_train[1], 
                             args.n_epoch, 
                             search_space(args.sqrt_grp_size))   
    else:
        print('--- previous run exist, continue the tuning ----')
        algorithm = define_algorithm(args.search_algo, False)
        algorithm.restore_from_dir(os.path.join(raw_log_dir, raw_log_name))
        if args.run_option == 'contine_finished':
            print(' ... contine tuning with more samples .....')
            tuner = define_tuner(raw_log_dir, 
                                 raw_log_name, 
                                 trainable, 
                                 data_train,
                                 data_valid,
                                 data_test,
                                 args.time_budget_s, 
                                 algorithm, 
                                 scheduler, 
                                 args.debug, 
                                 args.sqrt_grp_size, 
                                 args.idx_range_train[0], 
                                 args.idx_range_train[1], 
                                 args.n_epoch)
        else:
            print('... resume or restart errored training')
            resume_errored = (args.run_option=="resume_errored")
            restart_errored = (args.run_option=="restart_errored")
            tuner = tune.Tuner.restore(
                    os.path.join(raw_log_dir, raw_log_name), 
                    resume_errored, restart_errored)

    results = tuner.fit()
    best_result = results.get_best_result(metric="eff_net", mode="max")
    print('----- best result information ------')
    print('-->1: log_dir: ', best_result.path)
    print('-->2: eff_valid: ', best_result.metrics['eff_valid'])
    print('-->3: eff_mult_match_valid: ', best_result.metrics['eff_mult_match_valid'])
    print('-->4: eff_train: ', best_result.metrics['eff_train'])
    print('-->5: eff_mult_match_train: ', best_result.metrics['eff_mult_match_train'])
    print('-->6: eff_test: ', best_result.metrics['eff_test'])
    print('-->7: eff_mult_match_test: ', best_result.metrics['eff_mult_match_test'])
    print('-->8: best config: ', best_result.config)


if __name__ == '__main__':
    '''
     - networksize: sqrt (number of neurons) in each layer 
     - time_budget_s: total ray run time in seconds
     - run_option: 
        * contine_finished: continuing with more sample 
        * resume_errored: resume errored trial 
        * restart_errored: restart errored trial
     - search_algo: which search algorithm to use
     - debug: debug mode
     - idx_range_train: start and end index for training
     - idx_range_valid: start and end index for validation
     - idx_range_test: start and end index for testing
    '''
    # argparse input 
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_budget_s', type=int, required=False, default=100000000,
            help = "max run time for the entire ray run")
    parser.add_argument('--debug',  required=False, default=False, type=lambda x: bool(strtobool(x)), help='options:  True or False')
    parser.add_argument('--use_scheduler', required=False, default=True, type=lambda x: bool(strtobool(x)), help = "options:  True or False")
    parser.add_argument('--run_option', type=str, required=False, default="contine_finished",
            help = "options:  contine_finished, resume_errored, restart_errored")
    parser.add_argument('--search_algo', type=str, required=True, 
            help = "options: SkOptSearch, HyperOptSearch, BayesOptSearch, AxSearch")
    parser.add_argument('--sqrt_grp_size', required=True, type=json.loads, 
                        help='sqrt size of network, e.g. "[3, 1]", means 1st/2nd hidden layer has 9 and 1 neurons in each group, respectively')
    parser.add_argument('--idx_range_train', required=False, type=json.loads,
                        help='start and end index for training, e.g. "[0, 100]" for the first 100 samples', default=[0, 10])
    parser.add_argument('--idx_range_valid', required=False, type=json.loads,
                        help='start and end index for training, e.g. "[0, 100]" for the first 100 samples', default=[0, 10])
    parser.add_argument('--idx_range_test', required=False, type=json.loads,
                        help='start and end index for training, e.g. "[0, 100]" for the first 100 samples', default=[0, 10])
    parser.add_argument('--n_epoch', required=False, default=1, type=int, help='number of epoch')
    parser.add_argument('--torchvision_data', required=True, type=str, help='name of the torchvision dataset, e.g. MNIST')

    try: 
        args = parser.parse_args()
    except:
        print(''' 
               'ipython code.py -- --sqrt_grp_size "[3, 1]"  --time_budget_s 10000 --run_option contine_finished' or
               'python code.py --sqrt_grp_size "[3, 1]" --time_budget_s 10000 --run_option contine_finished'
                ''')
        sys.exit(0)

    init_ray(args) # initialize ray

    # pick the right hyperparameter search space
    if len(args.sqrt_grp_size) == 1:
        if args.sqrt_grp_size[0] == 1: # base model
            hyperparam = importlib.import_module(path_head+'.jobs_sub.ray_tune.hyperparam_tune_base')
        else:
            hyperparam = importlib.import_module(path_head+'.jobs_sub.ray_tune.hyperparam_tune_1_hlayer')
    elif len(args.sqrt_grp_size) >1:
        hyperparam = importlib.import_module(path_head+'.jobs_sub.ray_tune.hyperparam_tune_multi_hlayer')

    ##
    n_hlayer = len(args.sqrt_grp_size)
    if n_hlayer == 0 or n_hlayer >input_param.max_n_hlayer:
        print(f"!!! n_layer: {n_layer} need to be in betweem 1 and {input_param.max_n_hlayer} !!!")
        sys.exit(0)

    ##
    if args.use_scheduler == False:
        scheduler = None
        print('---- no scheduler is used ----')
    else:
        scheduler = ASHAScheduler(metric="eff_net", mode="max")
        print('---- ASHAScheduler is used as the scheduler')

    # run ray tune
    run_tune(args, scheduler)
