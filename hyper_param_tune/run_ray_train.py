from Brian2.import_common_module import *

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0" #keep originial working directory

ip_head = os.getenv("ip_head")
if ip_head == None:
    print('-- running locally ---')

# Model Definition
class SpikeNetwork(nn.Module):
    def __init__(self, config: Dict):
        super(SpikeNetwork, self).__init__()
        brian2.defaultclock.dt = input_param.defaultclock_dt 
        print('===: defaultclock.dt: ', brian2.defaultclock.dt)

        self.sqrt_grp_size = config['args'].sqrt_grp_size
        self.debug = config['args'].debug
        if len(self.sqrt_grp_size)==1:
            self.N_hidden = 'base'
        else:
            self.N_hidden = f'{str(len(self.sqrt_grp_size))}h'

        self.test_option = config['args'].test_option
        if self.test_option=='None':
            self.test_option=eval(self.test_option) # convert it back to NoneType

        self.hyperparam = self.get_hyperparams()

    # fixed hyperparameters for this model
    def get_hyperparams(self):
        fixed_hyperpar_file = os.path.join(
                brian_dir,
                f'jobs_sub/ray_tune/fixed_hyperpar_file_{self.N_hidden}')

        if os.path.isdir(fixed_hyperpar_file) == False:
            sys.exit(f'!!!! fixed hyperpar {fixed_hyperpar_file} not found !!!!')

        if self.test_option == 'add_more_neuron_div':
            print('---- syn from each worker has different hyperparam ----')
            worker_index = train.get_context().get_world_rank()
        else:
            print('---- all syn has the same hyperparam from rank 0 -----')
            worker_index = 0

        hyperparam = read_data(dir_name=fixed_hyperpar_file,
                                 worker_index=worker_index,
                                 data_format='json',
                                 single_value=True)

        print('--- model hyperpar loaded from ', fixed_hyperpar_file)

        return hyperparam

    # define the input parameters
    def set_args(self, task, epoch, label, torchvision_data, root_out=True):
        test_option = self.test_option
        if task == 'train':  # no test option during training
            test_option = None

        self.label = label
        self.task = task
        self.torchvision_data = torchvision_data
        self.root_out = root_out

        self.model = Spike_MNIST_Nlayer(
                task = task,
                idx_start_train = 0,
                idx_start = 0,
                idx_end = nsample_for_task[task],
                simulation_duration = input_param.sim_time,
                epoch = epoch,
                previous_seg_name = f'seg_0_{nsample_for_task["train"]}',
                sqrt_grp_size = self.sqrt_grp_size, 
                test_option = test_option,
                hyperParams = self.hyperparam,
                debug = self.debug,
                activate_input_spikemon = False, 
                root_out = self.root_out)

    # network architecture
    def forward(self, x): 
        # When making adv_sample in spsa, x is expanded beyond the original size to 
        # calculate gradient for each digit, thus y need to be expanded accordingly.
        if self.task == 'adv_make':
            y = torch.tensor([-1]*len(x)) 
        else:
            y = self.label

        input_data = self.torchvision_data.torch_tensor_to_numpy(x, y)

        list_counts_for_digit_last_layer, _, _ = self.model.run(input_data)

        normalized_list_counts_for_digit_last_layer = []
        for row in list_counts_for_digit_last_layer:
            norm_row = [r/sum(row['count']) for r in row['count']]
            normalized_list_counts_for_digit_last_layer.append(norm_row)

        # convert to tensor, needed for adversarial attack
        return torch.tensor(normalized_list_counts_for_digit_last_layer, 
                            dtype=torch.float)

# get data
def get_torchvision_data(args):
    torchvision_data = fetch_torchvision_data(
            idx_start_train = args.idx_range_train[0], idx_end_train = args.idx_range_train[1],
            idx_start_valid=0,  idx_end_valid=0,
            idx_start_test = args.idx_range_test[0], idx_end_test = args.idx_range_test[1],
            which_data = args.torchvision_data,
            frac_val = 0.0, fixed_seed = args.fixed_seed,
            num_workers = args.num_workers) # no validation is needed for training


    dict_trainset, _, dict_testset = torchvision_data.get_data_numpy()
    return torchvision_data

def cal_eff(tensors, y):
    eff = 0
    eff_mult_match = 0
    for t, label in zip(tensors, y):
        if t[label] == torch.max(t):
            eff += 1
            if torch.sum(t==torch.max(t))>1:
                eff_mult_match += 1
    eff /= len(tensors)
    eff_mult_match /= len(tensors)

    return eff, eff_mult_match

# for each worker
def train_func_per_worker(config: Dict):
    n_epoch = config["n_epoch"]
    batch_size_train = config["batch_size_train_per_worker"]
    batch_size_test = config["batch_size_test_per_worker"]

    # Get dataloaders inside the worker training function
    torchvision_data = get_torchvision_data(config["args"])
    train_dataloader, test_dataloader = torchvision_data.get_dataloader(batch_size_train=batch_size_train, batch_size_test=batch_size_test)

    # [1] Prepare Dataloader for distributed training
    # Shard the datasets among workers and move batches to the correct device
    # =======================================================================
    train_dataloader = train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = train.torch.prepare_data_loader(test_dataloader)

    # [2] Prepare and wrap your model with DistributedDataParallel
    # Move the model to the correct GPU/CPU device
    # ============================================================
    model = SpikeNetwork(config)
    model = train.torch.prepare_model(model, parallel_strategy=None)

    # mean test efficiency from all workers
    mean_train = torchmetrics.MeanMetric()
    mean_test = torchmetrics.MeanMetric()
    mean_advatk = torchmetrics.MeanMetric()
    mean_train_mult_match = torchmetrics.MeanMetric()
    mean_test_mult_match = torchmetrics.MeanMetric()
    mean_advatk_mult_match = torchmetrics.MeanMetric()

    # Model training loop
    eff_train = 0
    eff_test = 0
    eff_advatk = 0
    eff_mult_match_train = 0
    eff_mult_match_test = 0
    eff_mult_match_advatk = 0
    for epoch in range(n_epoch):
        if epoch == n_epoch - 1: # only record the last epoch
            metrics = {'epoch': epoch,
                       'worker_id': train.get_context().get_world_rank()}

        if config["args"].train == True:
            print('--- training: epoch', epoch)
            for x, y in train_dataloader: # len(train_dataloader)=1 by design to avoid complications
                model.set_args('train', epoch, y, torchvision_data)
                output_tensor = model(x)
                eff_train, eff_mult_match_train = cal_eff(output_tensor, y)

                # mean result from all workers for the last epoch
                if epoch == n_epoch-1:
                    mean_train(eff_train)
                    mean_train_mult_match(eff_mult_match_train)
                    mean_eff_train = mean_train.compute().item()
                    mean_eff_train_mult_match = mean_train_mult_match.compute().item()
                    metrics.update({'eff_train': eff_train,
                                    'eff_mult_match_train': eff_mult_match_train,
                                    'mean_eff_train': mean_eff_train, 
                                    'mean_eff_train_mult_match': mean_eff_train_mult_match}) 

        if epoch != n_epoch-1: # do test only in the last epoch to save time
            brian2.device.delete(force=True)
            print(f'=====>  no testing or advatk for epoch {epoch}. They are done only in the last epoch. <=====')
            continue

        if config["args"].test == True:
            print('--- testing: epoch', epoch)
            for x, y in test_dataloader: # len(test_dataloader)=1 by design to avoid complications
                model.set_args('test', epoch, y, torchvision_data)
                output_tensor = model(x)
                eff_test, eff_mult_match_test = cal_eff(output_tensor, y)

                # mean result from all workers for the last epoch
                if epoch == n_epoch-1:
                    mean_test(eff_test)
                    mean_test_mult_match(eff_mult_match_test)
                    mean_eff_test = mean_test.compute().item()
                    mean_eff_test_mult_match = mean_test_mult_match.compute().item()
                    metrics.update({'eff_test': eff_test, 
                                    'eff_mult_match_test': eff_mult_match_test,
                                    'mean_eff_test': mean_eff_test, 
                                    'mean_eff_test_mult_match': mean_eff_test_mult_match}) 

            if config["args"].save_test_data == True:
                import pickle
                data_to_save = torchvision_data.torch_tensor_to_numpy(x, y)
                with open(f'test_data_rank{train.get_context().get_world_rank()}.pkl', 'wb') as f:
                    pickle.dump(data_to_save, f)

        if config["args"].advatk!= None:
            print('--- advatk: epoch', epoch)
            for x, y in test_dataloader: # len(test_dataloader)=1 by design to avoid complications
                model.set_args('adv_make', epoch, y, torchvision_data, root_out=False) # for adv_make no root output
                if config["args"].advatk=='spsa':
                    print('--- adv_make: spsa @ epoch', epoch)
                    adv_x = spsa(model_fn = model, 
                             x = x, 
                             eps = 2.0, 
                             nb_iter = 50, 
                             norm = np.inf, 
                             spsa_samples = 4000, 
                             spsa_iters = 3) 
                elif config["args"].advatk=='hsja':
                    print('--- adv_make: hsja @ epoch', epoch)
                    adv_x = hop_skip_jump_attack(model_fn = model, 
                                             x = x, 
                                             norm = np.inf,
                                             y_target=None,
                                             image_target=None)
                elif config["args"].advatk=='noise':
                    print('--- adv_make: noise @ epoch', epoch)
                    adv_x = noise(x = x, eps = 0.3)

                # evaluation need root output
                model.set_args('advatk', epoch, y, torchvision_data)
                output_tensor = model(adv_x)
                eff_advatk, eff_mult_match_advatk = cal_eff(output_tensor, y)

                # mean result from all workers for the last epoch
                if epoch == n_epoch-1:
                    mean_advatk(eff_advatk)
                    mean_advatk_mult_match(eff_mult_match_advatk)
                    mean_eff_advatk = mean_advatk.compute().item()
                    mean_eff_advatk_mult_match = mean_advatk_mult_match.compute().item()
                    metrics.update({'eff_advatk': eff_advatk, 
                                    'eff_mult_match_advatk': eff_mult_match_advatk,
                                    'mean_eff_advatk': mean_eff_advatk, 
                                    'mean_eff_advatk_mult_match': mean_eff_advatk_mult_match})

        print('metrics', metrics)

        #clean up cpp_standalone
        brian2.device.delete(force=True)

        # [3] Report metrics to Ray Train
        # ===============================
        train.report(metrics)

# create all necessary directories
def create_data_directories(config):
    print('--- start creating data directories ---')
    task_list = []
    if config['args'].train == True:
        task_list.append('train')
    if config['args'].test == True:
        task_list.append('test')
        if args.advatk != None:
            task_list.append('advatk') # not for adv_make because no root output is needed for task='adv_make'

    for epoch in range(config["n_epoch"]):
        for task in task_list:
            Spike_MNIST_Nlayer(task = task,
                               idx_start_train = 0,
                               idx_start = 0,
                               idx_end = nsample_for_task[task],
                               simulation_duration = 0,
                               epoch = epoch,
                               previous_seg_name = f'seg_0_{nsample_for_task["train"]}',
                               sqrt_grp_size = config["args"].sqrt_grp_size,
                               test_option = None,
                               hyperParams = None,
                               debug = False,
                               activate_input_spikemon = False, 
                               root_out = True)
    print('--- data directories created ---')


def train_torchvision_data(args):
    #
    train_config = {
        "n_epoch": args.n_epoch,
        "batch_size_train_per_worker": batch_size_train_per_worker,
        "batch_size_test_per_worker": batch_size_test_per_worker,
        "args": args
    }

    # create all necessary directories
    if args.train == True: # when args.train == False, it means it's already created
        create_data_directories(train_config)

    # Configure computation resources
    scaling_config = ScalingConfig(num_workers=args.num_workers, use_gpu=False)
    raw_log_dir = os.path.join(os.getcwd(), "ray_log")
    run_config = train.RunConfig(name='train', storage_path=raw_log_dir, verbose=args.verbose)

    # Initialize a Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config,
        torch_config=TorchConfig(timeout_s=18000) # increase it from 30 min to 5 hours
    )
    # [4] Start distributed training
    # Run `train_func_per_worker` on all workers
    # =============================================
    result = trainer.fit()
    print(f"Training result: {result}")

if __name__ == "__main__":
    # argparse input
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx_range_train', required=False, type=json.loads,
                        help='start and end index for training, e.g. "[0, 100]" for the first 100 samples', default=[0, 10])
    parser.add_argument('--idx_range_test', required=False, type=json.loads,
                        help='start and end index for training, e.g. "[0, 100]" for the first 100 samples', default=[0, 10])
    parser.add_argument('--torchvision_data', required=True, type=str, help='name of the torchvision dataset, e.g. MNIST')
    parser.add_argument('--sqrt_grp_size', required=True, type=json.loads, help='sqrt size of network, e.g. "[3, 1]", means 1st/2nd hidden layer has 9 and 1 neurons in each group, respectively')
    parser.add_argument('--debug',  required=False, default=False, type=lambda x: bool(strtobool(x)), help='options:  True or False')
    parser.add_argument('--verbose',  required=False, default=0, type=int, help='options: 0, 1, 2')
    parser.add_argument('--num_workers', required=True, type=int, help='number of workers')
    parser.add_argument('--n_epoch', required=False, default=1, type=int, help='number of epoch')
    parser.add_argument('--test_option',  required=False, default=None, type=str, help='option: None, add_more_neuron, add_more_neuron_div, add_more_syn, use_rank0_info')
    parser.add_argument('--advatk',  required=False, default=None, type=str, help='option: None, spsa, hsja, noise')
    parser.add_argument('--train',  required=False, default=True, type=lambda x: bool(strtobool(x)), help='options:  True or False, i.e. train or not')
    parser.add_argument('--test',  required=False, default=True, type=lambda x: bool(strtobool(x)), help='options:  True or False, i.e. test or not')
    parser.add_argument('--fixed_seed',  required=False, default=False, type=lambda x: bool(strtobool(x)), help='options:  True or False, i.e. each worker process the same image or not')
    parser.add_argument('--save_test_data',  required=False, default=False, type=lambda x: bool(strtobool(x)), help='options:  True or False, i.e. save test data from each working to plot the image')

    try:
        args = parser.parse_args()
    except:
        sys.exit('!!!!! incorrect argument or missing argument !!!!!')

    if args.debug == False:
        level = logging.INFO
        logging.getLogger("ray.data").setLevel(level)
        logging.getLogger("ray.data").setLevel(level)
        logging.getLogger("ray.tune").setLevel(level)
        logging.getLogger("ray.rllib").setLevel(level)
        logging.getLogger("ray.train").setLevel(level)
        logging.getLogger("ray.serve").setLevel(level)
        logging.getLogger("ray").setLevel(level)

    #
    batch_size_train_per_worker =  (args.idx_range_train[1]-args.idx_range_train[0])//args.num_workers
    batch_size_test_per_worker =  (args.idx_range_test[1]-args.idx_range_test[0])//args.num_workers

    nsample_for_task = {'train': batch_size_train_per_worker, 
                        'test': batch_size_test_per_worker, 
                        'adv_make': batch_size_test_per_worker, 
                        'advatk': batch_size_test_per_worker}

    train_torchvision_data(args)

    # merge ROOT files from all ranks
    if args.train == True:
        root_file_train = find_files_with_partial_names(os.path.join(os.getcwd(), 'root_file'), 'train')
        merge_root_file(root_file_train)
    if args.test == True:
        root_file_test = find_files_with_partial_names(os.path.join(os.getcwd(), 'root_file'), 'test')
        merge_root_file(root_file_test)
        if args.advatk != None:
            root_file_advatk = find_files_with_partial_names(os.path.join(os.getcwd(), 'root_file'), 'advatk')
            merge_root_file(root_file_advatk)
