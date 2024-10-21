#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# these command make the cell width wider than default
from IPython.display import display, HTML
display(HTML("<style>:root { --jp-notebook-max-width: 100% !important; }</style>"))


# In[ ]:


from Brian2.import_common_module import *


# In[ ]:


def get_hyperparams(n_hlayer):
    try:
        fixed_hyperpar_file = os.path.join(brian_dir, 'jobs_sub/ray_tune/fixed_hyperpar_file_2h')
        hyperparam = read_data(dir_name=fixed_hyperpar_file,
                               worker_index=0,
                               data_format='json',
                               single_value=True)
        print('--- model hyperpar loaded from ', fixed_hyperpar_file)
    except:
        sys.exit(f'!!!  {fixed_hyperpar_file} not found ---')
    
    return hyperparam


# In[ ]:


if __name__ == '__main__':
    debug = False
    if debug == False:
        logger = logging.getLogger("ray.data")
        logger.setLevel(logging.CRITICAL)
        
    activate_input_spikemon = False  # save input_spikes, only for debugging
    idx_start_train = 0
    idx_end_train = 1000
    idx_start_valid = 0 
    idx_end_valid = 1000
    idx_start_test = 0
    idx_end_test = 1000
    n_grp_train = 1 # e.g. n_grp_train = 2 for 0-1000, means run 0-500, 500-100 consecutively.
    n_epoch = 1 
    sqrt_grp_size = [1,1]  #sqrt(size) for each group in a layer
    test_option = None
    task_list = ['train', 'valid', 'test']
    
    n_hlayer = len(sqrt_grp_size)
    hyperparams = get_hyperparams(n_hlayer)
    
    # MNIST data within given range
    torchvision_data = fetch_torchvision_data(idx_start_train, idx_end_train,
                       idx_start_valid, idx_end_valid,
                       idx_start_test, idx_end_test, 'MNIST')
    data_train, data_valid, data_test = torchvision_data.get_data_numpy()
    
    starttime = time.time()
    nsample_train_per_grp = int((idx_end_train-idx_start_train)/n_grp_train) # n sample in each group
    idx_start_prev = 0 # idx_start of previous group
    idx_end_prev = 0 # idx_end of previous group
    for epoch in range(n_epoch):
        for igrp in range(n_grp_train):
            #idx_start and idx_end of each group
            idx_start = int(igrp*nsample_train_per_grp) 
            idx_end = idx_start + nsample_train_per_grp
            previous_seg_name = 'seg_'+str(idx_start_prev)+'_'+str(idx_end_prev)
            if idx_start >= len(data_train['label']): # run out of data
                break
            print(f'--------  epoch: {epoch}, group: {igrp} ------------')
            data_train_subgrp = {key: value[idx_start:idx_end] for key, value in data_train.items()}
            input_data = {'train':data_train_subgrp, 'valid': data_valid, 'test': data_test}
            for task in task_list:
                model = Spike_MNIST_Nlayer(
                    task = task,
                    idx_start_train = idx_start_train,
                    idx_start = idx_start,
                    idx_end = idx_end, 
                    simulation_duration = input_param.sim_time,
                    epoch = epoch,
                    previous_seg_name = previous_seg_name,
                    sqrt_grp_size = sqrt_grp_size,
                    test_option = test_option,
                    hyperParams = hyperparams,
                    debug = debug,
                    activate_input_spikemon = False,
                    root_out = True
                )

                _, eff_last_layer, eff_mult_match_last_layer = model.run(input_data[task])
            
                print(f'''
                --- efficiency for {task} ---
                eff_last_layer: {eff_last_layer}, eff_mult_match_last_layer: {eff_mult_match_last_layer}
                ''')
            # record indices from previous groups
            idx_start_prev = idx_start + idx_start_train
            idx_end_prev = idx_start_prev + nsample_train_per_grp
            if idx_end_prev > idx_start_train+len(data_train['label']):
                idx_end_prev = idx_start_train+len(data_train['label'])
                
    # clean all the unused files after all epoch runs
    brian2.device.delete(force=True)
    
    endtime = time.time()
    print('total time: ', endtime-starttime)


# In[ ]:




