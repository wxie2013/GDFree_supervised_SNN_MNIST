from ray import tune
import numpy as np, os, sys, json
from Brian2.get_brian_dir_path_head import *
from Brian2.tools.general_tools import *
input_param = importlib.import_module(path_head+'.jobs_sub.ray_tune.input_param')

# hidden layer parameters
class hpar():
    def __init__(self, sqrt_grp_size):
        self.sqrt_grp_size = sqrt_grp_size
        self.N_hidden = len(sqrt_grp_size)
        if self.N_hidden<=1:
            sys.exit(' !!! ERROR: N_hidden must be greater than 1 !!!')
        self.define_range()
        print('..network with ', self.N_hidden, ' hidden layers w/neuron',sqrt_grp_size, '..')

    #
    def define_range(self):
        self.Rg_gmax_efe = [
                [1, 100, 1],
                [1, 100, 1],
                [1, 100, 1],
                [1, 100, 1],
                [1, 100, 1],
                ]

        self.Rg_norm_scale_S_efe = [
                [0.001, 0.5, 0.001],
                [0.001, 0.5, 0.001],
                [0.001, 0.5, 0.001],
                [0.001, 0.5, 0.001],
                [0.001, 0.5, 0.001],
                ]

        self.Rg_max_delay_efe = [
                [0, 200, 10],
                [0, 200, 10],
                [0, 200, 10],
                [0, 200, 10],
                [0, 200, 10],
                ]

    def get_param(self):
        # load trained hyperparameters from 1-hlayer model 
        hyperpar_file_list = []
        hyperpar_dir = brian_dir+'/jobs_sub/ray_tune/fixed_hyperpar_file_base'
        for filename in os.listdir(hyperpar_dir):
            if not filename.endswith('.json'):
                continue
            with open(os.path.join(hyperpar_dir, filename), 'r') as f:
                content = f.read().replace('\n', '').replace("'", '"')
                content = content.replace('True', 'true').replace('False', 'false')

            hyperpar = json.loads(content)

            # expand to other hlayers with the same hyperparameters
            hyperpar = expand_dictionary(hyperpar, '0', self.N_hidden-1)
            hyperpar_file_list.append(hyperpar)
        print('--- loaded all files in  fixed_hyperpar_file_base ---')
        keys_to_expand = ['w_sat_scale', 'w_sat_shift', 'vt_sat_scale', 'vt_sat_shift', 'nu_pre_ee']
        search_space = merge_and_replace(hyperpar_file_list, keys_to_expand)
        print(search_space)
        for i in range(self.N_hidden):
            if i< self.N_hidden-1:
                search_space['gmax_efe'+str(i)] = tune.qloguniform(
                        self.Rg_gmax_efe[i][0],
                        self.Rg_gmax_efe[i][1],
                        self.Rg_gmax_efe[i][2])

                search_space['norm_scale_S_efe'+str(i)] = tune.qloguniform(
                        self.Rg_norm_scale_S_efe[i][0],
                        self.Rg_norm_scale_S_efe[i][1],
                        self.Rg_norm_scale_S_efe[i][2])
                search_space['max_delay_efe'+str(i)] = tune.choice([0])
                search_space['max_delay_efe'+str(i)] = tune.quniform(
                        self.Rg_max_delay_efe[i][0],
                        self.Rg_max_delay_efe[i][1],
                        self.Rg_max_delay_efe[i][2])  #*brian2.ms, 
                search_space['penalty_efe'+str(i)] = tune.choice([1])

        return search_space
