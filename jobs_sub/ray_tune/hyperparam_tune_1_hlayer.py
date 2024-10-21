from ray import tune
import numpy as np
import json, os
from Brian2.get_brian_dir_path_head import *
input_param = importlib.import_module(path_head+'.jobs_sub.ray_tune.input_param')

# hidden layer parameters
class hpar():
    def __init__(self, sqrt_grp_size):
        self.sqrt_grp_size = sqrt_grp_size
        self.N_hidden = len(sqrt_grp_size)
        self.define_range()
        print('..network with ', self.N_hidden, ' hidden layers w/neuron',sqrt_grp_size, '..')
    #
    def define_range(self):
        self.Rg_dW_e2e = [
                [0.01, 100, 0.01],
                [0.01, 100, 0.01],
                [0.01, 100, 0.01],
                [0.01, 100, 0.01],
                [0.01, 100, 0.01],
                ]
        self.Rg_delta_vt = [
                [0.001, 0.1, 0.001],
                [0.001, 0.1, 0.001],
                [0.001, 0.1, 0.001],
                [0.001, 0.1, 0.001],
                [0.001, 0.1, 0.001],
                ]

    def get_param(self):
        # load trained hyperparameters from base model
        try:
            best_hyperPar_from_base_file = os.path.join(brian_dir, 'jobs_sub/ray_tune/best_hyperPar_from_base.json')
            with open(best_hyperPar_from_base_file, 'r') as f:
                content = f.read().replace('\n', '').replace("'", '"')
                content = content.replace('True', 'true').replace('False', 'false')

            search_space = json.loads(content)

            print('--- loaded "best_hyperPar_from_base.json" ---')
        except:
            sys.exit('!!! ERROR: cannot load "best_hyperPar_from_base.json" !!!')

        for i in range(self.N_hidden):
            search_space['dW_e2e'+str(i)] =  tune.qloguniform(
                    self.Rg_dW_e2e[i][0],
                    self.Rg_dW_e2e[i][1],
                    self.Rg_dW_e2e[i][2])

            search_space['delta_vt'+str(i)] = tune.qloguniform(
                    self.Rg_delta_vt[i][0],
                    self.Rg_delta_vt[i][1],
                    self.Rg_delta_vt[i][2]) #*brian2.mV

        return search_space
