from ray import tune
import numpy as np
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
        # same for all hidden layers
        self.Rg_tau_adpt = [
                [10, 1e8, 10], 
                [10, 1e8, 10], 
                [10, 1e8, 10], 
                [10, 1e8, 10], 
                [10, 1e8, 10], 
                ]
        self.Rg_delta_vt = [
                [0.0001, 0.1, 0.0001],
                [0.0001, 0.1, 0.0001],
                [0.0001, 0.1, 0.0001],
                [0.0001, 0.1, 0.0001],
                [0.0001, 0.1, 0.0001],
                ]
        self.Rg_tau_membrane_exci = [
                [10, 200, 10],
                [10, 200, 10],
                [10, 200, 10],
                [10, 200, 10],
                [10, 200, 10],
                ]
        self.Rg_tau_ge = [
                [0.1, 10, 0.1],
                [0.1, 10, 0.1],
                [0.1, 10, 0.1],
                [0.1, 10, 0.1],
                [0.1, 10, 0.1],
                ]
        self.Rg_tau_gi = [
                [1, 10, 1],
                [1, 10, 1],
                [1, 10, 1],
                [1, 10, 1],
                [1, 10, 1],
                ]
        self.Rg_gmax_input2e = [
                [1, 100, 1], 
                [1, 100, 1], 
                [1, 100, 1], 
                [1, 100, 1], 
                [1, 100, 1], 
                ]
        self.Rg_gmax_efe = [
                [1, 100, 1], 
                [1, 100, 1], 
                [1, 100, 1], 
                [1, 100, 1], 
                [1, 100, 1], 
                ]
        self.Rg_norm_scale_S_input2e = [
                [0.001, 0.5, 0.001], 
                [0.001, 0.5, 0.001], 
                [0.001, 0.5, 0.001], 
                [0.001, 0.5, 0.001], 
                [0.001, 0.5, 0.001], 
                ]
        self.Rg_norm_scale_S_efe = [
                [0.1, 0.5, 0.1],
                [0.1, 0.5, 0.1],
                [0.1, 0.5, 0.1],
                [0.1, 0.5, 0.1],
                [0.1, 0.5, 0.1],
                ]
        self.Rg_dW_e2e = [
                [0.01, 100, 0.01],
                [0.01, 100, 0.01],
                [0.01, 100, 0.01],
                [0.01, 100, 0.01],
                [0.01, 100, 0.01],
                ]

        self.stdp_type = 1
    #
    def get_param(self):
        search_space = {
                'switch_norm':  tune.choice([True]),
                'n_syn': tune.choice([1]),
                'stdp_type': tune.choice([self.stdp_type]),
                'max_dendritic_delay': tune.choice([0]), #*brian2.ms,
                'sigma_noise': tune.choice([0]), #*brian2.mV
                'v_thres_exci': tune.choice([-52]), #*brian2.mV
                'v_reversal_e_exci': tune.choice([0]), #*brian2.mV
                'v_reversal_i_exci': tune.choice([-100]), #*brian2.mV
                'v_rest_exci': tune.choice([-65]), #*brian2.mV
                'v_reset_exci': tune.choice([-65]), #*brian2.mV
                'refrac_time_exci': tune.choice([5]), #*brian2.ms
                'w_sat_scale': tune.qloguniform(1e-3, 1.0, 1e-3),
                'w_sat_shift': tune.quniform(0, 1.0, 0.1),
                'vt_sat_scale': tune.qloguniform(1e-3, 1.0, 1e-3), 
                'vt_sat_shift': tune.quniform(0.0, 1.0, 0.1), 
                }
        for i in range(self.N_hidden):
            search_space['tau_adpt'+str(i)] = tune.choice([10, 1e2, 1e4, 1e5, 1e6,1e7, 1e8])
            search_space['delta_vt'+str(i)] = tune.qloguniform(
                    self.Rg_delta_vt[i][0], 
                    self.Rg_delta_vt[i][1], 
                    self.Rg_delta_vt[i][2]) #*brian2.mV
            search_space['tau_membrane_exci'+str(i)] =  tune.quniform(
                    self.Rg_tau_membrane_exci[i][0], 
                    self.Rg_tau_membrane_exci[i][1], 
                    self.Rg_tau_membrane_exci[i][2]) #*brian2.ms
            search_space['tau_ge'+str(i)] = tune.quniform(
                    self.Rg_tau_ge[i][0], 
                    self.Rg_tau_ge[i][1], 
                    self.Rg_tau_ge[i][2]) #*brian2.ms 
            search_space['tau_gi'+str(i)] =  tune.quniform(
                    self.Rg_tau_gi[i][0], 
                    self.Rg_tau_gi[i][1], 
                    self.Rg_tau_gi[i][2])#*brian2.ms

            if i< self.N_hidden-1:
                search_space['gmax_efe'+str(i)] = tune.choice([1])
                search_space['norm_scale_S_efe'+str(i)] = tune.choice([0])
                search_space['penalty_efe'+str(i)] = tune.choice([1])
                search_space['max_delay_efe'+str(i)] = tune.choice([0])  #*brian2.ms, 

            search_space['max_delay_input2e'+str(i)] = tune.quniform(0, 200, 10)  #*brian2.ms, 
            search_space['dW_e2e'+str(i)] =  tune.qloguniform(
                    self.Rg_dW_e2e[i][0], 
                    self.Rg_dW_e2e[i][1], 
                    self.Rg_dW_e2e[i][2]) 
            sqrt_input_neuron = (input_param.Num_input_neuron)**0.5

        # parameter for stdp_type=1. Note: diable for now for easier training
        if self.stdp_type == 1:
            self.nu_post_ee = 0.01
            search_space['nu_pre_ee'] =  tune.qloguniform(0.00001, self.nu_post_ee, 0.00001)
            search_space['nu_post_ee'] =  tune.choice([self.nu_post_ee])
            search_space['tc_pre_ee'] =  tune.choice([20]) #*brian2.ms,
            search_space['tc_post_1_ee'] =  tune.choice([20]) #*brian2.ms,
            search_space['tc_post_2_ee'] =  tune.choice([40]) #*brian2.ms,
        elif self.stdp_type == 0:
            search_space['taupre'] = 20 #*brian2.ms
            search_space['taupost'] = 20 #*brian2.ms
            search_space['d_Apre'] = 0.001
            search_space['d_Apost_scale'] = 1.05 

        n_hlayer_used = 1
        for i in range(n_hlayer_used):
            search_space['gmax_input2e'+str(i)] = tune.qloguniform(0.1, 100, 0.1)
            search_space['norm_scale_S_input2e'+str(i)] = tune.qloguniform(
                    self.Rg_norm_scale_S_input2e[i][0], 
                    self.Rg_norm_scale_S_input2e[i][1],
                    self.Rg_norm_scale_S_input2e[i][2])
            search_space['penalty_input2e'+str(i)] = tune.choice([1])

        return search_space
