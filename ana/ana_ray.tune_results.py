import argparse, ROOT, os, json, math
from tqdm import tqdm
from Brian2.get_brian_dir_path_head import *
input_param = importlib.import_module(path_head+'.jobs_sub.ray_tune.input_param')

#-------------------------------------------------------------------------------------------
class plot_ray_data():
    def __init__(self, ray_log_dir, out_file_name):
        self.ray_log_dir = ray_log_dir

        #
        self.dfs = None

        out_file_name = os.path.join(ray_log_dir, out_file_name)
        self.result = ROOT.TFile(out_file_name+".root", "RECREATE")
        self.result_text = open(out_file_name+".txt", "w")

        self.index = 0
        self.data = []
        self.n_color = 10
        self.n_plot_before_clear = self.n_color
        self.time_refresh = 0.01
        self.col_epoch = []
        self.col_eff_test = []
        self.col_eff_test_mult = []
        self.col_eff_valid = []
        self.col_eff_valid_mult = []
        self.col_eff_train = []
        self.col_eff_train_mult= []
        self.last_trial_id = ''

    # define ntuple structure 
    def setup_nt(self):
        self.nt_var_value = ROOT.std.vector('<float>')()

        nt_var = 'index:epoch'
        nt_var += ':eff_train:eff_train_mult'
        nt_var += ':eff_valid:eff_valid_mult'
        nt_var += ':eff_test:eff_test_mult'
        nt_var += ':max_dendritic_delay'
        nt_var += ':n_syn'
        nt_var += ':stdp_type'
        nt_var += ':N_hidden'
        nt_var += ':sigma_noise'
        nt_var += ':v_thres_exci'
        nt_var += ':v_reversal_e_exci'
        nt_var += ':v_reversal_i_exci'
        nt_var += ':v_rest_exci'
        nt_var += ':v_reset_exci'
        nt_var += ':refrac_time_exci'
        nt_var += ':switch_norm'
        nt_var += ':sim_time'
        nt_var += ':w_sat_scale'
        nt_var += ':vt_sat_scale'
        nt_var += ':w_sat_shift'
        nt_var += ':vt_sat_shift'
        for i in range(self.n_hlayer):
            if i < self.n_hlayer-1: # in between layers
                nt_var += ':max_delay_efe'+str(i)
                nt_var += ':gmax_efe'+str(i)
                nt_var += ':penalty_efe'+str(i)
                nt_var += ':norm_scale_S_efe'+str(i)

            nt_var += ':norm_scale_S_input2e'+str(i)
            nt_var += ':gmax_input2e'+str(i)
            nt_var += ':penalty_input2e'+str(i)
            nt_var += ':max_delay_input2e'+str(i)

            nt_var += ':dW_e2e'+str(i)
            nt_var += ':delta_vt'+str(i)
            nt_var += ':tau_adpt'+str(i)
            nt_var += ':tau_ge'+str(i)
            nt_var += ':tau_gi'+str(i)
            nt_var += ':tau_membrane_exci'+str(i)

        if self.stdp_type==0:
            nt_var += ':taupre'
            nt_var += ':taupost'
            nt_var += ':d_Apre'
            nt_var += ':d_Apost_scale'
        elif self.stdp_type==1:
            nt_var += ':nu_pre_ee'
            nt_var += ':nu_post_ee'
            nt_var += ':tc_pre_ee'
            nt_var += ':tc_post_1_ee'
            nt_var += ':tc_post_2_ee'

        self.nt = ROOT.TNtuple("nt", "", nt_var)

    # fill ntuple
    def fill_nt(self, idx, d):
        for ep, eff_valid, eff_valid_mult, eff_train, eff_train_mult, eff_test, eff_test_mult in zip(d['epoch'], d['eff_valid'], d['eff_valid_mult'], d['eff_train'], d['eff_train_mult'], d['eff_test'], d['eff_test_mult']):
            self.nt_var_value.clear()
            self.nt_var_value.push_back(idx)
            self.nt_var_value.push_back(float(ep))
            self.nt_var_value.push_back(float(eff_train))
            self.nt_var_value.push_back(float(eff_train_mult))
            self.nt_var_value.push_back(float(eff_valid))
            self.nt_var_value.push_back(float(eff_valid_mult))
            self.nt_var_value.push_back(float(eff_test))
            self.nt_var_value.push_back(float(eff_test_mult))
            self.nt_var_value.push_back(float(d['params']['max_dendritic_delay']))
            self.nt_var_value.push_back(float(d['params']['n_syn']))
            self.nt_var_value.push_back(float(d['params']['stdp_type']))
            self.nt_var_value.push_back(self.n_hlayer)
            self.nt_var_value.push_back(float(d['params']['sigma_noise']))
            self.nt_var_value.push_back(float(d['params']['v_thres_exci']))
            self.nt_var_value.push_back(float(d['params']['v_reversal_e_exci']))
            self.nt_var_value.push_back(float(d['params']['v_reversal_i_exci']))
            self.nt_var_value.push_back(float(d['params']['v_rest_exci']))
            self.nt_var_value.push_back(float(d['params']['v_reset_exci']))
            self.nt_var_value.push_back(float(d['params']['refrac_time_exci']))
            self.nt_var_value.push_back(float(d['params']['switch_norm']))
            self.nt_var_value.push_back(input_param.sim_time)
            self.nt_var_value.push_back(float(d['params']['w_sat_scale']))
            self.nt_var_value.push_back(float(d['params']['vt_sat_scale']))
            self.nt_var_value.push_back(float(d['params']['w_sat_shift']))
            self.nt_var_value.push_back(float(d['params']['vt_sat_shift']))
            for i in range(self.n_hlayer):
                if i < self.n_hlayer-1:
                    self.nt_var_value.push_back(float(d['params']['max_delay_efe'+str(i)]))
                    self.nt_var_value.push_back(float(d['params']['gmax_efe'+str(i)]))
                    self.nt_var_value.push_back(float(d['params']['penalty_efe'+str(i)]))
                    self.nt_var_value.push_back(float(d['params']['norm_scale_S_efe'+str(i)]))

                self.nt_var_value.push_back(float(d['params']['norm_scale_S_input2e'+str(i)]))
                self.nt_var_value.push_back(float(d['params']['gmax_input2e'+str(i)]))
                self.nt_var_value.push_back(float(d['params']['penalty_input2e'+str(i)]))
                self.nt_var_value.push_back(float(d['params']['max_delay_input2e'+str(i)]))

                self.nt_var_value.push_back(float(d['params']['dW_e2e'+str(i)]))

                self.nt_var_value.push_back(float(d['params']['delta_vt'+str(i)])) 
                self.nt_var_value.push_back(float(d['params']['tau_adpt'+str(i)]))
                self.nt_var_value.push_back(float(d['params']['tau_ge'+str(i)]))
                self.nt_var_value.push_back(float(d['params']['tau_gi'+str(i)]))
                self.nt_var_value.push_back(float(d['params']['tau_membrane_exci'+str(i)]))

            if self.stdp_type==0:
                self.nt_var_value.push_back(float(d['params']['taupre']))
                self.nt_var_value.push_back(float(d['params']['taupost']))
                self.nt_var_value.push_back(float(d['params']['d_Apre']))
                self.nt_var_value.push_back(float(d['params']['d_Apost_scale']))
            elif self.stdp_type==1:
                self.nt_var_value.push_back(float(d['params']['nu_pre_ee']))
                self.nt_var_value.push_back(float(d['params']['nu_post_ee']))
                self.nt_var_value.push_back(float(d['params']['tc_pre_ee']))
                self.nt_var_value.push_back(float(d['params']['tc_post_1_ee']))
                self.nt_var_value.push_back(float(d['params']['tc_post_2_ee']))

            self.nt.Fill(self.nt_var_value.data())

    # load all trainable data
    def load_data(self):
        self.load_data_from_ray_log()
        self.setup_nt()

    # locate all trainable directories and put them into a list
    def get_trainable_dirs(self, rootdir):
        data_dir = os.path.join(rootdir, 'ray_log/tune')
        print('... reading from trainable directories of ', data_dir,'  ........')
        list_dir = []
        for it in os.scandir(data_dir):
            if it.is_dir() and 'trainable_' in it.name:
                list_dir.append(it.path)
        return list_dir

    # load data from ray_load/brian2
    def load_data_from_ray_log(self):
        self.n_hlayer = 0
        trainable_dirs = self.get_trainable_dirs(self.ray_log_dir)
        for idx, sdir in enumerate(tqdm(trainable_dirs)):
            self.col_epoch.clear()
            self.col_eff_valid.clear()
            self.col_eff_valid_mult.clear()
            self.col_eff_test.clear()
            self.col_eff_test_mult.clear()
            self.col_eff_train.clear()
            self.col_eff_train_mult.clear()
            trial_id = ''
            try:
                with open(os.path.join(sdir, 'result.json'), 'r') as f:
                    json_data = [json.loads(line) for line in f]
                    for d in json_data:
                        self.col_epoch.append(str(d['epoch']))
                        self.col_eff_valid.append(str(d['eff_valid']))
                        self.col_eff_valid_mult.append(str(d['eff_mult_match_valid']))
                        self.col_eff_test.append(str(d['eff_test']))
                        self.col_eff_test_mult.append(str(d['eff_mult_match_test']))
                        self.col_eff_train.append(str(d['eff_train']))
                        self.col_eff_train_mult.append(str(d['eff_mult_match_train']))
                        trial_id = 'trainable_'+d['trial_id']
                        params = d['config']
                        self.n_hlayer = len([k for (k,v) in params.items() if 'tau_adpt' in k])

                #
                self.data.append(
                        {
                            'index':idx, 
                            'epoch':self.col_epoch[:],
                            'eff_valid':self.col_eff_valid[:], 
                            'eff_valid_mult': self.col_eff_valid_mult[:], 
                            'eff_test':self.col_eff_test[:], 
                            'eff_test_mult': self.col_eff_test_mult[:], 
                            'eff_train':self.col_eff_train[:], 
                            'eff_train_mult': self.col_eff_train_mult[:], 
                            'trial_id':trial_id, 
                            'params':params
                            }
                        )
            except:
                pass  # tuning is not done yet

        # the following are common for all tries in this specific series
        self.stdp_type = self.data[0]['params']['stdp_type']

    # plot the result in animation
    def draw(self):
        self.c = ROOT.TCanvas("c", "", 700, 500)
        self.c.SetGridx(1)
        self.c.SetGridy(1)
        self.h = ROOT.TH1F("h", "", 100, 0, 10)
        self.h.SetMaximum(1)
        self.h.SetMinimum(0)
        self.h.Draw()
        for i, d in enumerate(self.data):
            self.fill_nt(d['index'], d)

            cond = 'index=='+str(i)
            self.nt.SetLineWidth(2)
            self.nt.SetLineColor(i%self.n_color+1)
            self.nt.Draw("eff_valid-eff_valid_mult:epoch", cond, "Lsame")
            self.c.Update()
            self.c.Modified()
            title = ['\n---------------------------------------------------\n']
            title += ['epoch    ', 'eff_valid    ',  'trial_id: ', d['trial_id'],
                    '    index: ', str(d['index']), '\n']
            self.result_text.writelines(title)
            title.remove('\n')
            print(*title)
            print('---------------------------------------------------')
            for ep, eff in zip(d['epoch'], d['eff_valid']):
                val = [ep, '     ', eff, '\n']
                self.result_text.writelines(val)
                val.remove('\n')
                print(*val)

            self.result_text.writelines(json.dumps(d['params']))

            if i%self.n_plot_before_clear == 0:
                print('================================================================')
                #ROOT.getchar()
                self.c.Clear()
                self.h.Draw()
            #time.sleep(self.time_refresh)

        self.result.cd()
        self.nt.Write();
        self.result.Close()
        self.result_text.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--ray_log_dir', type=str, required=True, help="path to ray_log/brian2")
    parser.add_argument('-f', '--out_file_name', type=str, required=True, help="output file name")
    args = parser.parse_args()
    plot = plot_ray_data(args.ray_log_dir, args.out_file_name)
    plot.load_data()
    plot.draw()
