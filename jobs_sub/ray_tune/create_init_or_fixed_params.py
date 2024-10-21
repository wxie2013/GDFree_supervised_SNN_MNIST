import ROOT, argparse, json, os
from array import *

class create_init_or_fixed_params():
    def __init__(self, N_hidden, min_eff_diff, min_eff_valid, dir_in, epoch_min):
        self.root_file = os.path.join(dir_in, 'result.root')
        self.txt_file = os.path.join(dir_in, 'result.txt')
        self.min_eff_diff = min_eff_diff
        self.min_eff_valid = min_eff_valid
        self.epoch_min = epoch_min
        sqrtgrp = dir_in[dir_in.index('sqrtgrp'):dir_in.index('_idxtrain')]
        self.fixed_hyperpar_file = f'fixed_hyperpar_file_{N_hidden}_{sqrtgrp}'
        if os.path.exists(self.fixed_hyperpar_file) == False:
            os.makedirs(self.fixed_hyperpar_file)

    # find the index with eff_valid >= min_eff_valid and eff_valid-eff_valid_mult>min_eff_diff
    def get_eff_index(self, nt, min_eff_diff, min_eff_valid, epoch_min):
        eff_valid = array('f', [0])
        eff_valid_mult = array('f', [0])
        index = array('f', [0]) 
        epoch = array('f', [0])
        nt.SetBranchAddress("eff_valid", eff_valid)
        nt.SetBranchAddress("eff_valid_mult", eff_valid_mult)
        nt.SetBranchAddress("index", index)
        nt.SetBranchAddress("epoch", epoch)
        
        result_dict = []
        for i in range(nt.GetEntries()):
            nt.GetEntry(i)
            eff_diff = eff_valid[0] - eff_valid_mult[0]
            if eff_diff >= min_eff_diff and eff_valid[0]>min_eff_valid and epoch[0] >= epoch_min:
                result_dict.append({
                    'index': int(index[0]),
                    'eff_valid': eff_valid[0],
                    'eff_valid_mult': eff_valid_mult[0],
                    'epoch': epoch[0]
                })
    
        # sort in ascending order with eff_valid
        result_dict = sorted(result_dict, key=lambda x:x['eff_valid'], reverse=True)
        return result_dict

    # create list of initial parameters for ray.tune
    def filter_list(self):
        f = ROOT.TFile(self.root_file)
        eff_index_list = self.get_eff_index(f.Get("nt"), self.min_eff_diff, self.min_eff_valid, self.epoch_min)
        with open(self.txt_file) as ftxt:
            data_txt = ftxt.read()
    
        self.initial_params_list = []
        for eff_index in eff_index_list:
            print('--: ', eff_index)
            idx1 = data_txt.index('index: '+str(eff_index['index']))
            idx2 = data_txt[idx1:].index('{') +idx1
            idx3 = data_txt[idx2:].index('}') + idx2 + 1
            init_par_dict = json.loads(data_txt[idx2:idx3])
            if init_par_dict['n_syn'] <=5:
                self.initial_params_list.append(init_par_dict)
    
    # write the list of initial params for ray.tune
    def create_init_param(self):
        with open("initial_params.json", "w") as fp:
            json.dump(self.initial_params_list, fp)

    # create list of initial parameters for ray.tune
    def create_fixed_hyperpar_file(self):
        os.chdir(self.fixed_hyperpar_file)
        for i, par in enumerate(self.initial_params_list):
            with open(f'rank{i}.json', "w") as fp:
                json.dump(par, fp)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_eff_diff", type=float, required=True, help="min eff_valid - eff_valid_mult")
    parser.add_argument("--min_eff_valid", type=float, required=True, help="min eff_valid")
    parser.add_argument("--task", type=str, required=True, default='fixed', help="fixed or init")
    parser.add_argument("--dir", type=str, required=True,  help="input file directory")
    parser.add_argument("--N_hidden", type=str, required=True,  help="base, 2h")
    parser.add_argument("--epoch_min", type=int, required=False, default=0,  help="min epoch")
    args = parser.parse_args()
   
    cif = create_init_or_fixed_params(
            args.N_hidden, args.min_eff_diff, args.min_eff_valid, args.dir, args.epoch_min)

    cif.filter_list()
    if args.task=='init':
        cif.create_init_param()
    elif args.task=='fixed':
        cif.create_fixed_hyperpar_file()
