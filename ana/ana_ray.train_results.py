import json, argparse, ROOT, math
from ROOT import TFile, TTree, gROOT, addressof, TGraphErrors, TDirectory, TGraph
from ROOT import TH1F, TLine, TCanvas, TPad, gStyle, TPaveText, TLegend, TText, gPad, TLatex
from distutils.util import strtobool
from Brian2.tools.general_tools import *

# data structure for tree creation
gROOT.ProcessLine(
"struct eff_data_t {\
   float eff_test;\
   float eff_test_mult;\
   float eff_train;\
   float eff_train_mult;\
   int nsample_train;\
   int epoch;\
   int fixed_seed;\
   std::string *sqrtgrp;\
   std::string *test_option;\
};" 

"struct detail_data_t {\
   int sidx;\
   int rank;\
   int label;\
   std::vector<int> *predict;\
   std::vector<int> *rates;\
   int rate_max;\
   int n_max;\
   int match;\
   int nsample_train;\
   int epoch;\
   int fixed_seed;\
   std::string *sqrtgrp;\
   std::string *test_option;\
};" );

def make_eff_vs_nsample(tree, n, sqrtgrp, test_option, fixed_seed):
    gra = TGraphErrors(tree.GetEntries(f'sqrtgrp=="{sqrtgrp}" && test_option=="{test_option}" && fixed_seed=={fixed_seed}'))
    gra.SetName(f'gra_test_sqrtgrp{sqrtgrp}_test_option_{test_option}_fixed_seed{fixed_seed}')

    gra_mult = TGraphErrors(tree.GetEntries(f'sqrtgrp=="{sqrtgrp}" && test_option=="{test_option}" && fixed_seed=={fixed_seed}'))
    gra_mult.SetName(f'gra_test_mult_sqrtgrp{sqrtgrp}_test_option_{test_option}_fixed_seed{fixed_seed}')

    x_list = []
    y_list = []
    ey_list = []
    y_mult_list = []
    ey_mult_list = []
    for i, entry in enumerate(tree):
        if entry.sqrtgrp.data() == sqrtgrp and entry.test_option.data() == test_option and entry.fixed_seed==fixed_seed:
            x_list.append(entry.nsample_train)
            y_list.append(entry.eff_test)
            ey_list.append(math.sqrt(entry.eff_test*(1-entry.eff_test)/n))
            y_mult_list.append(entry.eff_test_mult)
            ey_mult_list.append(math.sqrt(entry.eff_test_mult*(1-entry.eff_test_mult)/n))

    # reorder x_list in ascending order for plotting with option "L"
    x_list, y_list, ey_list, y_mult_list, ey_mult_list = sort_lists_by_key(x_list, y_list, ey_list, y_mult_list, ey_mult_list)
    for i, (x, y, ey, ym, eym) in enumerate(zip(x_list, y_list, ey_list, y_mult_list, ey_mult_list)):
        gra.SetPoint(i, x, y)
        gra.SetPointError(i, 0, ey)

        gra_mult.SetPoint(i, x, ym)
        gra_mult.SetPointError(i, 0, eym)

    if gra.GetN() == 0:
        return None, None

    return gra, gra_mult

# make efficiency tree
def make_eff_tree(nsamples_train, mean_eff_test, mean_eff_test_mult, mean_eff_train, mean_eff_train_mult, epochs, sqrtgrps, test_options, fixed_seed_list):
    eff_data.sqrtgrp = ROOT.string()
    eff_data.test_option = ROOT.string()

    eff_tree = TTree('eff_tree', 'for efficiency related plots')
    eff_tree.Branch('eff_test', addressof(eff_data, 'eff_test'), 'eff_test/F')
    eff_tree.Branch('eff_test_mult', addressof(eff_data, 'eff_test_mult'), 'eff_test_mult/F')
    eff_tree.Branch('eff_train', addressof(eff_data, 'eff_train'), 'eff_train/F')
    eff_tree.Branch('eff_train_mult', addressof(eff_data, 'eff_train_mult'), 'eff_train_mult/F') 
    eff_tree.Branch('nsample_train', addressof(eff_data, 'nsample_train'), 'nsample_train/I') 
    eff_tree.Branch('epoch', addressof(eff_data, 'epoch'), 'epoch/I')
    eff_tree.Branch('fixed_seed', addressof(eff_data, 'fixed_seed'), 'fixed_seed/I')
    eff_tree.Branch('sqrtgrp', eff_data.sqrtgrp)
    eff_tree.Branch('test_option', eff_data.test_option)

    for eff_test, eff_test_mult, eff_train, eff_train_mult, nsample_train, epoch, sqrtgrp, test_option, fixed_seed in zip(
            mean_eff_test, mean_eff_test_mult, mean_eff_train, mean_eff_train_mult,
            nsamples_train, epochs, sqrtgrps, test_options, fixed_seed_list):

        eff_data.eff_test = eff_test 
        eff_data.eff_test_mult = eff_test_mult 
        eff_data.eff_train = eff_train 
        eff_data.eff_train_mult = eff_train_mult 
        eff_data.nsample_train = int(nsample_train) 
        eff_data.epoch = int(epoch)
        eff_data.fixed_seed = int(fixed_seed)
        eff_data.sqrtgrp.assign(sqrtgrp)
        eff_data.test_option.assign(test_option)

        eff_tree.Fill()

    return eff_tree

# make detail tree
def make_detail_tree(sidx_list, label_list, rates_list, predict_list, n_max_list, match_list, rate_max_list, nsample_train_list, epoch_list, sqrtgrp_list, test_options, fixed_seed_list):
    detail_data.sqrtgrp = ROOT.string()
    detail_data.test_option = ROOT.string()
    detail_data.predict = ROOT.vector[int]()

    detail_tree = TTree('detail_tree', 'label, prediction, etc')
    detail_tree.Branch('sidx', addressof(detail_data, 'sidx'), 'sidx/I')
    detail_tree.Branch('rank', addressof(detail_data, 'rank'), 'rank/I')
    detail_tree.Branch('label', addressof(detail_data, 'label'), 'label/I')
    detail_tree.Branch('rates', detail_data.rates)
    detail_tree.Branch('predict', detail_data.predict)
    detail_tree.Branch('n_max', addressof(detail_data, 'n_max'), 'n_max/I')
    detail_tree.Branch('match', addressof(detail_data, 'match'), 'match/I')
    detail_tree.Branch('rate_max', addressof(detail_data, 'rate_max'), 'rate_max/I')
    detail_tree.Branch('nsample_train', addressof(detail_data, 'nsample_train'), 'nsample_train/I') 
    detail_tree.Branch('epoch', addressof(detail_data, 'epoch'), 'epoch/I')
    detail_tree.Branch('fixed_seed', addressof(detail_data, 'fixed_seed'), 'fixed_seed/I')
    detail_tree.Branch('sqrtgrp', detail_data.sqrtgrp)
    detail_tree.Branch('test_option', detail_data.test_option)

    for sidx, label, rates, predict, n_max, match, rate_max, nsample_train, epoch, sqrtgrp, test_option, fixed_seed in zip(
            sidx_list, label_list, rates_list, predict_list, n_max_list, match_list, rate_max_list, 
            nsample_train_list, epoch_list, sqrtgrp_list, test_options, fixed_seed_list):
        detail_data.sidx = int(sidx)
        detail_data.label = int(label)
        detail_data.rates = ROOT.vector[int](rates)
        detail_data.predict = ROOT.vector[int](predict)
        detail_data.n_max = int(n_max)
        detail_data.match = int(match)
        detail_data.rate_max = int(rate_max)
        detail_data.nsample_train = int(nsample_train)
        detail_data.epoch = int(epoch)
        detail_data.sqrtgrp.assign(sqrtgrp)
        detail_data.test_option.assign(test_option)
        detail_data.fixed_seed = int(fixed_seed)

        detail_tree.Fill()
    
    return detail_tree

# analyze and produce efficiency related plots
def ana_eff(dir_list, output, debug):
    # Load data from JSON file
    test_options = []
    sqrtgrps = []
    epochs = []
    fixed_seed_list = []
    mean_eff_test = []
    mean_eff_test_mult = []
    mean_eff_train = []
    mean_eff_train_mult = []
    nsamples_train = []

    n_test = None
    for root_dir in dir_list:
        result_file = find_file('result.json', os.path.join(root_dir, 'ray_log/train'))
        if result_file is None:
            print(f'!!!!! result.json: {result_file} not exist or empty in {root_dir} !!!!')
            continue

        try:
            n_epoch = int(root_dir[root_dir.index('n_epoch-')+len('n_epoch-'):root_dir.index('_note')])
            test_option = root_dir[root_dir.index('_test_option-')+len('_test_option-'):root_dir.index('-n_epoch')]
        except:
            n_epoch = 1
            test_option = root_dir[root_dir.index('_test_option-')+len('_test_option-'):root_dir.index('_note')]

        sqrtgrp = root_dir[root_dir.index('sqrtgrp')+len('sqrtgrp'):root_dir.index('_idxtrain0')]
        sqrtgrp = sqrtgrp.replace('-', '_') # root doesn't recognize a command with '-'

        if 'fixed_img_seed' in root_dir:
            fixed_seed = 1
        else:
            fixed_seed = 0

        n = int(root_dir[root_dir.index('idxtrain0-')+len('idxtrain0-'):root_dir.index('_idxtest0-')])
        n /= int(root_dir[root_dir.index('num_workers-')+len('num_workers-'):root_dir.index('_test_option')])
        n *= n_epoch

        if n_test is None:
            n_test = int(root_dir[root_dir.index('idxtest0-')+len('idxtest0-'):root_dir.index('_num_workers')])

        with open(result_file, "r") as f:
            for line in f:
                try:
                    nsamples_train.append(n)
                    sqrtgrps.append(sqrtgrp)
                    fixed_seed_list.append(fixed_seed)
                    test_options.append(test_option)
                    data = json.loads(line)
                    epochs.append(data["epoch"])
                    if "mean_eff_test" in data:
                        mean_eff_test.append(data["mean_eff_test"])
                    if "mean_eff_test_mult_match" in data:
                        mean_eff_test_mult.append(data["mean_eff_test_mult_match"])
                    if "mean_eff_train" in data:
                        mean_eff_train.append(data["mean_eff_train"])
                    if "mean_eff_train_mult_match" in data:
                        mean_eff_train_mult.append(data["mean_eff_train_mult_match"])
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")
    
    dir_eff = output.mkdir("eff")
    dir_eff.cd()
    eff_tree = make_eff_tree(nsamples_train, mean_eff_test, mean_eff_test_mult, mean_eff_train, mean_eff_train_mult, epochs, sqrtgrps, test_options, fixed_seed_list)

    gra_list = []
    gra_mult_list = []
    for test_option in set(test_options):
        for sqrtgrp in set(sqrtgrps):
            for fixed_seed in [0, 1]:
                gra, gra_mult = make_eff_vs_nsample(eff_tree, n_test, sqrtgrp, test_option, fixed_seed)
                if gra is None: # doesn't exist
                    continue
                gra_list.append(gra)
                gra_mult_list.append(gra_mult)

    eff_tree.Write()
    for gra, gra_mult in zip(gra_list, gra_mult_list):
        gra.Write()
        gra_mult.Write()

# analyze and produce correlation plots
def ana_correlation(dir_list, output, debug):
    # define the function process_nt
    def process_nt(previous_sidx, previous_label, previous_n_max, previous_match, previous_rank, sqrtgrp, epoch, n, test_option, fixed_seed, debug):
        rate_max = max(previous_rates)
        max_indices = [i for i, x in enumerate(previous_rates) if x == rate_max]

        test_options.append(test_option)
        sqrtgrp_list.append(sqrtgrp)
        fixed_seed_list.append(fixed_seed)
        epoch_list.append(epoch)
        nsample_train_list.append(n)
        label_list.append(previous_label)
        n_max_list.append(previous_n_max)
        match_list.append(previous_match)
        rank_list.append(previous_rank)
        sidx_list.append(previous_sidx)
        rate_max_list.append(rate_max)
        predict_list.append(ROOT.vector[int](max_indices))
        rates_list.append(ROOT.vector[int](previous_rates))
        #
        previous_rates.clear()
        if debug:
            print('----------------------------------')
            print(f'sidx_list: {sidx_list}')
            print(f'label_list: {label_list}')
            print(f'predict_list: {predict_list}')
            print(f'rates_list: {rates_list}')
            print(f'rate_max_list: {rate_max_list}')
            print(f'match_list: {match_list}')
            print(f'n_max_list: {n_max_list}')
            print(f'rank_list: {rank_list}')

    #
    sidx_list = []
    rank_list = []
    label_list = []
    fixed_seed_list = []
    predict_list = []
    rates_list = []
    n_max_list = []
    match_list = []
    rate_max_list = []
    nsample_train_list = []
    epoch_list = []
    sqrtgrp_list = []
    test_options = []
    for root_dir in dir_list:
        try:
            n_epoch = int(root_dir[root_dir.index('n_epoch-')+len('n_epoch-'):root_dir.index('_note')])
            test_option = root_dir[root_dir.index('_test_option-')+len('_test_option-'):root_dir.index('-n_epoch')]
        except:
            n_epoch = 1
            test_option = root_dir[root_dir.index('_test_option-')+len('_test_option-'):root_dir.index('_note')]

        sqrtgrp = root_dir[root_dir.index('sqrtgrp')+len('sqrtgrp'):root_dir.index('_idxtrain0')]
        sqrtgrp = sqrtgrp.replace('-', '_') # root doesn't recognize a command with '-'

        if 'fixed_img_seed' in root_dir:
            fixed_seed = 1
        else:
            fixed_seed = 0

        n = int(root_dir[root_dir.index('idxtrain0-')+len('idxtrain0-'):root_dir.index('_idxtest0-')])
        n /= int(root_dir[root_dir.index('num_workers-')+len('num_workers-'):root_dir.index('_test_option')])
        n *= n_epoch

        # find the test root file and get the ntuple
        out_test_dict = find_files_with_partial_names(root_dir, 'eff_test_')
        if len(out_test_dict) == 0:
            print(f'---- out_test_dict: {out_test_dict} is empty in {root_dir} ----')
            continue

        for key, values in out_test_dict.items():
            out_test_file  = os.path.join(key, values[0])
            epoch = out_test_file[out_test_file.index('epoch_')+len('epoch_'):out_test_file.index('/seg')]
            f_eff_test = TFile(out_test_file)
            nt = f_eff_test.Get('nt')
            if not nt:  # nt does not exist
                print(f'---- result.root empty in {out_test_file} ----')
                continue;

            previous_sidx  = -1
            previous_label = -1
            previous_n_max = -1
            previous_match = -1
            previous_rank = -1
            previous_rates = []
            for entry in nt:
                if entry.sidx != previous_sidx and previous_sidx != -1:
                    process_nt(previous_sidx, previous_label, previous_n_max, previous_match, previous_rank, sqrtgrp, epoch, n, test_option, fixed_seed, debug)

                previous_rates.append(int(entry.rate))
                previous_sidx = entry.sidx
                previous_label = entry.label
                previous_n_max = entry.n_max
                previous_match = entry.match
                previous_rank = entry.rank

            # for the last sample
            process_nt(previous_sidx, previous_label, previous_n_max, previous_match, previous_rank, sqrtgrp, epoch, n, test_option, fixed_seed, debug)

            f_eff_test.Close()

    dir_detail = output.mkdir("detail")
    dir_detail.cd()
    detail_tree = make_detail_tree(sidx_list, label_list, rates_list, predict_list, n_max_list, match_list, rate_max_list, nsample_train_list, epoch_list, sqrtgrp_list, test_options, fixed_seed_list)
    detail_tree.Write()

# make prediction and identified for each label
def make_rate_vs_group_ID(root_file, test_option, nsample_train, ymax, match, sqrtgrp, fixed_seed):
    f = TFile(root_file)
    detail_tree = f.Get("detail/detail_tree")
    
    n_label = 10;
    shift = 0.5
    pad_offset = 0.03 # leaves space for title
    h_pred = [TH1F(f"h_pred_{i}", "", n_label, -shift, n_label-shift) for i in range(n_label)]
    h_pred_clone = [TH1F(f"h_pred_{i}_clone", "", n_label, -shift, n_label-shift) for i in range(n_label)]
    for entry in detail_tree:
        for i in range(n_label):
            if entry.label==i and entry.match==match and entry.test_option==test_option and entry.nsample_train==nsample_train and entry.sqrtgrp==sqrtgrp and entry.fixed_seed==fixed_seed:
                for m in range(entry.rates.size()):
                    h_pred[i].Fill(m-shift, entry.rates[m])
    
    for i in range(n_label):
        scale = 1.0/h_pred[i].Integral() if h_pred[i].Integral()>0 else 1
        h_pred[i].Scale(scale)
    
    gStyle.SetOptStat(0)
    for i in range(n_label):
        h_pred[i].SetMaximum(ymax)
        h_pred[i].SetMinimum(0)
        h_pred[i].SetFillStyle(0)
        h_pred[i].GetYaxis().SetNdivisions(5)
        h_pred[i].GetYaxis().SetMaxDigits(1)
        h_pred[i].GetYaxis().SetTitleSize(0.1)
        h_pred[i].GetYaxis().SetTitle("spike counts (arb. unit)")
        h_pred[i].GetYaxis().CenterTitle()
        h_pred[i].GetYaxis().SetLabelSize(0.08)
        h_pred[i].GetXaxis().SetLabelSize(0.08)
        h_pred[i].GetXaxis().SetLabelColor(1)
        h_pred[i].GetXaxis().SetLabelOffset(-0.01)
        h_pred[i].GetXaxis().SetTickLength(0)
        h_pred[i].SetLineWidth(1)
        h_pred[i].SetLineColor(14)
        h_pred[i].SetFillColor(14)
        h_pred[i].Draw("barhist")
    
        h_pred_clone[i].SetBinContent(i+1, h_pred[i].GetBinContent(i+1))
        h_pred_clone[i].SetFillColor(9)
        h_pred_clone[i].Draw("barhistsame")
    
    #
    c_pred = TCanvas("c_pred", "", 1500, 1000)
    c_pred.Divide(1, 2)
    pads = []
    pad_width = 1.9*1.0 / n_label
    for i in range(n_label):
        if i<n_label/2:
            c_pred.cd(1)
            x1 = i * pad_width + pad_offset
            x2 = (i + 1) * pad_width + pad_offset
        else:
            x1 = (i - n_label/2) * pad_width + pad_offset
            x2 = (i - n_label/2 + 1) * pad_width + pad_offset
            c_pred.cd(2)
    
        if x2 > 1.0:
            x2 = 1.0
    
        pads.append(TPad("pad%d" % i, "pad%d" % i, x1, 0.05, x2, 1))
        pads[i].SetLeftMargin(0.0)
        pads[i].SetRightMargin(0.0)
        if i == 0 or i==n_label/2:
            pads[i].SetLeftMargin(0.15)
        elif i==n_label-1 or i==n_label/2-1:
            pads[i].SetRightMargin(0.15)
        pads[i].Draw()
        text = TText()
        text.SetTextAlign(23)  # center alignment
        text.SetTextSize(0.05)  # set text size
        text.SetTextFont(42)  # set text font
        text.SetTextAngle(90)  # set text font
        text.DrawTextNDC(0.002, 0.5, "Spike Counts (arb. unit)")  # x, y, text
    #
    legend = []
    for i in range(n_label):
        if i == 0:
            legend.append(TLegend(0.25,0.8,0.85,0.85))
        elif i==n_label-1:
            legend.append(TLegend(0.1,0.8,0.7,0.85))
        else:
            legend.append(TLegend(0.2,0.8,0.8,0.85))
    
        legend[i].AddEntry(h_pred[i], f"Label: {i}","");
        legend[i].SetTextSize(0.1)
        legend[i].SetBorderSize(0)
    
    for i in range(n_label):
        pads[i].cd()
        h_pred[i].Draw("barhist")
        h_pred_clone[i].Draw("barhistsame")
        legend[i].Draw("same");
    
    #
    xlabel = [TLatex(0.5, 0.026, "Neuron group ID")]*2
    for i in range(2):
        c_pred.cd(i+1)
        xlabel[i].SetTextAlign(22);
        xlabel[i].SetTextSize(0.05);
        xlabel[i].SetTextFont(42)
        xlabel[i].Draw();
    
        c_pred.SaveAs(f"rate-{test_option}-nsample_train{nsample_train}_match{match}_sqrtgrp{sqrtgrp}_fixed_seed{fixed_seed}.pdf")

# make efficiency plots 
def make_eff_vs_nsample_plot(root_file):
    f = TFile(root_file)
    dir = f.Get("eff")
    
    # Get a list of all objects in the directory
    keys = dir.GetListOfKeys()
    
    # Iterate over the list to find TGraph objects
    gra_list = []
    for key in keys:
        obj = key.ReadObj()
        if isinstance(obj, TGraph):
            gra_list.append(obj)
    
    # test_option_None
    N_hidden_neuron = ['10', '40', '90', '250', '250(I)', '250(II)']
    
    eff_list = []
    eff_mult_list = []
    for val in ['sqrtgrp1', 'sqrtgrp2', 'sqrtgrp3', 'sqrtgrp5']:
        eff_list.append(next((obj for obj in gra_list if val in obj.GetName() and 'mult' not in obj.GetName() and 'test_option_None' in obj.GetName()), None))
        eff_mult_list.append(next((obj for obj in gra_list if val in obj.GetName() and 'mult' in obj.GetName() and 'test_option_None' in obj.GetName()), None))
    
    # test option_add_more_neuron_div
    for fixed_seed in ['fixed_seed0']:
        for sqrtgrp in ['sqrtgrp1_test', 'sqrtgrp1_1_test']:
            eff_list.append(next((obj for obj in gra_list if 'test_option_add_more_neuron_div' in obj.GetName() and 'mult' not in obj.GetName() and fixed_seed in obj.GetName() and sqrtgrp in obj.GetName()), None))
            eff_mult_list.append(next((obj for obj in gra_list if 'test_option_add_more_neuron_div' in obj.GetName() and 'mult' in obj.GetName() and fixed_seed in obj.GetName() and sqrtgrp in obj.GetName()), None))
    
    c_eff_vs_sample = TCanvas("c_eff_vs_sample", "", 1200, 800)
    pad1 = TPad('pad1', '', 0, 0, 0.5, 1)
    pad2 = TPad('pad2', '', 0.5, 0, 1.0, 1)
    pad1.SetGridy(1)
    pad1.SetLogx(1)
    pad1.SetLeftMargin(0.15)
    pad1.SetRightMargin(0.05)
    pad2.SetGridy(1)
    pad2.SetLogx(1)
    pad2.SetLogy(1)
    pad1.Draw()
    pad2.Draw()
    color = [8, 9, 6, 4, 1, 2]
    eff_marker_style = [24, 25, 26, 28, 32, 30]
    eff_mult_marker_style = [24, 25, 26, 28, 32, 30]
    #eff_mult_marker_style = [20, 21, 22, 34, 29, 23]
    h1=TH1F("h1", "", 100, 6, 1e5)
    h1.SetMaximum(100)
    h1.SetMinimum(0)
    h1.GetYaxis().CenterTitle()
    h1.GetXaxis().CenterTitle()
    h1.GetXaxis().SetTitleOffset(1.2)
    h1.GetYaxis().SetTitleOffset(1.5)
    h1.GetXaxis().SetTitle("Number of Training Samples per Worker")
    h1.GetYaxis().SetTitle("Test Accuracy (%) - Overall")
    
    h2=TH1F("h2", "", 100, 6, 1e5)
    h2.SetMaximum(50)
    h2.SetMinimum(1e-1)
    h2.GetYaxis().CenterTitle()
    h2.GetXaxis().CenterTitle()
    h2.GetXaxis().SetTitleOffset(1.2)
    h2.GetYaxis().SetTitleOffset(1.2)
    h2.GetXaxis().SetTitle("Number of Training Samples per Worker")
    h2.GetYaxis().SetTitle("Test Accuracy (%) - Ambiguous")
    
    gStyle.SetOptStat(0)
    pad1.cd()
    h1.Draw()
    pad2.cd()
    h2.Draw()
    
    legend = TLegend(0.55,0.20,0.90,0.40)
    legend.SetNColumns(2)
    legend.SetTextAlign(22)
    legend.SetTextSize(0.035)
    legend.SetBorderSize(1)
    legend.SetHeader("Number of neurons")
    for i, (eff, eff_mult) in enumerate(zip(eff_list, eff_mult_list)):
        print(i, eff.GetName(), eff_mult.GetName())
        eff.SetMarkerColor(color[i])
        eff.SetLineColor(color[i])
        eff.SetMarkerStyle(eff_marker_style[i])
        eff.SetMarkerSize(1.5)
        eff.SetLineWidth(1)
    
        eff_mult.SetMarkerColor(color[i])
        eff_mult.SetLineColor(color[i])
        eff_mult.SetMarkerStyle(eff_mult_marker_style[i])
        eff_mult.SetMarkerSize(1.5)
        eff_mult.SetLineWidth(1)
    
        # convert it to percentage
        for m in range(eff.GetN()):
            eff.SetPoint(m, eff.GetPointX(m), eff.GetPointY(m)*100)
            eff.SetPointError(m, eff.GetErrorX(m), eff.GetErrorY(m)*100)
            eff_mult.SetPoint(m, eff_mult.GetPointX(m), eff_mult.GetPointY(m)*100)
            eff_mult.SetPointError(m, eff_mult.GetErrorX(m), eff_mult.GetErrorY(m)*100)
    
        pad1.cd()
        eff.Draw("LP")
    
        pad2.cd()
        eff_mult.Draw("LP")
    
        legend.AddEntry(eff, f"{N_hidden_neuron[i]}","p")
    
    pad1.cd()
    legend.Draw("same")
    c_eff_vs_sample.SaveAs("eff_vs_sample.pdf")

#
def make_eff_vs_label_plot(root_file, nsample_train):
    f = TFile(root_file)
    detail_tree = f.Get("detail/detail_tree")
    n_label = 10;
    shift = 0.5
    pad_offset = 0.03 # leaves space for title
    fixed_seed = 0
    test_option_with_sqrtgrp = {
            "None":['1', '2', '3', '5'], 
            "add_more_neuron_div": ['1', '1_1']} 
    color = {
            "None": [8, 9, 6, 4], 
            "add_more_neuron_div": [1, 2]}
    marker_style = {
            "None":[24, 25, 26, 28], 
            "add_more_neuron_div": [32, 30]}
    N_hidden_neuron = {
            "None":['10', '40', '90', '250'], 
            "add_more_neuron_div": ['250(I)', '250(II)']}
    
    h_eff = {}
    h_eff_mult = {}
    h_eff_denom = {}
    gra_eff = {}
    gra_eff_mult = {}
    for test_option, sqrtgrp_list in test_option_with_sqrtgrp.items():
        h_eff[f'{test_option}'] = [TH1F(f"h_eff_{test_option}_sqrtgrp{i}", "", n_label, -shift, n_label-shift) for i in range(len(sqrtgrp_list))]
        h_eff_mult[f'{test_option}'] = [TH1F(f"h_eff_mult{test_option}_sqrtgrp{i}", "", n_label, -shift, n_label-shift) for i in range(len(sqrtgrp_list))]
        h_eff_denom[f'{test_option}'] = [TH1F(f"h_eff_denorm_{test_option}_sqrtgrp{i}", "", n_label, -shift, n_label-shift) for i in range(len(sqrtgrp_list))]
    
        gra_eff[f'{test_option}'] = [None for i in range(len(sqrtgrp_list))]
        gra_eff_mult[f'{test_option}'] = [None for i in range(len(sqrtgrp_list))]
    
    for entry in detail_tree:
        if entry.fixed_seed != fixed_seed:
            continue

        if entry.nsample_train==nsample_train: 
            try:
                idx = test_option_with_sqrtgrp[entry.test_option].index(entry.sqrtgrp)
                h_eff_denom[entry.test_option][idx].Fill(entry.label-shift)
                if entry.match==1:
                    h_eff[entry.test_option][idx].Fill(entry.label-shift)
                    if entry.n_max>1:
                        h_eff_mult[entry.test_option][idx].Fill(entry.label-shift)
            except:
                continue
    
    for test_option, sqrtgrp_list in test_option_with_sqrtgrp.items():
        for i in range(len(sqrtgrp_list)):
            if h_eff[f'{test_option}'][i].GetEntries() == 0:
                continue
            h_eff[f'{test_option}'][i].Divide(h_eff_denom[f'{test_option}'][i])
            h_eff_mult[f'{test_option}'][i].Divide(h_eff_denom[f'{test_option}'][i])
    
            h_eff[f'{test_option}'][i].SetMarkerColor(color[f'{test_option}'][i])
            h_eff_mult[f'{test_option}'][i].SetMarkerColor(color[f'{test_option}'][i])
            h_eff[f'{test_option}'][i].SetMarkerStyle(marker_style[f'{test_option}'][i])
            h_eff_mult[f'{test_option}'][i].SetMarkerStyle(marker_style[f'{test_option}'][i])
    
            # calculate errors
            for ib in range(h_eff[f'{test_option}'][i].GetNbinsX()):
                eff = h_eff[f'{test_option}'][i].GetBinContent(ib+1)
                n = h_eff_denom[f'{test_option}'][i].GetBinContent(ib+1)
                err = math.sqrt(eff*(1-eff)/n)
                h_eff[f'{test_option}'][i].SetBinError(ib+1, err)
    
                eff_mult = h_eff_mult[f'{test_option}'][i].GetBinContent(ib+1)
                err_mult = math.sqrt(eff_mult*(1-eff_mult)/n)
                h_eff_mult[f'{test_option}'][i].SetBinError(ib+1, err_mult)
    
    c_eff_vs_label = TCanvas("c_eff_vs_label", "", 1200, 800)
    pad1 = TPad('pad1', '', 0, 0, 0.5, 1)
    pad2 = TPad('pad2', '', 0.5, 0, 1.0, 1)
    pad1.SetGridy(1)
    pad1.SetLeftMargin(0.15)
    pad1.SetRightMargin(0.05)
    pad2.SetGridy(1)
    pad2.SetLogy(1)
    pad1.Draw()
    pad2.Draw()
    
    h1=TH1F("h1", "", n_label, -shift, n_label-shift)
    h1.SetMaximum(100)
    h1.SetMinimum(0)
    h1.GetYaxis().CenterTitle()
    h1.GetXaxis().CenterTitle()
    h1.GetXaxis().SetTitleOffset(1.2)
    h1.GetYaxis().SetTitleOffset(1.5)
    h1.GetXaxis().SetTitle("Input Label")
    h1.GetYaxis().SetTitle("Test Accuracy (%) - Overall")
    
    h2=TH1F("h2", "", n_label, -shift, n_label-shift)
    h2.SetMaximum(50)
    h2.SetMinimum(1e-2)
    h2.GetYaxis().CenterTitle()
    h2.GetXaxis().CenterTitle()
    h2.GetXaxis().SetTitleOffset(1.2)
    h2.GetYaxis().SetTitleOffset(1.2)
    h2.GetXaxis().SetTitle("Input Label")
    h2.GetYaxis().SetTitle("Test Accuracy (%) - Ambiguous")
    
    gStyle.SetOptStat(0)
    pad1.cd()
    h1.Draw()
    pad2.cd()
    h2.Draw()
    
    legend = TLegend(0.5,0.30,0.85,0.50)
    legend.SetNColumns(2)
    legend.SetTextAlign(22)
    legend.SetTextSize(0.035)
    legend.SetBorderSize(1)
    legend.SetHeader("Number of neurons")
    for test_option, sqrtgrp_list in test_option_with_sqrtgrp.items():
        for i in range(len(sqrtgrp_list)):
            h_eff[f'{test_option}'][i].Scale(1e2)
            h_eff_mult[f'{test_option}'][i].Scale(1e2)
    
            pad1.cd()
            gra_eff[f'{test_option}'][i] = TGraphErrors(h_eff[f'{test_option}'][i])
            for m in range(gra_eff[f'{test_option}'][i].GetN()):
                gra_eff[f'{test_option}'][i].SetPointError(m, 0, gra_eff[f'{test_option}'][i].GetErrorY(m))
            gra_eff[f'{test_option}'][i].SetMarkerSize(1.5)
            gra_eff[f'{test_option}'][i].SetLineColor(color[f'{test_option}'][i])
            gra_eff[f'{test_option}'][i].Draw("LP")
            pad2.cd()
            gra_eff_mult[f'{test_option}'][i] = TGraphErrors(h_eff_mult[f'{test_option}'][i])
            for m in range(gra_eff_mult[f'{test_option}'][i].GetN()):
                gra_eff_mult[f'{test_option}'][i].SetPointError(m, 0, gra_eff_mult[f'{test_option}'][i].GetErrorY(m))
            gra_eff_mult[f'{test_option}'][i].SetMarkerSize(1.5)
            gra_eff_mult[f'{test_option}'][i].SetLineColor(color[f'{test_option}'][i])
            gra_eff_mult[f'{test_option}'][i].Draw("LP")
    
            legend.AddEntry(gra_eff[f'{test_option}'][i], f"{N_hidden_neuron[test_option][i]}","p")
    
    pad1.cd()
    legend.Draw("same")
    c_eff_vs_label.SaveAs(f"eff_vs_label_nsample_train{nsample_train}.pdf")

#
def make_predict_vs_label(root_file, test_option, nsample_train, ymax, match, sqrtgrp, fixed_seed):
    f = TFile(root_file)
    detail_tree = f.Get("detail/detail_tree")
    
    n_label = 10;
    shift = 0.5
    pad_offset = 0.03 # leaves space for title
    h_pred = [TH1F(f"h_pred_{i}", "", n_label, -shift, n_label-shift) for i in range(n_label)]
    h_pred_clone = [TH1F(f"h_pred_{i}_clone", "", n_label, -shift, n_label-shift) for i in range(n_label)]
    for entry in detail_tree:
        for i in range(n_label):
            if entry.label==i and entry.match==match and entry.test_option==test_option and entry.nsample_train==nsample_train and entry.sqrtgrp==sqrtgrp and entry.fixed_seed==fixed_seed:
                for m in range(entry.predict.size()):
                    h_pred[i].Fill(entry.predict[m]-shift)
    
    for i in range(n_label):
        scale = 1.0/h_pred[i].Integral() if h_pred[i].Integral()>0 else 1
        h_pred[i].Scale(scale)
    
    gStyle.SetOptStat(0)
    for i in range(n_label):
        h_pred[i].SetMaximum(ymax)
        h_pred[i].SetMinimum(0)
        h_pred[i].SetFillStyle(0)
        h_pred[i].GetYaxis().SetNdivisions(5)
        h_pred[i].GetYaxis().SetMaxDigits(3)
        h_pred[i].GetYaxis().SetTitleSize(0.1)
        h_pred[i].GetYaxis().CenterTitle()
        h_pred[i].GetYaxis().SetLabelSize(0.08)
        h_pred[i].GetXaxis().SetLabelSize(0.08)
        h_pred[i].GetXaxis().SetLabelColor(1)
        h_pred[i].GetXaxis().SetLabelOffset(-0.01)
        h_pred[i].GetXaxis().SetTickLength(0.0)
        h_pred[i].SetLineWidth(1)
        h_pred[i].SetLineColor(1)
        h_pred[i].SetFillColor(1)
        h_pred[i].Draw("barhist")
    
        h_pred_clone[i].SetBinContent(i+1, h_pred[i].GetBinContent(i+1))
        h_pred_clone[i].SetFillColor(1)
        h_pred_clone[i].Draw("barhistsame")
    
    #
    c_pred = TCanvas("c_pred", "", 1500, 1000)
    c_pred.Divide(1, 2)
    pads = []
    pad_width = 1.9*1.0 / n_label
    for i in range(n_label):
        if i<n_label/2:
            c_pred.cd(1)
            x1 = i * pad_width + pad_offset
            x2 = (i + 1) * pad_width + pad_offset
        else:
            x1 = (i - n_label/2) * pad_width + pad_offset
            x2 = (i - n_label/2 + 1) * pad_width + pad_offset
            c_pred.cd(2)
    
        if x2 > 1.0:
            x2 = 1.0
    
        pads.append(TPad("pad%d" % i, "pad%d" % i, x1, 0.05, x2, 1))
        pads[i].SetLeftMargin(0.0)
        pads[i].SetRightMargin(0.0)
        if i == 0 or i==n_label/2:
            pads[i].SetLeftMargin(0.15)
        elif i==n_label-1 or i==n_label/2-1:
            pads[i].SetRightMargin(0.15)
        pads[i].Draw()
        text = TText()
        text.SetTextAlign(23)  # center alignment
        text.SetTextSize(0.05)  # set text size
        text.SetTextFont(42)  # set text font
        text.SetTextAngle(90)  # set text font
        text.DrawTextNDC(0.002, 0.5, "Counts (arb. unit)")  # x, y, text
    #
    legend = []
    for i in range(n_label):
        if i == 0:
            legend.append(TLegend(0.25,0.8,0.85,0.85))
        elif i==n_label-1:
            legend.append(TLegend(0.1,0.8,0.7,0.85))
        elif i==(n_label/2-1):
            legend.append(TLegend(0.1,0.8,0.7,0.85))
        else:
            legend.append(TLegend(0.2,0.8,0.8,0.85))
    
        legend[i].AddEntry(h_pred[i], f"Label: {i}","");
        legend[i].SetTextSize(0.1)
        legend[i].SetBorderSize(0)
    
    for i in range(n_label):
        pads[i].cd()
        h_pred[i].Draw("barhist")
        h_pred_clone[i].Draw("barhistsame")
        legend[i].Draw("same");
    
    #
    xlabel = [TLatex(0.5, 0.026, "Incorrect Predictions")]*2
    for i in range(2):
        c_pred.cd(i+1)
        xlabel[i].SetTextAlign(22);
        xlabel[i].SetTextSize(0.05);
        xlabel[i].SetTextFont(42)
        xlabel[i].Draw();
    
        c_pred.SaveAs(f"pred-{test_option}-nsample_train{nsample_train}_match{match}_sqrtgrp{sqrtgrp}_fixed_seed{fixed_seed}.pdf")

##
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='basic input')
    parser.add_argument('--option', required=True, type=str, help='make_tree, make_plot')
    parser.add_argument('--dir', required=False, type=str, default='', help='base dir of ray.train output')
    parser.add_argument('--debug', required=False, default=False, type=lambda x: bool(strtobool(x)), help='options:  True or False')
    args = parser.parse_args()

    if args.option == 'make_tree':
        dir_list = find_directory(args.dir, 'ray.train')

        eff_data = ROOT.eff_data_t()
        detail_data = ROOT.detail_data_t()

        output = TFile("result.root", "RECREATE");
        ana_eff(dir_list, output, args.debug)
        ana_correlation(dir_list, output, args.debug)
    elif args.option == 'make_plot':
        test_option_list = ['add_more_neuron_div']
        nsample_train_list = [10, 30000]
        ymax_list = [0.12, 0.2]
        sqrtgrp_list = ['1', '1_1']
        for test_option in test_option_list:
            for nsample_train, ymax in zip(nsample_train_list, ymax_list):
                for sqrtgrp in sqrtgrp_list:
                    print(test_option, nsample_train, ymax, sqrtgrp)
                    make_rate_vs_group_ID('result.root', test_option, nsample_train, ymax, match=1, sqrtgrp=sqrtgrp, fixed_seed=0)
                    make_predict_vs_label('result.root', test_option, nsample_train, 0.7, match=0, sqrtgrp=sqrtgrp, fixed_seed=0)
        make_eff_vs_nsample_plot('result.root')
        make_eff_vs_label_plot('result.root', nsample_train=30000)
