void plot_trial_result(string result_file, int n_hlayer, int nx, int ny)
{
    vector<TCanvas*> cc(n_hlayer);
    vector<vector<string>> var_draw(n_hlayer);

    for(int ih = 0; ih<n_hlayer; ih++) {
        string var_draw_layer[] = {
            "max_delay_input2e"+to_string(ih), 
            "max_delay_efe"+to_string(ih), 
            "norm_scale_S_input2e"+to_string(ih), 
            "gmax_input2e"+to_string(ih),
            "penalty_input2e"+to_string(ih),
            "dW_e2e"+to_string(ih),
            "delta_vt"+to_string(ih),
            "tau_adpt"+to_string(ih),
            "tau_ge"+to_string(ih),
            "tau_gi"+to_string(ih),
            "tau_membrane_exci"+to_string(ih),
            "w_sat_scale" , 
            "vt_sat_scale", 
            "w_sat_shift" , 
            "vt_sat_shift",  
            "nu_pre_ee"  
        };

        int size_of_var_draw_layer = sizeof(var_draw_layer)/sizeof(var_draw_layer[0]);
        var_draw[ih].insert(var_draw[ih].end(), var_draw_layer, var_draw_layer+size_of_var_draw_layer);

        string cc_name = "cc"+to_string(ih);
        cc[ih] = new TCanvas(cc_name.c_str(), "", 1000, 800);
        cc[ih]->Divide(nx, ny);

        TFile* f = new TFile(result_file.c_str());
        TNtuple* nt = (TNtuple*)f->Get("nt");
        nt->SetMarkerStyle(24);
        nt->SetMarkerSize(0.1);

        for(int i = 0; i<size_of_var_draw_layer; i++) {
            cc[ih]->cd(i+1);
            string content = var_draw[ih][i]+":eff_valid-eff_valid_mult";
            nt->Draw(content.c_str(), "", "colz");
        }
        string image_name = "layer_"+to_string(ih)+".pdf";
        cc[ih]->SaveAs(image_name.c_str());
    }
}
