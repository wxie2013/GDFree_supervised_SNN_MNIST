void scan(string fname, float eff_min, float epoch_max)
{
    TFile f(fname.c_str());
    string condition = "eff_valid-eff_mult_match_validation >="+to_string(eff_min)+"&& epoch>="+to_string(epoch_max);
    TNtuple * nt = (TNtuple*)f.Get("nt")->Clone("nt");
    string content = "index     : epoch     : eff_valid : eff_mult_match_validation : max_delay : max_dendritic_delay : n_syn     : stdp_type : N_hidden  : sigma_noise : v_thres_exci : v_reversal_e_exci : v_reversal_i_exci : v_rest_exci : v_reset_exci : refrac_time_exci : switch_norm : switch_reinforce : switch_crossover : silence_time : sim_time  : norm_scale_S_input2e0 : gmax_input2e0 : penalty_input2e0 : Ne0       : dW_e2e_same_group0 : dW_e2e_diff_group0 : delta_vt0 : prob_e2e_same_group0 : prob_e2e_diff_group0 : overlap0  : max_delay_e2e_same_group0 : max_delay_e2e_diff_group0 : tau_adpt0 : tau_ge0   : tau_gi0   : tau_membrane_exci0 : nu_pre_ee : nu_post_ee : tc_pre_ee : tc_post_1_ee : tc_post_2_ee";

    nt->Scan(content.c_str(), condition.c_str());
}

