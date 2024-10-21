void plot_details(float rate_e0, string infile)
{
    gStyle->SetOptStat(0);
    TFile *f = new TFile(infile.c_str());
    TNtuple* nt = (TNtuple*)f->Get("nt");
    TCanvas* c1 = new TCanvas("c1", "c1", 1500, 1100);
    c1->Divide(4, 3);
    for(int i=0; i<10; i++){ 
        c1->cd(i+1);
        string condition = "rate_i>0 && rate_e0>"+to_string(rate_e0)+" && grp_e0=="+to_string(i);
        string content = "-x_i:y_i>>h"+to_string(i)+"(56,0, 28, 56, -28, 0)";
        nt->Draw(content.c_str(), condition.c_str(), "colz");
    }
}
