//R__LOAD_LIBRARY($ROOTSYS/test/libEvent.so)

#include "TROOT.h"
#include "Riostream.h"
#include "TFile.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TString.h"
#include "TH1F.h"
//#include "TSeq.h"

void counter_macro(TString file_path="", TString skim_path="") // was void
{
    TString dir = file_path;

    std::unique_ptr<TFile> in_file(TFile::Open(dir));
    TTree *in_runs;
    in_file->GetObject("Runs", in_runs);

    in_runs->SetBranchStatus("*", 1);
    in_runs->GetEntry();

    TFile skim_file(skim_path, "update");

    const auto genEventSumw = in_runs->GetLeaf("genEventSumw")->GetValue();
    int genEventCount = in_runs->GetLeaf("genEventCount")->GetValue();
    int nLHEPdfSumw = in_runs->GetLeaf("nLHEPdfSumw")->GetValue();
    int nLHEScaleSumw = in_runs->GetLeaf("nLHEScaleSumw")->GetValue();

    TH1* h_sumw = new TH1F("genEventSumw", "genEventSumw", 1, 0, 1);
    TH1* h_nevents = new TH1F("genEventCount", "genEventCount", 1, 0, 1);
    TH1* h_pdf = new TH1F("LHEPdfSumw", "LHEPdfSumw", nLHEPdfSumw, 0, 1);
    TH1* h_scale = new TH1F("LHEScaleSumw", "LHEScaleSumw", nLHEScaleSumw, 0, 1);

    h_sumw->SetBinContent(1, genEventSumw);
    h_nevents->SetBinContent(1, genEventCount);

    for(int i=0;i<nLHEPdfSumw;i++){
        h_pdf->SetBinContent(i+1, in_runs->GetLeaf("LHEPdfSumw")->GetValue(i) * genEventSumw);
    }
    for(int i=0;i<nLHEScaleSumw;i++){
        h_scale->SetBinContent(i+1, in_runs->GetLeaf("LHEScaleSumw")->GetValue(i) * genEventSumw);
    }

    skim_file.Write();

    return ;
}
