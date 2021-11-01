import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from uproot3_methods.classes.TH1 import Methods as TH1Methods


class TH1(TH1Methods, list):
    pass


class TAxis(object):
    def __init__(self, fNbins, fXmin, fXmax):
        self._fNbins = fNbins
        self._fXmin = fXmin
        self._fXmax = fXmax


def yahist_to_root(hist, label, name, overflow='all'):

    sumw, sumw2 = hist.counts, hist.errors**2
    edges = hist.edges[1:-1]

    out = TH1.__new__(TH1)
    out._fXaxis = TAxis(len(edges) - 1, edges[0], edges[-1])
    out._fXaxis._fName = name
    out._fXaxis._fTitle = label

    out._fXaxis._fXbins = edges.astype(">f8")

    if overflow=='under' or overflow=='all':
        sumw[1] += sumw[0]
        sumw2[1] += sumw2[0]
    
    if overflow=='over' or overflow=='all':
        sumw[-2] += sumw[-1]
        sumw2[-2] += sumw2[-1]

    centers = (edges[:-1] + edges[1:]) / 2.0
    out._fEntries = out._fTsumw = out._fTsumw2 = sumw[1:-1].sum()
    out._fTsumwx = (sumw[1:-1] * centers).sum()
    out._fTsumwx2 = (sumw[1:-1] * centers**2).sum()

    out._fName = "histogram"
    out._fTitle = label

    out._classname = b"TH1D"
    out.extend(sumw.astype(">f8"))
    out._fSumw2 = sumw2.astype(">f8")

    return out
