import pandas as pd

symmetric_coeffs = [
    'ctlT',
    'cQlM',
    'cte',
    'ctlS',
    'cQe',
    'cQl3',
    'ctl',
]

zero_weight = 'EFTrwgt331_ctlTi_0.0_ctq1_0.0_ctq8_0.0_cQq83_0.0_cQq81_0.0_cQlMi_0.0_cbW_0.0_cpQ3_0.0_ctei_0.0_cQei_0.0_ctW_0.0_cpQM_0.0_ctlSi_0.0_ctZ_0.0_cQl3i_0.0_ctG_0.0_cQq13_0.0_cQq11_0.0_cptb_0.0_ctli_0.0_ctp_0.0_cpt_0.0'

class eft_point:
    def __init__(self,
                 weight_name=zero_weight,
                 divider='_',
                 expand=False,
                 ):
        point = weight_name.replace('_nlo','')  # strips postfix
        point = point.split(divider)[1:]  # removes LHEWeight or EFTrwgt_ from the names
        point_d = {x: float(y) for x,y in zip(point[0::2], point[1::2])}
        point_d_ext = {}
        for k in point_d:
            if k.endswith('i') and expand:
                for i in range(3):
                    point_d_ext[k.replace('i', str(i+1))] = point_d[k]
            else:
                point_d_ext[k] = point_d[k]

        self.point = point_d_ext

    def get_coord(self):
        #print (sorted(self.point))
        return tuple([self.point[x] for x in sorted(self.point)])

    def reset(self):
        self.point = {x:0 for x in self.point}

    def set(self, point, reset=False):
        if reset: self.reset()
        for x in point:
            self.point[x] = point[x]

    def show(self, precision=2):
        df = pd.DataFrame([self.point])
        print (df.round(precision))

    @classmethod
    def from_customize_card(cls, data_card, identifier='set param_card c', replacer='', compress=True):
        with open(data_card, 'r') as r:
            card = r.readlines()
        points = ['dummy']
        for line in card:
            if line.count(identifier):
                tmp = line.replace(replacer,'').replace('\n','')
                tmp = tmp.split(' ')
                if tmp[2][:-1] in symmetric_coeffs and compress:
                    if tmp[2][-1] == '1':
                        points += [tmp[2].replace('1', 'i'), tmp[3]]
                else:
                    points += tmp[2:]
                #points[tmp[2]] = float(tmp[3])
                #.append(line.replace(replacer,'').replace('\n',''))
        dummy_weight = '_'.join(points)
        return cls(dummy_weight)

class eft_setup:
    def __init__(self, data_card, events, maxN=999, weight_name_filter='EFT'):
        self.points = [eft_point(weight) for weight in events.LHEWeight.fields if weight.startswith(weight_name_filter)]
        self.points = self.points[:maxN]
        self.ref_point = eft_point.from_customize_card(data_card)

    def get_coordinates(self):
        return [p.get_coord() for p in self.points]

    def get_ref_point(self):
        return self.ref_point.get_coord()


def get_weight_names_from_card(data_card, identifier='EFTrwgt', replacer='launch --rwgt_name='):
    with open(data_card, 'r') as r:
        card = r.readlines()
    weights = []
    for line in card:
        if line.count(identifier): weights.append(line.replace(replacer,'').replace('\n',''))
    return weights

def get_points(weight, divider='_'):
    points = weight.replace('_nlo','')  # strips postfix
    points = points.split(divider)[1:]  # removes LHEWeight or EFTrwgt_ from the names
    points_d = {x: float(y) for x,y in zip(points[0::2], points[1::2])}
    points_d_ext = {}
    for k in points_d:
        if k.endswith('i'):
            for i in range(3):
                points_d_ext[k.replace('i', str(i+1))] = points_d[k]
        else:
            points_d_ext[k] = points_d[k]

    return points_d_ext
    #vals = [ float(x.replace('p','.')) for x in points[1::2] ]


def get_coordinates(points, divider='_'):
    points = points.replace('_nlo','')  # strips postfix
    points = points.split(divider)[1:]  # removes LHEWeight or EFTrwgt_ from the names
    vals = [ float(x.replace('p','.')) for x in points[1::2] ]
    return tuple(vals)

def get_coefficients(points, divider=' '):
    points = points.replace('_nlo','')  # strips postfix
    points = points.split(divider)[1:]  # removes LHEWeight or EFTrwgt_ from the names
    return tuple(points[0::2])

def get_coordinates_and_ref(f_in, is2D=False):
    import uproot

    tree = uproot.open(f_in)['Events']
    weights = [ x for x in tree.keys() if x.startswith('LHEWeight_c') ]
    #ref_point = [ x for x in tree.keys() if x.startswith('ref_point') ][0]
    #ref_point = ref_point.replace('ref_point_','')
    if is2D:
        ref_point = "cpt_0p_cpQM_0p"
    else:
        ref_point = "ctZ_2p_cpt_4p_cpQM_4p_cpQ3_4p_ctW_2p_ctp_2p"

    coordinates = [get_coordinates(weight) for weight in weights]
    ref_coordinates = [ float(x.replace('p','.')) for x in ref_point.split('_')[1::2] ]
    
    return coordinates, ref_coordinates




if __name__ == '__main__':
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    NanoAODSchema.warn_missing_crossrefs = False

    f_in = '/home/users/dspitzba/TOP/CMSSW_10_6_19/src/ttW_EFT_LO.root'
    n_max = 5000
    events = NanoEventsFactory.from_root(
        f_in,
        schemaclass = NanoAODSchema,
        entry_stop = n_max,
    ).events()

    #test = eft_setup('../production/cards/ttlnuJet_all22WCs/ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0_customizecards.dat', events, maxN=276)
    test = eft_setup('../production/cards/ttlnuJet_all22WCs/ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0_customizecards.dat', events)

    from Tools.awkwardHyperPoly import *
    hp = HyperPoly(2)


    weight_names = events.LHEWeight.fields#[:276]
    weights = [getattr(events.LHEWeight, w) for w in weight_names[2:]]
    w = np.array(weights)

    # coeff = hp.get_parametrization(ak.Array(weights))  # FIXME


    # own implementation
    #


    import itertools
    import numpy as np

    combination  = {}
    counter = 0
    order = 2
    nvar = 22
    for o in range(order+1):
        for comb in itertools.combinations_with_replacement( range(nvar), o ):
            combination[counter] = comb
            counter += 1

    m = 332
    n = int((nvar+1)*(nvar+2)/2)
    A = np.empty( [m, n ] )
    param_points = test.get_coordinates()
    ref_point = test.get_ref_point()
    combinations = list(combination.values())
    for d in range(m):
        for e in range(n):
            A[d][e] = np.prod([param_points[d][x]-ref_point[x] for x in combinations[e]])
            #[np.prod([test_point[d][x]-ref_point[x] for x in combination[c]]) for c in combination]
            #if d > e:
            #    A[d][e] = A[e][d]
            #else:
            #A[d][e] = self.expectation(self.combination[d] + self.combination[e])
            #
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    T = np.dot(Vt.T, np.linalg.inv(np.diag(S)))
    R = np.dot(U, T)
    s = np.dot(R.T, w[:,0])


    # NOTE this actually seems to work, per event. can this
    res = np.linalg.lstsq(A, w[:,0], rcond=None)

    new_point = [0]*n
    x_vec = [ np.prod([new_point[x]-ref_point[x] for x in combinations[e]]) for e in range(n) ]
    new_weight = np.dot(res[0], x_vec)
    print (new_weight)
    # this corresponds to getattr(events.LHEWeight, weight_names[-1])[0]

    test_point = eft_point.from_customize_card('../production/cards/ttlnuJet_all22WCs/ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0_customizecards.dat')
    test_point.set({"cpt":1.0}, reset=True)
    test_point.show()
    x_vec = [ np.prod([test_point.get_coord()[x]-ref_point[x] for x in combinations[e]]) for e in range(n) ]
    new_weight = np.dot(res[0], x_vec)

    print (new_weight)

    ##
    ##
    print ("Sanity check with 2D EFT samples")

    #f_in = '../notebooks/excl_topW.root'
    f_in = '/ceph/cms/store/user/dspitzba/ProjectMetis/TTWToLNu_TtoLep_aTtoHad_5f_EFT_NLO_RunIISummer20_NanoGEN_NANO_v13/output_100.root'
    n_max = 5000
    events = NanoEventsFactory.from_root(
        f_in,
        schemaclass = NanoAODSchema,
        entry_stop = n_max,
    ).events()

    from Tools.awkwardHyperPoly import *
    hp = HyperPoly(2)
    coord, ref = get_coordinates_and_ref( f_in, is2D=True)
    hp.initialize( coord, ref )

    weight_names = events.LHEWeight.fields
    weights = [getattr(events.LHEWeight, w) for w in weight_names[1:]]
    w = np.array(weights)
    coeff = hp.get_parametrization(w)

    print(hp.eval(np.array([coeff[:,0]]), [3,3])[0])
    print(events.LHEWeight.cpt_3p_cpqm_3p_nlo[0])

    combination  = {}
    counter = 0
    order = 2
    nvar = 2
    for o in range(order+1):
        for comb in itertools.combinations_with_replacement( range(nvar), o ):
            combination[counter] = comb
            counter += 1

    m = 6
    n = int((nvar+1)*(nvar+2)/2)
    A = np.empty( [m, n ] )
    param_points = test.get_coordinates()
    ref_point = test.get_ref_point()
    combinations = list(combination.values())
    for d in range(m):
        for e in range(n):
            A[d][e] = np.prod([coord[d][x]-ref[x] for x in combinations[e]])

    res = np.linalg.lstsq(A, w[:,0], rcond=None)
    coord = [3,3]
    x_vec = [ np.prod([coord[x]-ref[x] for x in combinations[e]]) for e in range(n) ]
    new_weight = np.dot(res[0], x_vec)
    print (new_weight)
    # This does indeed close. Yay
