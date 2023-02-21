

def get_coordinates(points):
    points = points.replace('LHEWeight_','').replace('_nlo','')
    vals = [ float(x.replace('p','.')) for x in points.split('_')[1::2] ]
    return tuple(vals)

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
