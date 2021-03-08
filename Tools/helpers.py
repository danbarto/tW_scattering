'''
Just a collection of useful functions
Most of these functions need to be updated for awkward1.
'''
import pandas as pd
import numpy as np
import awkward1 as ak

#import yaml
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

#from yaml import Loader, Dumper

import os
import shutil
import math
import copy

import glob

data_path = os.path.expandvars('$TWHOME/data/')

def get_samples():
    with open(data_path+'samples.yaml') as f:
        return load(f, Loader=Loader)

def loadConfig():
    with open(data_path+'config.yaml') as f:
        config = load(f, Loader=Loader)
    return config

def dumpConfig(cfg):
    with open(data_path+'config.yaml', 'w') as f:
        dump(cfg, f, Dumper=Dumper, default_flow_style=False)
    return True

def get_scheduler_address():
    with open(os.path.expandvars('$TWHOME/scheduler_address.txt'), 'r') as f:
        lines = f.readlines()
        scheduler_address = lines[0].replace('\n','')
    return scheduler_address

def getName( DAS ):
    split = DAS.split('/')
    if split[-1].count('AOD'):
        return '_'.join(DAS.split('/')[1:3])
    else:
        return '_'.join(DAS.split('/')[-3:-1])
        #return'dummy'

def finalizePlotDir( path ):
    path = os.path.expandvars(path)
    if not os.path.isdir(path):
        os.makedirs(path)
    shutil.copy( os.path.expandvars( '$TWHOME/Tools/php/index.php' ), path )
    

def doAwkwardLookup(h, ar):
    '''
    takes a ya_hist histogram (which has a lookup function) and an awkward array.
    '''
    return ak.unflatten(
        h.lookup(
            ak.to_numpy(
                ak.flatten(ar)
            ) 
        ), ak.num(ar) )

def getCutFlowTable(output, processes=['tW_scattering', 'TTW', 'ttbar'], lines=['skim', 'twoJet', 'oneBTag'], significantFigures=3, absolute=True, signal=None, total=False):
    '''
    Takes the output of a coffea processor (i.e. a python dictionary) and returns a formated cut-flow table of processes.
    Lines and processes have to follow the naming of the coffea processor output.
    '''
    res = {}
    eff = {}
    for proc in processes:
        res[proc] = {line: "%s +/- %s"%(round(output[proc][line], significantFigures-len(str(int(output[proc][line])))), round(math.sqrt(output[proc][line+'_w2']), significantFigures-len(str(int(output[proc][line]))))) for line in lines}
        
        # for efficiencies. doesn't deal with uncertainties yet
        eff[proc] = {lines[i]: round(output[proc][lines[i]]/output[proc][lines[i-1]], significantFigures) if (i>0 and output[proc][lines[i-1]]>0) else 1. for i,x in enumerate(lines)}
    
    if total:
        res['total'] = {line: "%s"%round( sum([ output[proc][line] for proc in total ] ), significantFigures-len(str(int(sum([ output[proc][line] for proc in total ] ))))) for line in lines }
    
    # if a signal is specified, calculate S/B
    if signal is not None:
        backgrounds = copy.deepcopy(processes)
        backgrounds.remove(signal)
        res['S/B'] = {line: round( output[signal][line]/sum([ output[proc][line] for proc in backgrounds ]) if sum([ output[proc][line] for proc in backgrounds ])>0 else 1, significantFigures) for line in lines }
            
    if not absolute:
        res=eff
    df = pd.DataFrame(res)
    df = df.reindex(lines) # restores the proper order
    return df

## event shape variables. computing intensive
def cosTheta(obj):
    '''
    cos(Omega_{i,j}) -> projected on transverse plane
    '''
    return np.cos(obj.cross(obj).i0.phi - obj.cross(obj).i1.phi)

def cosOmega(obj):
    '''
    theta = 2*arctan(exp(-eta))
    cos(Omega_{i,j}) = cos(theta_i)*cos(theta_j) + sin(theta_i)*sin(theta_j)*cos(phi_i - phi_j)
    '''
    return  np.cos(2*np.arctan(np.exp(-obj.cross(obj).i0.eta))) * np.cos(2*np.arctan(np.exp(-obj.cross(obj).i1.eta))) + \
            np.sin(2*np.arctan(np.exp(-obj.cross(obj).i0.eta))) * np.sin(2*np.arctan(np.exp(-obj.cross(obj).i1.eta))) * \
            np.cos(obj.cross(obj).i0.phi - obj.cross(obj).i1.phi)

def Wij(obj):
    return obj.cross(obj).i0.pt * obj.cross(obj).i1.pt

def FWMT1(obj):
    '''
    First Fox-Wolfram Moment reduced to transverse plane, uses simplified solid angle
    '''
    return (Wij(obj)*cosTheta(obj)).sum()/ (np.maximum(obj.pt.sum(), np.ones(len(obj.pt)))**2)

def FWMT2(obj):
    '''
    Second Fox-Wolfram Moment reduced to transverse plane, uses simplified solid angle
    '''
    return (Wij(obj)*(3*cosTheta(obj)**2-np.ones(len(obj.pt)))/2.).sum() / (np.maximum(obj.pt.sum(), np.ones(len(obj.pt)))**2)
    
def FWMT(obj):
    '''
    Calculate the first 5 fox wolfram moments for the given objects
    '''
    Wij_tmp = Wij(obj)
    denom = (np.maximum(obj.pt.sum(), np.ones(len(obj.pt)))**2)
    cosOmega_tmp = cosOmega(obj)
    M0 = np.ones(len(obj.pt))
    M1 = (Wij_tmp*cosOmega_tmp).sum() / denom
    M2 = (Wij_tmp*(1/2.)*(3*cosOmega_tmp**2-1.)).sum() / denom
    M3 = (Wij_tmp*(1/2.)*(5*cosOmega_tmp**3-3*cosOmega_tmp)).sum() / denom
    M4 = (Wij_tmp*(1/8.)*(35*cosOmega_tmp**4-30*cosOmega_tmp**2+3)).sum() / denom
    M5 = (Wij_tmp*(1/8.)*(63*cosOmega_tmp**5-70*cosOmega_tmp**3+15*cosOmega_tmp)).sum() / denom

    del Wij_tmp, denom, cosOmega_tmp
    return M0, M1, M2, M3, M4, M5


def sphericity(obj):
    '''
    Attempt to calculate sphericity.
    S^{a,b} = sum_i(p_i^a p_i^b) / sum(|p_i|**2)
    S = 3/2 * (l2 + l3)
    with l2 and l3 the eigenvalues of S^{a,b}, l1>l2>l3, l1+l2+l3=1
    S=1: isotropic event (l1=l2=l3=1/3)
    S=0: linear event (l2=l3=0)
    S is not infrared safe. There's a linearized version, too.
    Circularity is C = 2*l2/(l1+l2)

    Numpy lesson:
    This is how you would easily get a 3x3 matrix from two vectors.
    row = np.array([[1, 3, 2]])
    col = np.array([[1], [3], [2]])
    M = col.dot(row)
    '''
    x, y, z = obj.p4.x, obj.p4.y, obj.p4.z
    # calculate sphericity tensor by hand
    S = np.array([[(x*x).sum(), (x*y).sum(), (x*z).sum()],[(x*y).sum(),(y*y).sum(),(z*y).sum()], [(x*z).sum(), (z*y).sum(),(z*z).sum()]] / np.maximum(np.ones(len(obj.p4)),(obj.p4.p**2).sum()) )
    # S[:,:,0] shows you the first matrix
    # np.linalg.eig(S[:,:,0])[0][1:].sum() * 3/2. gives the sphericity for the first event

    # sorted eigenvalues, l0>l1>l2. S needs to be transposed for np.linalg.eig to work
    l = -np.sort(-np.linalg.eig(S.transpose())[0])
    return l[:,1:].sum(axis=1) * 3/2.
    #return (l[:,1] + l[:,2]) * 3/2. # not sure why sum won't work

def sphericityBasic(obj):
    # same as above, but only using very basic AwkwardArray elements
    x, y, z = obj.p4.fPt*np.cos(obj.p4.fPhi), obj.p4.fPt*np.sin(obj.p4.fPhi), obj.p4.fPt*np.sinh(obj.p4.fEta)
    psq = x*x+y*y+z*z
    S_ten = np.array([[(x*x).sum(), (x*y).sum(), (x*z).sum()],[(x*y).sum(),(y*y).sum(),(z*y).sum()], [(x*z).sum(), (z*y).sum(),(z*z).sum()]] / np.maximum(np.ones(len(obj.p4)),(psq).sum()) )
    l = -np.sort(-np.linalg.eig(S_ten.transpose())[0])
    del x, y, z, S_ten
    return l[:,1:].sum(axis=1) * 3/2.

def mergeArray(a1, a2):
    '''
    Merge two arrays into one, e.g. electrons and muons
    '''
    import awkward
    a1_tags = awkward.JaggedArray(a1.starts, a1.stops, np.full(len(a1.content), 0, dtype=np.int64))
    a1_index = awkward.JaggedArray(a1.starts, a1.stops, np.arange(len(a1.content), dtype=np.int64))
    a2_tags = awkward.JaggedArray(a2.starts, a2.stops, np.full(len(a2.content), 1, dtype=np.int64))
    a2_index = awkward.JaggedArray(a2.starts, a2.stops, np.arange(len(a2.content), dtype=np.int64))
    tags = awkward.JaggedArray.concatenate([a1_tags, a2_tags], axis=1)
    index = awkward.JaggedArray.concatenate([a1_index, a2_index], axis=1)
    return awkward.JaggedArray(tags.starts, tags.stops, awkward.UnionArray(tags.content, index.content, [a1.content, a2.content]))

def mt(pt1, phi1, pt2, phi2):
    '''
    Calculate MT
    '''
    return np.sqrt( 2*pt1*pt2 * (1 - np.cos(phi1-phi2)) )

def pad_and_flatten(val): 
    try:
        return val.pad(1, clip=True).fillna(0.).flatten()#.reshape(-1, 1)
    except AttributeError:
        return val.flatten()
