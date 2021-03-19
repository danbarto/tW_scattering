import ROOT,os
import ctypes
import copy
import math

def getContours(h, plotDir):
    _h = h.Clone()
    contlist = [0.5,1,2]
    idx = contlist.index(1)
    c_contlist = ((ctypes.c_double)*(len(contlist)))(*contlist)
    ctmp = ROOT.TCanvas()
    _h.SetContour(len(contlist),c_contlist)
    _h.Draw("contzlist")
    _h.GetZaxis().SetRangeUser(0.01,3)
    ctmp.Update()
    contours = ROOT.gROOT.GetListOfSpecials().FindObject("contours")
    graph_list = contours.At(idx)
    contours = []
    np = 0
    #print contours, ROOT.gROOT.GetListOfSpecials(), graph_list.GetEntries()
    for i in range(graph_list.GetEntries()):
            contours.append( graph_list.At(i).Clone("cont_"+str(i)) )
    for c in contours:
        c.Draw('same')
    ctmp.Print(os.path.join(plotDir, h.GetName()+".png"))
    _h.Draw("colz")
    for c in contours:
        c.Draw('same')
    ctmp.Print(os.path.join(plotDir, h.GetName()+"_colz.png"))
    del ctmp
    return contours

def cleanContour(g, model="T2tt"):
    x, y = ROOT.Double(), ROOT.Double()
    remove=[]
    for i in range(g.GetN()):
        g.GetPoint(i, x, y)
        if model=="T2tt" and False:
            if  (x<1100):# or x-y<200 or y>550 or x>1000:
                remove.append(i)
        else: pass# print model, "not implemented"
    for i in reversed(remove):
        g.RemovePoint(i)


    #if model=="T8bbllnunu_XCha0p5_XSlep0p05":
    #    for i in range(g.GetN()):
    #        print i, x, y
    #    #g.SetPoint(500,15)

def extendContour(g):
    x, y = ROOT.Double(), ROOT.Double()
    points = []
    for i in range(g.GetN()):
        g.GetPoint(i, x, y)
        points.append((copy.deepcopy(x),copy.deepcopy(y)))


    f = (2*points[-1][0]-points[-2][0], 2*points[-1][1]-points[-2][1])
    l = (2*points[0][0]-points[1][0], 2*points[0][1]-points[1][1])
    points.insert(0,l)
    points.append(f)
    for i,p in enumerate(points):
        g.SetPoint(i, p[0], p[1])

def getPoints(g):
    x, y = ROOT.Double(), ROOT.Double()
    points = []
    for i in range(g.GetN()):
        g.GetPoint(i, x, y)
        points.append((copy.deepcopy(x),copy.deepcopy(y)))

    return points

def getProjection(x, y, ref_x, ref_y):
    delta_x = x - ref_x
    delta_y = y - ref_y
    r = math.sqrt(delta_x**2 + delta_y**2)
    phi = math.atan2(delta_y,delta_x)
    return {'r':r, 'phi':phi}

