from EFT_multidim import get_NLL
import os
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
plt.style.use(hep.style.CMS)


years = ['2018']

x_vals = np.arange(-10,11)

def cpt_eq(x):
    return 0.004126797600997696*x**2 + 0.06930915985048229*x + 0.9999999999999998

def cpqm_eq(x):
    return 0.004108122982236317*x**2 - 0.10496295789483141*x + 1.0000000000000013

cpt_s = []
cpt = []
cpqm_s = []
cpqm = []

res_SM = get_NLL(years = years,point = [0, 0, 0, 0, 0, 0], bsm_scales={'TTZ':1}, systematics=True)

for i in x_vals:
    res = get_NLL(years = years,point = [0, i, 0, 0, 0, 0], bsm_scales={'TTZ': cpt_eq(i)}, systematics=True)
    cpt_s.append(res-res_SM)

for i in x_vals:
    res = get_NLL(years = years,point = [0, i, 0, 0, 0, 0], bsm_scales={'TTZ':1}, systematics=True)
    cpt.append(res-res_SM)

for i in x_vals:
    res = get_NLL(years = years,point = [0, 0, i, 0, 0, 0], bsm_scales={'TTZ': cpqm_eq(i)}, systematics=True)
    cpqm_s.append(res-res_SM)

for i in x_vals:
    res = get_NLL(years = years,point = [0, 0, i, 0, 0, 0], bsm_scales={'TTZ':1}, systematics=True)
    cpqm.append(res-res_SM)


plot_dir = "/home/users/hbronson/public_html/tW_scattering/"

plt.figure()
fig, ax = plt.subplots()
hep.cms.label(
        "Work in progress",
        data=True,
        #year=2018,
        lumi=60.0,#+41.5+35.9,
        loc=0,
        ax=ax,
        )

plt.plot(x_vals, cpt, label='ttZ constant', color='r')
plt.plot(x_vals, cpt_s, label='ttZ scaled', color='b')
plt.ylabel(r'$\sigma/\sigma_{SM}$')
plt.xlabel(r'$C_{\varphi t}$')
plt.legend()
plt.savefig(os.path.expandvars(plot_dir+'cpt.png'))


plt.figure()
fig, ax = plt.subplots()
hep.cms.label(
        "Work in progress",
        data=True,
        #year=2018,
        lumi=60.0,#+41.5+35.9,
        loc=0,
        ax=ax,
        )


plt.plot(x_vals, cpqm, label='ttZ constant', color='r')
plt.plot(x_vals, cpqm_s, label='ttZ scaled', color='b')
plt.ylabel(r'$\sigma/\sigma_{SM}$')
plt.xlabel(r'$C_{\varphi Q}^{-}$')
plt.legend()
plt.savefig(os.path.expandvars(plot_dir+'cpqm.png'))
