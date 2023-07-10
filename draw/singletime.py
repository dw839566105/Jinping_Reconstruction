import matplotlib as mpl
import seaborn as sns
mpl.use('pdf')
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as LG

plt.style.use('seaborn')

import matplotlib.pyplot as plt
import tables
import numpy as np

plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)
plt.rcParams['font.size'] = 25
plt.rcParams['lines.markersize'] = 10
plt.rcParams['legend.markerscale'] = 1

path = '/home/douwei/Recon1t/calib/coeff_point_10_photon_1MeV/'
rd = np.arange(0.01,0.65,0.01)

qts = ['0.01','0.05', '0.1', '0.2','0.5']
path = '/home/douwei/Recon1t/calib/coeff_point_10_photon_2MeV/'


# fig, ax = plt.subplots(2,3,figsize=(15,7), sharex=True, dpi=600, )
fig, ax = plt.subplots()

p = []
for qt in qts:
    coeff_2MeV = []
    for r in rd:
        h = tables.open_file(path + 'time_10_%s_1t_%+.3f.h5' % (qt, r))
        coeff_2MeV.append(h.root.coeff10[:])
        h.close()
    coeff_2MeV = np.array(coeff_2MeV)

    for i in np.arange(1):
        p1, = ax.plot(rd/0.65, coeff_2MeV.T[i], 
                   marker='<', linewidth=0, alpha=0.6,
                   label = r'$\tau$ = ' + qt)
        p.append(p1)
ax.set_xlabel(r'$r/r_\mathrm{LS}$', fontsize=25, fontweight='normal')
ax.set_ylabel(r'$T_0$', fontsize=25, fontweight='normal')
ax.legend(handles=p, loc='center left', ncol=2, fontsize=25, title='%dth timing coefficients' % i)
ax.locator_params(nbins=8, axis='x')
ax.locator_params(nbins=6, axis='y')
ax.yaxis.grid(True, which='minor')
ax.tick_params(axis = 'both', which = 'major', labelsize = 18)


plt.subplots_adjust(wspace=0.2, hspace=0.1)
fig.align_ylabels()
fig.align_xlabels()
#plt.tight_layout()
fig.savefig('coeff_time0.pdf')
plt.close()