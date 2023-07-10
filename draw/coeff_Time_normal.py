import matplotlib as mpl
import seaborn as sns
mpl.use('pdf')
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as LG

import matplotlib.pyplot as plt
import tables
import numpy as np

plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=25)
plt.rcParams['font.size'] = 22
plt.rcParams['lines.markersize'] = 3
plt.rcParams['legend.markerscale'] = 1

path = '/home/douwei/Recon1t/calib/coeff_point_10_photon_1MeV/'
rd = np.arange(0.01,0.65,0.01)

qts = ['0.01','0.05', '0.1', '0.2','0.5']
path = '/home/douwei/Recon1t/calib/coeff_point_10_photon_2MeV/'


fig, ax = plt.subplots(2,3,figsize=(15,7), sharex=True, dpi=600, )
ax0 = fig.add_subplot(111, frameon=False)
ax0.set_xlabel(r'$r/r_\mathrm{LS}$',labelpad=20)
ax0.set_ylabel(r'Timing coefficients')
ax0.set_xticks([])
ax0.set_yticks([])
ax = ax.reshape(-1)

p = []
for qt in qts:
    coeff_2MeV = []
    for r in rd:
        h = tables.open_file(path + 'time_10_%s_1t_%+.3f.h5' % (qt, r))
        coeff_2MeV.append(h.root.coeff10[:])
        h.close()
    coeff_2MeV = np.array(coeff_2MeV)

    for i in np.arange(6):
        p1, = ax[i].plot(rd/0.65, coeff_2MeV.T[i], 
                   marker='<', linewidth=0, alpha=0.6,
                   markersize=5,
                   label = r'$\tau$ = ' + qt)
        p2 = ax[i].axvline(0.88, c='k', lw=1, ls='dashed', label='TR')
        # if i == 4:
        #    ax[i].set_xlabel(r'$r/r_\mathrm{LS}$', fontsize=20, fontweight='normal')
        # if not i % 3:
        #    ax[i].set_ylabel(r'Coefficients', fontsize=20, fontweight='normal')

        ax[i].legend(handles=[], fontsize=20, title=r'$c^T_%d(r)$' % i)
        ax[i].locator_params(nbins=8, axis='x')
        ax[i].locator_params(nbins=6, axis='y')
        ax[i].yaxis.grid(True, which='minor')
        ax[i].tick_params(axis = 'both', which = 'major', labelsize = 20)
        ax[i].set_xlim(0,1)
    p.append(p1)
p.append(p2)
ax[i].legend(handles=p, loc='lower left', fontsize=15, ncol=2, title=r'$c^T_%d(r)$' % i)


plt.subplots_adjust(wspace=0.2, hspace=0.1)
fig.align_ylabels()
fig.align_xlabels()
#plt.tight_layout()
fig.savefig('coeff_time.pdf')
plt.close()