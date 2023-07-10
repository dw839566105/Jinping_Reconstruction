import matplotlib as mpl
import seaborn as sns
mpl.use('pdf')
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as LG

import matplotlib.pyplot as plt
import tables, h5py
import numpy as np

plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=25)
plt.rcParams['font.size'] = 22
plt.rcParams['lines.markersize'] = 3
plt.rcParams['legend.markerscale'] = 2

path = '/home/douwei/Recon1t/calib/coeff_point_10_photon_1MeV/'
rd = np.arange(0.01,0.65,0.01)

qts = [1,2,3]
path = '/mnt/stage/douwei/JP_1t_paper/coeff/Legendre/Time/'


fig, ax = plt.subplots(2,3,figsize=(15,5), sharex=True, dpi=600, )
ax0 = fig.add_subplot(111, frameon=False)
ax0.set_xlabel(r'$r/r_\mathrm{LS}$',labelpad=20)
ax0.set_ylabel(r'Timing coefficients')
ax0.set_xticks([])
ax0.set_yticks([])
ax = ax.reshape(-1)

ylims = ((27.3, 28.7), 
        (-3.5, 0.3),
        (-0.8, 0.3),
        (-1.5, 0.3), 
        (-1.2, 0.3),
        (-2.0, 0.3))
p = []
for qt in qts:
    coeff = []
    stds = []
    for r in rd:
        if (qt == 1) and (r>0.61):
            coeff.append(np.full_like(coef, np.nan))
            stds.append(np.full_like(std, np.nan))
        else:
            if r < 0.55:
                with h5py.File(path + '%s/%.2f/30.h5' % (qt, r)) as h:
                    coef = h['coeff'][:]
                    std = h['coeff'].attrs['std']
            else:
                with h5py.File(path + '%s/%.3f/30.h5' % (qt, r)) as h:
                    coef = h['coeff'][:]
                    std = h['coeff'].attrs['std']
            coeff.append(coef)
            stds.append(std)
    coeff = np.array(coeff)
    stds = np.array(stds)

    for i in np.arange(6):
        # p1, = ax[i].plot(rd/0.65, coeff.T[i], 
        #           marker='<', linewidth=0, alpha=0.6,
        #           markersize=5,
        #           label = r'%s MeV' % qt)
        p1 = ax[i].errorbar(rd/0.65, coeff.T[i], yerr = stds.T[i], 
                            fmt='o',
                            label = r'%s MeV' % qt)
        p2 = ax[i].axvline(0.88, c='k', lw=1, ls='dashed', label='TIR')
        # if i == 4:
        #    ax[i].set_xlabel(r'$r/r_\mathrm{LS}$', fontsize=20, fontweight='normal')
        # if not i % 3:
        #    ax[i].set_ylabel(r'Coefficients', fontsize=20, fontweight='normal')

        ax[i].legend(handles=[], fontsize=15, title=r'$c^T_%d(r)$' % i)
        ax[i].locator_params(nbins=8, axis='x')
        # ax[i].locator_params(nbins=6, axis='y')
        ax[i].yaxis.grid(True, which='minor')
        ax[i].tick_params(axis = 'both', which = 'major', labelsize = 18)
        ax[i].set_xlim(0,1)
        ax[i].set_ylim(ylims[i])
    p.append(p1)
p.append(p2)
ax[i].legend(handles=p, loc='lower left', fontsize=15, ncol=2, title=r'$c^T_%d(r)$' % i)


plt.subplots_adjust(wspace=0.2, hspace=0.1)
fig.align_ylabels()
fig.align_xlabels()
#plt.tight_layout()
fig.savefig('coeff_time_new.pdf')
plt.close()