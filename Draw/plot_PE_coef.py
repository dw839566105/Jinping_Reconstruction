import matplotlib as mpl
import seaborn as sns
mpl.use('pdf')
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as LG
import matplotlib.pyplot as plt
import h5py
import numpy as np

plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=25)
plt.rcParams['font.size'] = 22
plt.rcParams['lines.markersize'] = 3
plt.rcParams['legend.markerscale'] = 2

rd1 = np.arange(0.01,0.55,0.01)
rd2 = np.arange(0.552,0.638,0.002)

path = '/mnt/stage/douwei/JP_1t_github/coeff/Legendre/PE/2'
coeff = []
coeff_error = []
for r in rd1:
    with h5py.File('%s/%.2f/10.h5' % (path, r)) as h:
        coeff.append(h['coeff'][:])
        coeff_error.append(h['coeff'].attrs['std'])

for r in rd2:
    with h5py.File('%s/%.3f/10.h5' % (path, r)) as h:
        coeff.append(h['coeff'][:])
        coeff_error.append(h['coeff'].attrs['std'])

coeff = np.array(coeff)
coeff_error = np.array(coeff_error)
rd = np.hstack((rd1, rd2))/0.65

fig, ax = plt.subplots(2,3, figsize=(15, 5), sharex=True, dpi=300)
ax = ax.reshape(-1)
ax0 = fig.add_subplot(111, frameon=False)
ax0.set_xlabel(r'$r/r_\mathrm{LS}$', labelpad=20)
ax0.set_ylabel(r'PE coefficients',labelpad=35)
ax0.set_xticks([])
ax0.set_yticks([])

for i in np.arange(6):
    mark = 2
    if i == 0:
        ax[i].errorbar(rd, coeff.T[i] - np.log(2), yerr= coeff_error.T[i],
                   fmt='o', alpha=0.5,
                   label = '2 MeV')
        ax[i].tick_params(axis = 'both', which = 'major', labelsize = 20)
    else:
        p1 = ax[i].errorbar(rd, coeff.T[i], yerr= coeff_error.T[i],
                   fmt='o', alpha=0.5,
                   label = '2 MeV')
        ax[i].tick_params(axis = 'both', which = 'major', labelsize = 18)
    p2 = ax[i].axvline(0.88, c='k', lw=1.5, ls='dashed', label='TIR')
    ax[i].locator_params(nbins=5, axis='x')
    ax[i].locator_params(nbins=5, axis='y')
    ax[i].yaxis.grid(True, which='minor')
    ax[i].set_xlim(0,1)

    if i == 0:
        ax[i].legend(handles=[], fontsize=15, title=r'$c^\lambda_%d(r) - \ln E$' % i)
    elif i == 5:
        ax[i].legend(handles=[p1, p2], loc='upper left', ncol=2, fontsize=15, title=r'$c^\lambda_%d(r)$' % i)
    else:
        ax[i].legend(handles=[], fontsize=15, title=r'$c^\lambda_%d(r)$' % i)

plt.tight_layout()    
fig.savefig('coeff_PE.pdf')
plt.close()