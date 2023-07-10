import matplotlib as mpl
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
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=25)
plt.rcParams['font.size'] = 22
plt.rcParams['lines.markersize'] = 3
plt.rcParams['legend.markerscale'] = 2

path = '/home/douwei/Recon1t/calib/coeff_point_10_photon_1MeV/'
rd = np.hstack((np.arange(0.01,0.57,0.03), np.arange(0.57,0.64,0.01)))
coeff_1MeV = []
coeff_1MeV_error = []
for r in rd:
    h = tables.open_file(path + 'PE_30_1t_%+.3f.h5' % r)
    coeff_1MeV.append(h.root.coeff30[:])
    coeff_1MeV_error.append(h.root.std30[:])
    h.close()
coeff_1MeV = np.array(coeff_1MeV)
coeff_1MeV_error = np.array(coeff_1MeV_error)

path = '/home/douwei/Recon1t/calib/coeff_point_10_photon_2MeV/'
coeff_2MeV = []
coeff_2MeV_error = []
for r in rd:
    h = tables.open_file(path + 'PE_30_1t_%+.3f.h5' % r)
    coeff_2MeV.append(h.root.coeff30[:])
    coeff_2MeV_error.append(h.root.std30[:])
    h.close()
coeff_2MeV = np.array(coeff_2MeV)
coeff_2MeV_error = np.array(coeff_2MeV_error)

path = '/home/douwei/Recon1t/calib/coeff_point_10_photon_3MeV/'
coeff_3MeV = []
coeff_3MeV_error = []
for r in rd:
    h = tables.open_file(path + 'PE_30_1t_%+.3f.h5' % r)
    coeff_3MeV.append(h.root.coeff30[:])
    coeff_3MeV_error.append(h.root.std30[:])
    h.close()
coeff_3MeV = np.array(coeff_3MeV)
coeff_3MeV_error = np.array(coeff_3MeV_error)

fig, ax = plt.subplots(2,3, figsize=(14, 7), sharex=True, dpi=600)
ax = ax.reshape(-1)
ax0 = fig.add_subplot(111, frameon=False)
ax0.set_xlabel(r'$r/r_\mathrm{LS}$', labelpad=20)
ax0.set_ylabel(r'PE coefficients',labelpad=35)
ax0.set_xticks([])
ax0.set_yticks([])
rd = rd/0.65
for i in np.arange(6):
    mark = 2
    if i == 0:
        ax[i].plot(rd, coeff_1MeV.T[i], 
                   color='r', marker='*', linewidth=0, alpha=0.5,
                   label = '1 MeV')
        #ax[i].fill_between(rd, (coeff_1MeV - coeff_1MeV_error).T[i], 
        #                 (coeff_1MeV + coeff_1MeV_error).T[i], facecolor='r', alpha=0.5)
        ax[i].plot(rd, coeff_2MeV.T[i] - np.log(2), 
                   color='b', marker='<', linewidth=0, alpha=0.5,
                   label = '2 MeV')
        #ax[i].fill_between(rd, (coeff_2MeV - np.log(2) - coeff_2MeV_error).T[i], 
        #                 (coeff_2MeV - np.log(2) + coeff_2MeV_error).T[i], facecolor='b', alpha=0.5)
        ax[i].plot(rd, coeff_3MeV.T[i] - np.log(3), 
                   color='g', marker='^', linewidth=0, alpha=0.5,
                   label = '3 MeV')
        #ax[i].fill_between(rd, (coeff_3MeV - np.log(3) - coeff_3MeV_error).T[i], 
        #                 (coeff_3MeV - np.log(3) + coeff_3MeV_error).T[i], facecolor='g', alpha=0.5)
        ax[i].tick_params(axis = 'both', which = 'major', labelsize = 15)
    else:
        p1, = ax[i].plot(rd, coeff_1MeV.T[i], 
                   color='r', marker='*', linewidth=0, alpha=0.5,
                   label = '1 MeV')
        #ax[i].fill_between(rd, (coeff_1MeV - coeff_1MeV_error).T[i], 
        #                 (coeff_1MeV + coeff_1MeV_error).T[i], facecolor='r', alpha=0.5)
        p2, = ax[i].plot(rd, coeff_2MeV.T[i], 
                   color='b', marker='<', linewidth=0, alpha=0.5,
                   label = '2 MeV')
        #ax[i].fill_between(rd, (coeff_2MeV - coeff_2MeV_error).T[i], 
        #                 (coeff_2MeV + coeff_2MeV_error).T[i], facecolor='b', alpha=0.5)
        p3, = ax[i].plot(rd, coeff_3MeV.T[i], 
                   color='g', marker='^', linewidth=0, alpha=0.5,
                   label = '3 MeV')
        #ax[i].fill_between(rd, (coeff_3MeV - coeff_3MeV_error).T[i], 
        #                 (coeff_3MeV + coeff_3MeV_error).T[i], facecolor='g', alpha=0.5)
        ax[i].tick_params(axis = 'both', which = 'major', labelsize = 15)
    p4 = ax[i].axvline(0.88, c='k', lw=1.5, ls='dashed', label='TIR')
    ax[i].locator_params(nbins=5, axis='x')
    ax[i].locator_params(nbins=5, axis='y')
    ax[i].yaxis.grid(True, which='minor')
    # ax[i].text(0.3, 0.8, 'order %d' % i, horizontalalignment='center',
    #           verticalalignment='center', transform=ax[i].transAxes, fontsize=20)
    #ax[i].legend(title = 'order %d' % i, handles=[])
    if i == 0:
        ax[i].legend(handles=[], fontsize=20, title=r'$c^\lambda_%d(r) - \ln E$' % i)
    elif i == 5:
        ax[i].legend(handles=[p1, p2, p3, p4], loc='upper left', ncol=1, fontsize=16, title=r'$c^\lambda_%d(r)$' % i, ncol=2)
    else:
        ax[i].legend(handles=[], fontsize=20, title=r'$c^\lambda_%d(r)$' % i)

plt.tight_layout()    
fig.savefig('coeff_PE.pdf')
plt.close()