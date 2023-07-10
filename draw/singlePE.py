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
plt.rcParams['lines.markersize'] = 5
plt.rcParams['legend.markerscale'] = 1

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

fig, ax = plt.subplots()
rd = rd/0.65
mark = 10
for i in np.arange(1):

    p1, = ax.plot(rd, coeff_1MeV.T[i], 
               color='r', marker='*', linewidth=0, alpha=0.5, ms = mark,
               label = '1 MeV')
    #ax[i].fill_between(rd, (coeff_1MeV - coeff_1MeV_error).T[i], 
    #                 (coeff_1MeV + coeff_1MeV_error).T[i], facecolor='r', alpha=0.5)
    p2, = ax.plot(rd, coeff_2MeV.T[i] - np.log(2), 
               color='b', marker='<', linewidth=0, alpha=0.5, ms = mark,
               label = '2 MeV')
    #ax[i].fill_between(rd, (coeff_2MeV - np.log(2) - coeff_2MeV_error).T[i], 
    #                 (coeff_2MeV - np.log(2) + coeff_2MeV_error).T[i], facecolor='b', alpha=0.5)
    p3, = ax.plot(rd, coeff_3MeV.T[i] - np.log(3), 
               color='g', marker='^', linewidth=0, alpha=0.5, ms = mark,
               label = '3 MeV')
    #ax[i].fill_between(rd, (coeff_3MeV - np.log(3) - coeff_3MeV_error).T[i], 
    #                 (coeff_3MeV - np.log(3) + coeff_3MeV_error).T[i], facecolor='g', alpha=0.5)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
        
    ax.locator_params(nbins=5, axis='x')
    ax.locator_params(nbins=5, axis='y')
    ax.yaxis.grid(True, which='minor')
    # ax[i].text(0.3, 0.8, 'order %d' % i, horizontalalignment='center',
    #           verticalalignment='center', transform=ax[i].transAxes, fontsize=20)
    #ax[i].legend(title = 'order %d' % i, handles=[])
    ax.set_xlabel(r'$r/r_\mathrm{LS}$', fontsize=25, fontweight='normal')
    ax.set_ylabel(r'$\lambda_0 - \ln E$', fontsize=25, fontweight='normal')
    ax.legend(handles=[p1, p2, p3], loc='center left', ncol=1, fontsize=25, title='%dth PE coefficients' % i)
    
fig.align_ylabels()
fig.align_xlabels()
# plt.tight_layout()    
fig.savefig('coeff_PE0.pdf')
plt.close()