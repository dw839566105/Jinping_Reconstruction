import tables
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.polynomial import legendre as LG
from matplotlib.backends.backend_pdf import PdfPages

plt.style.use('seaborn')
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)
plt.rcParams['font.size'] = 20
plt.rcParams['lines.markersize'] = 5
plt.rcParams['legend.markerscale'] = 2

def readfile(file):
    for index, i in enumerate(rd):
        filename = f'{file}/{i:.2f}.h5' if i < 0.55 else f'{file}/{i:.3f}.h5'
        h = tables.open_file(filename)
        if index == 0:
            data = pd.DataFrame(h.root.coeff[:][:,np.newaxis].T)
        else:
            data = pd.concat([data, pd.DataFrame(h.root.coeff[:][:,np.newaxis].T)])
        h.close()
    return data

rd = np.hstack((np.arange(0.01,0.55,0.01), np.arange(0.55, 0.644, 0.002)))

with PdfPages('Gather.pdf') as pdf:
    for types in ['PE', 'Time']:
        data = readfile(f'../coeff/Legendre/{types}/10')
        fig, ax = plt.subplots(2,3, figsize=(14, 7), sharex=True, dpi=600)
        ax = ax.reshape(-1)
        for i in np.arange(0, 6):
            plt.subplot(2, 3, i+1)
            x = rd/0.65
            y = data[pd.RangeIndex(start=i, stop = i+1)].values
            p = ax[i].plot(x, y, 
                color='r', marker='*', markersize=5, linewidth=0, alpha=0.5,
                label = '2 MeV')

            X = np.hstack((x, -x))
            if not i%2:
                Y = np.vstack((y, y))
            else:
                Y = np.vstack((y, -y))

            B, _ = LG.legfit(X, Y, deg = 80, full = True)
            res = LG.legval(x, B)
            ax[i].plot(x, res.flatten(), label = 'fit', alpha=0.5)
            ax[i].tick_params(axis = 'both', which = 'major', labelsize = 15)
            ax[i].tick_params(axis = 'both', which = 'major', labelsize = 15)
            ax[i].locator_params(nbins=5, axis='x')
            ax[i].locator_params(nbins=5, axis='y')
            ax[i].yaxis.grid(True, which='minor')
            ax[i].text(0.3, 0.8, 'order %d' % i, horizontalalignment='center',
                verticalalignment='center', transform=ax[i].transAxes, fontsize=20)

        fig.text(0.5, 0.02, r'Relative radius', ha='center', fontsize=20)
        fig.text(0.04, 0.5, r'Coefficients', va='center', rotation='vertical', fontsize=20)
        fig.legend(handles=p, loc='upper right', ncol=3, fontsize=20, bbox_to_anchor=(0.91, 0.98))
        fig.align_ylabels()
        fig.align_xlabels()
        pdf.savefig(fig)
        plt.close()
