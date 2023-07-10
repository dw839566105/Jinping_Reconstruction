import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from zernike import RZern
import sys
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter, NullFormatter
import matplotlib

score = []

o1 = np.arange(40,100,10)
o2 = np.arange(10,55,5)

score = np.empty((len(o1), len(o2)))
for i_idx, i in enumerate(o1):
    for j_idx, j in enumerate(o2):
        L = pd.read_csv('/mnt/stage/douwei/JP_1t_paper/coeff/Legendre/Gather/PE/2/%d/%d.csv_1' % (i, j), index_col=0, header=None).T
        score[i_idx, j_idx] = L['PE'].values[0]

score_Z = np.empty((2,4))
for i_idx, i in enumerate(np.arange(25,45,5)):
    L = pd.read_csv('/mnt/stage/douwei/JP_1t_paper/coeff/Zernike/PE/2/shell/%d.csv' % i, index_col=0, header=None).T
    cart = RZern(i)
    score_Z[0, i_idx] = np.sum(cart.mtab>=0)
    score_Z[1, i_idx] = L['PE'].values[0]
    
score_L = np.empty((2,2))
for i_idx, i in enumerate(np.arange(20,30,5)):
    L = pd.read_csv('/mnt/stage/douwei/JP_1t_paper/coeff/dLegendre/PE/2/shell/%d/30.csv' % i, index_col=0, header=None).T
    score_L[0, i_idx] = 30*i
    score_L[1, i_idx] = L['PE'].values[0]

with PdfPages('score.pdf') as pp:
    fig, ax = plt.subplots(dpi=200)
    p1 = ax.plot(np.outer(o2,o1/2), np.max(score) - score.T + 0.02, marker='^', label='Varying\n coefficient', c='k', ls='--')
    for i in range(len(score)):
        if i < 3:
            ax.text(o1[i]*25*1.2, np.max(score) - score[i,-1], r'$%d$' % (o1[i]/2), fontsize=25, color=p1[i].get_color())
        elif i ==3:
            ax.text(1500, 0.1, r'$%d$' % (o1[i]/2), fontsize=25, color=p1[i].get_color())
        else:
            ax.text(2300, 50/ ((i-3)**3), r'$%d$' % (o1[i]/2), fontsize=25, color=p1[i].get_color())
    # p2, = ax.plot(score_Z[0], np.max(score) - score_Z[1] + 1, marker='^', ls='--', c='k', label='Zernike')
    # p3, = ax.plot(score_L[0], np.max(score) - score_L[1] + 1, marker='^', ls='dotted', c='k', label='dLeg')
    p2, = ax.plot(score_Z[0], np.max(score) - score_Z[1] + 1, marker='^', ls='-', label='Zernike')
    p3, = ax.plot(score_L[0], np.max(score) - score_L[1] + 1, marker='^', ls='dotted', label='dLeg')

    #ax.legend(list(o1) + ['Zernike', 'dLeg'], ncol=1, loc=1)
    ax.legend(handles=[p1[-1], p2, p3], ncol=1, loc='lower left')
    
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([200, 500, 1000, 2000, 4000])
    ax.get_xaxis().get_major_formatter().labelOnlyBase = False
    ax.set_xlabel('Number of parameters')
    ax.set_ylabel('Relative score')

    pp.savefig(fig)
    
    plt.figure(dpi=200)
    
    ax = plt.gca()
    ax = sns.heatmap(np.max(score) - score + 1e-2, annot=True, fmt='.0f',
                    xticklabels = list(o2),
                    yticklabels = list(np.int16(o1/2)),
                    annot_kws={"fontsize":15},
                    norm=LogNorm())
    ax.invert_yaxis()
    plt.ylabel(r'Number of parameters on $\theta$', fontsize=22)
    plt.xlabel(r'Number of parameters on $r$', fontsize=22)

    plt.tight_layout()
    pp.savefig()