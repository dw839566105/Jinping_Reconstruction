import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as mpl
# mpl.rc('font', family='sans-serif')
up = 150
t = np.arange(-3, up)
T = 27.5
tau = 0.1
ts = 3
y1 = (1-tau)*(t<T)*(T-t) + (tau)*(t>T)*(t-T)

import uproot
with uproot.open('/mnt/stage/douwei/JP_1t_paper/root/point/x/2/0.00.root') as h:
    pulse = h['SimTriggerInfo']['PEList/PEList.PulseTime'].array(library='np')
    a, b = np.histogram(np.hstack(pulse), bins=np.arange(0,300,1))
    
a = a/np.sum(a)
b = (b[:-1] + b[1:])/2
a = a[b<up]
b = b[b<up]
with PdfPages('QRpdf.pdf') as pp:
    fig, [ax1, ax2] = plt.subplots(2,1, sharex='col')
    ax1.plot(t, y1)
    ax1.axvline(T, color='k', lw=1, ls='--')
    ax1.set_ylabel(r'$\mathcal{R}_{\tau}$', labelpad=10)
    
    for ts_ in [2, 3, 5]:
        y2 = np.exp(-y1/ts_) / (ts_/((1-tau)*(tau)))
        ax2.plot(t, y2, label='$t_s$ = %.1f' % ts_, alpha=0.7)
    ax2.plot(b, a, label='MC', lw=2, color='k')
    ax2.axvline(T, color='k', lw=1, ls='--')
    ax2.legend(fontsize=18)
    ax2.set_xlabel('Time/ns')
    ax2.set_ylabel('$R$')
    pp.savefig(fig)
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ts_ = 3
    y2 = np.exp(-y1/ts_) / (ts_/((1-tau)*(tau)))
    ax.plot(t, np.cumsum(y2), label='Fit')
    ax.plot(b, np.cumsum(a), label='MC',lw=2, color='k')
    id1 = np.where(np.abs(b - T) == np.min(np.abs(b-T)))[0][0]
    ax.axvline(T, color='k', lw=1, ls='--')
    ax.axhline(np.cumsum(a)[id1], color='k', lw=1, ls='--')
    ax.set_xlabel('Time/ns')
    ax.set_ylabel('Quantile value')
    ax.legend()
    pp.savefig(fig)

    
