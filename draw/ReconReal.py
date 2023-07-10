import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as LG

from tqdm import tqdm
import sys

import tables
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
    
xx1s = []
yy1s = []
zz1s = []
EE1s = []
xx1f = []
yy1f = []
zz1f = []
EE1f = []
xx2s = []
yy2s = []
zz2s = []
EE2s = []
xx2f = []
yy2f = []
zz2f = []
EE2f = []

t01f = []
t01s = []
t02f = []
t02s = []

time1f = []
time1s = []
time2f = []
time2s = []

for i in tqdm(np.arange(258, 290)):
    with tables.open_file('/mnt/stage/douwei/Bi-Po-214/run%d_bg1.h5' % i) as h:
        xx1s.append(h.root.xx1s[:])
        yy1s.append(h.root.yy1s[:])
        zz1s.append(h.root.zz1s[:])
        EE1s.append(h.root.data1s[:])
        xx1f.append(h.root.xx1f[:])
        yy1f.append(h.root.yy1f[:])
        zz1f.append(h.root.zz1f[:])
        EE1f.append(h.root.data1f[:])
        yy2s.append(h.root.yy2s[:])
        zz2s.append(h.root.zz2s[:])
        xx2s.append(h.root.xx2s[:])
        EE2s.append(h.root.data2s[:])
        xx2f.append(h.root.xx2f[:])
        yy2f.append(h.root.yy2f[:])
        zz2f.append(h.root.zz2f[:])
        EE2f.append(h.root.data2f[:])

        time1f.append(h.root.time1f[:])
        time1s.append(h.root.time1s[:])
        time2f.append(h.root.time2f[:])
        time2s.append(h.root.time2s[:])

        t01f.append(h.root.t01f[:])
        t01s.append(h.root.t01s[:])
        t02f.append(h.root.t02f[:])
        t02s.append(h.root.t02s[:])
    
yy1s = np.hstack(yy1s)
zz1s = np.hstack(zz1s)
EE1s = np.hstack(EE1s)
xx1f = np.hstack(xx1f)
yy1f = np.hstack(yy1f)
zz1f = np.hstack(zz1f)
EE1f = np.hstack(EE1f)
yy2s = np.hstack(yy2s)
zz2s = np.hstack(zz2s)
xx2s = np.hstack(xx2s)
EE2s = np.hstack(EE2s)
xx2f = np.hstack(xx2f)
yy2f = np.hstack(yy2f)
zz2f = np.hstack(zz2f)
EE2f = np.hstack(EE2f)

time1f = np.hstack(time1f)
time1s = np.hstack(time1s)
time2f = np.hstack(time2f)
time2s = np.hstack(time2s)

t01f = np.hstack(t01f)
t01s = np.hstack(t01s)
t02f = np.hstack(t02f)
t02s = np.hstack(t02s)

from scipy.optimize import curve_fit
  
# Define the Gaussian function
def Gauss(x, A, mu, sigma, C):
    y = A*np.exp(-1*(x - mu)**2/2/sigma**2) + C
    return y

def Exp(x, A, mu, C):
    mu = mu/np.log(2)
    y = A * np.exp(-x/mu) + C
    return y

with PdfPages('ReconReal1.pdf') as pp:
    plt.figure()
    rr2 = np.sqrt((xx2s-xx2f)**2 + (yy2s-yy2f)**2 + (zz2s-zz2f)**2)
    # all cut
    index0 = (EE2f<3.3) & (EE2s>0.6) & (EE2s<1) & ((time2s-time2f)>10*1e-6) & ((time2s-time2f)<1000*1e-6) & (rr2<0.3)
    index = (EE2f<4) & (EE2s>0.6) & (EE2s<1) & ((time2s-time2f)>10*1e-6) & ((time2s-time2f)<1000*1e-6) & (rr2<0.3)
    plt.hist(EE2f[index], bins=np.arange(0.5,3.5,0.025), histtype='stepfilled', alpha=0.7)
    H, x_edges = np.histogram(EE2f[index], bins=np.arange(0.5,3.3,0.025))
    data = EE2f[index0]
    x_axis = (x_edges[1:] + x_edges[:-1])/2
    parameters, covariance = curve_fit(Gauss, x_axis, H, p0=[3000,1.8,0.5, 0], bounds=(0, 1e8))
    
    A = parameters[0]
    B = parameters[1]
    C = parameters[2]
    D = parameters[3]
    fit_y = Gauss(x_axis, A, B, C, D)

    plt.plot(x_axis, fit_y, c='k', ls='--')
    plt.text(0.5, 3200, r'$A \mathcal{N}(\mu, \sigma^2) + C$', fontsize=20)
    plt.text(0.5, 2800, r'$A = %.2f \pm %.2f$' % (parameters[0], np.sqrt(np.diag(covariance)[0])), fontsize=20)
    plt.text(0.5, 2400, r'$\mu = %.3f \pm %.3f$' % (parameters[1], np.sqrt(np.diag(covariance)[1])), fontsize=20)
    plt.text(0.5, 2000, r'$\sigma = %.3f \pm %.3f$' % (parameters[2], np.sqrt(np.diag(covariance)[2])), fontsize=20)
    plt.text(0.5, 1600, r'$C = %.2f^{+%.2f}_{-0}$' % (parameters[3], np.sqrt(np.diag(covariance)[3])), fontsize=20)
    plt.axvline(3.3, color='red',linewidth=1, alpha=0.5)
    plt.xlabel('Prompt energy/MeV')
    plt.ylabel('Counts')
    plt.tight_layout()
    pp.savefig()
    
    plt.figure()
    index = (EE2f<3.3) & ((time2s-time2f)>10*1e-6) & ((time2s-time2f)<1000*1e-6) & (rr2<0.3) & (EE2s<4)
    index0 = (EE2f<3.3) & (EE2s<1) & (EE2s>0.6) & ((time2s-time2f)>10*1e-6) & ((time2s-time2f)<1000*1e-6) & (rr2<0.3) & (EE2s<4)
    plt.hist(EE2s[index], bins=np.arange(0.5, 3.5, 0.025), histtype='stepfilled', alpha=0.7)
    H, x_edges = np.histogram(EE2s[index0], bins=np.arange(0.6, 1.0, 0.025))
    data = EE2s[index0]
    x_axis = (x_edges[1:] + x_edges[:-1])/2
    parameters, covariance = curve_fit(Gauss, x_axis, H, p0=[10000, 0.8, 0.1, 3000], maxfev=5000)
    
    A = parameters[0]
    B = parameters[1]
    C = parameters[2]
    D = parameters[3]
    
    fit_y = Gauss(x_axis, A, B, C, D)
    plt.plot(x_axis, fit_y, c='k', ls='--')
    plt.text(1.5, 14000, r'$A \mathcal{N}(\mu, \sigma^2) + C$', fontsize=20)
    plt.text(1.5, 12500, r'$A = %.2f \pm %.2f$' % (parameters[0], np.sqrt(np.diag(covariance)[0])), fontsize=20)
    plt.text(1.5, 11000, r'$\mu = %.3f \pm %.3f$' % (parameters[1], np.sqrt(np.diag(covariance)[1])), fontsize=20)
    plt.text(1.5, 9500, r'$\sigma = %.3f \pm %.3f$' % (parameters[2], np.sqrt(np.diag(covariance)[2])), fontsize=20)
    plt.text(1.5, 8000, r'$C = %.2f \pm %.2f$ ' % (parameters[3], np.sqrt(np.diag(covariance)[3])), fontsize=20)

    plt.axvline(0.6, color='red',linewidth=1, alpha=0.5)
    plt.axvline(1, color='red',linewidth=1, alpha=0.5)
    plt.xlabel('Delayed energy/MeV')
    plt.ylabel('Counts')
    plt.tight_layout()
    pp.savefig()
    
    plt.figure()
    index = (EE2f<3.3) & (EE2s>0.6) & (EE2s<1) & (rr2<0.3)
    time_diff = (time2s-time2f)*1e6
    index = index & (time_diff < 1200)
    plt.hist(time_diff[index], bins=np.arange(0, 1200, 5), histtype='stepfilled', alpha=0.7)
    
    H, x_edges = np.histogram(time_diff[index], bins=np.arange(10, 1000, 5))
    data = EE2s[index0]
    x_axis = (x_edges[1:] + x_edges[:-1])/2
    parameters, covariance = curve_fit(Exp, x_axis, H, p0=[3000, 160, 100], maxfev=5000)
    
    A = parameters[0]
    B = parameters[1]
    C = parameters[2]
    
    fit_y = Exp(x_axis, A, B, C)
    plt.plot(x_axis, fit_y, c='k', ls='--')
    plt.text(500, 2500, r'$A \exp( - x\ln2/\lambda) + C$', fontsize=20)
    plt.text(500, 2150, r'$A = %.2f \pm %.2f$' % (parameters[0], np.sqrt(np.diag(covariance)[0])), fontsize=20)
    plt.text(500, 1800, r'$\lambda = %.3f \pm %.3f$' % (parameters[1], np.sqrt(np.diag(covariance)[1])), fontsize=20)
    plt.text(500, 1450, r'$C = %.2f \pm %.2f$ ' % (parameters[2], np.sqrt(np.diag(covariance)[2])), fontsize=20)
    plt.plot(x_axis, fit_y, c='k', ls='--')
    
    plt.xlim([0,1200])
    plt.axvline(10, color='red',linewidth=1, alpha=0.5)
    plt.axvline(1000, color='red',linewidth=1, alpha=0.5)
    plt.xlabel('Delayed time/us')
    plt.ylabel('Counts')
    plt.tight_layout()
    pp.savefig()
    
    plt.figure()
    index = (EE2f<3.3) & (EE2s>0.6) & (EE2s<1) & ((time2s-time2f)>10*1e-6) & ((time2s-time2f)<1000*1e-6)
    plt.hist(1000*rr2[index], bins=np.arange(0, 1200, 10), histtype='stepfilled', alpha=0.7)
    plt.axvline(300, color='red', linewidth=1, alpha=0.5)
    plt.xlabel('Distance/mm')
    plt.ylabel('Counts')
    plt.tight_layout()
    pp.savefig()
    plt.close()