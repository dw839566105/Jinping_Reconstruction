import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

###### all radius should be normalized
N0 = 50000
N1 = 30
radius = 0.65
pmt = np.array((0, 1.3/radius))


##theta = np.arccos(np.linspace(-1,1,N))

theta = np.linspace(-np.pi, np.pi, N0)
v = 0.95 * np.vstack((np.sin(theta), np.cos(theta))).T

def ratio(v):
    a0 = (v[:,1] - pmt[1])/(v[:,0] - pmt[0])
    a1 = - v[:,0] * a0 + v[:,1]
    a = a0**2 + 1
    b = 2*a0*a1
    c = a1**2 - 1**2
    delta = np.sqrt(b**2 - 4*a*c)
    #x1 = (-b - delta)/2/a
    x2 = (-b + delta)/2/a
    intercept = np.vstack((x2, a0*x2 + a1)).T
    dist = np.linalg.norm(pmt - v, axis=1)
    cth = np.sum((intercept-v)*intercept, axis=1)/np.linalg.norm((intercept-v), axis=1)
    #r = np.linalg.norm(v, axis=1)
    th1 = np.arccos(np.clip(cth, -1, 1))
    th2 = np.nan_to_num(np.arcsin(np.sin(th1)*1.5/1.33), nan = np.pi/2)
    t_ratio = 2*1.5*np.cos(th1)/(1.5*np.cos(th1) + 1.33*np.cos(th2))
    tr = 1 - (t_ratio - 1)**2
    expect = np.nan_to_num(cth/dist**2 * np.nan_to_num(tr))

    return expect
'''
def ratio(v):
    dist = np.linalg.norm(pmt - v, axis=1)
    cth = np.sum((pmt-v)*pmt, axis=1)/np.linalg.norm((pmt-v), axis=1)/np.linalg.norm(pmt)
    expect = cth/dist**2 #*np.nan_to_num(t_ratio)
    return expect
'''
expect = ratio(v)


dist_t = []

with PdfPages('theta_min_1.3.pdf') as pdf:
    fig = plt.figure(dpi=200)
    plt.plot(expect)
    pdf.savefig(fig)

    for i in np.arange(2, N1):
        data0 = expect[::np.int64(N0/i)]
        v = 0.99*np.vstack((np.sin(theta), np.cos(theta))).T
        expect = ratio(v)
        dist = []
        for k in np.arange(N0):
            data = np.roll(expect, k)[::np.int64(N0/i)]
            dist.append(1 - np.sum(data0*data)/np.linalg.norm(data0)/np.linalg.norm(data))

        fig = plt.figure(dpi=200)
        plt.plot(np.linspace(0, 2*np.pi, N0), np.array(dist))
        pdf.savefig(fig)