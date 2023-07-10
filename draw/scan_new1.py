import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import tables
from polynomial import *
from tqdm import *

class LH:
    def Likelihood(vertex, *args, expect=False):
        '''
        vertex[1]: r
        vertex[2]: theta
        vertex[3]: phi
        '''
        PMT_pos, pe_array, coeff, cart = args
        basis = LH.Calc_basis(vertex, PMT_pos, cart)
        L1, energy = LH.Likelihood_PE(basis, pe_array, coeff)
        if expect:
            return energy
        else:
            # L2 = LH.Likelihood_Time(basis_time, vertex[-1], fired_PMT, time_array, coeff_time)
            return L1


    def Calc_basis(vertex, PMT_pos, cart): 
        # boundary
        v = vertex[:3]
        rho = np.linalg.norm(v)
        if rho > 1-1e-3:
            rho = 1-1e-3
        # calculate cos theta
        cos_theta = np.dot(v, PMT_pos.T) / (np.linalg.norm(v)*np.linalg.norm(PMT_pos,axis=1))
        cos_theta = np.nan_to_num(cos_theta)
        theta = np.arccos(cos_theta)
        # Generate Zernike basis
        rho = rho + np.zeros(len(PMT_pos))
        zo = cart.mtab>=0
        zs_radial = cart.coefnorm[zo, np.newaxis] * polyval(cart.rhotab.T[:, zo, np.newaxis], rho)
        zs_angulars = angular(cart.mtab[zo].reshape(-1, 1), theta)
        return (zs_radial * zs_angulars).T

    def Likelihood_PE(basis, pe_array, coeff_pe):
        expect = np.exp(np.matmul(basis, coeff_pe))

        # Energy fit
        nml = np.sum(pe_array)/np.sum(expect)
        expect *= nml
        lnL = - pe_array * np.log(expect) + expect
        return lnL.sum(), nml
    
def r2c(c):
    v = np.zeros(3)
    v[2] = c[0] * np.cos(c[1]) #z
    rho = c[0] * np.sin(c[1])
    v[0] = rho * np.cos(c[2]) #x
    v[1] = rho * np.sin(c[2]) #y
    return v

def c2r(c):
    v = np.zeros(3)
    v[0] = np.linalg.norm(c)
    v[1] = np.arccos(c[2]/(v[0]+1e-6))
    #v[2] = np.arctan(c[1]/(c[0]+1e-6)) + (c[0]<0)*np.pi
    v[2] = np.arctan2(c[1],c[0])
    return v

from zernike import RZern
cart = RZern(30)

with tables.open_file('/mnt/stage/douwei/JP_1t_paper/coeff/Zernike/PE/2/shell/30.h5') as h:
    coeff = h.root.coeff[:]
PMT = np.loadtxt('../PMT_1t.txt')

import ROOT
from tqdm import tqdm

tTruth = ROOT.TChain("SimTriggerInfo")


radius = 0.1
for i in np.arange(0,1):
    if i == 0:
        tTruth.Add('/mnt/stage/douwei/JP_1t_paper/root/point/z/2/%.2f.root' % radius)
    else:
        tTruth.Add('/mnt/stage/douwei/JP_1t_paper/root/point/z/2/%.2f_%d.root' % (radius, i))

tr = []
pid = []

for event in tTruth:
    for PE in event.PEList:
        tr.append(event.TriggerNo)
        pid.append(PE.PMTId)
        
H, _, _ = np.histogram2d(tr, pid, bins=(np.arange(1,np.max(tr)+2), np.arange(31)))

from matplotlib.backends.backend_pdf import PdfPages
fig, ax1 = plt.subplots(dpi=100)

with PdfPages('scan1.pdf') as pp:
    for midx in tqdm([0,]):
        x0 = np.array((0, 0, 0.09/0.65))
        result_in = minimize(LH.Likelihood, x0=x0, method='SLSQP', 
                    bounds=((-1, 1), (-1, 1), (-1, 1)),
                    args = (PMT, H[midx], coeff, cart), tol=1e-10)
        result_in.x = np.array((0, 0, 0.1))/0.65
        x0 = np.array((0, 0, 0.6/0.65))
        result_out = minimize(LH.Likelihood, x0=x0, method='SLSQP', 
                    bounds=((-1, 1), (-1, 1), (-1, 1)),
                    args = (PMT, H[midx], coeff, cart), tol=1e-10)
        scan = np.arange(-0.19, 1.08, 0.01)
        print(result_in.x, result_out.x)
        vec = result_out.x - result_in.x
        if(np.linalg.norm(vec)<0.001):
            vec = result_in.x / np.linalg.norm(result_in.x)
        L = []
        r = []
        cos = []
        for j in scan:
            point = result_in.x + vec * j
            L.append(LH.Likelihood(point, *(PMT, H[midx], coeff, cart)))
            cos.append(np.exp(np.matmul(LH.Calc_basis(point, PMT, cart), coeff)))
            r.append(np.linalg.norm(point))
        L = np.array(L)
        # cosine = (np.log(cos)*np.log(base)).sum(-1)/np.linalg.norm(np.log(cos), axis=-1)/np.linalg.norm(np.log(base))
        base = np.exp(np.matmul(LH.Calc_basis([0, 0, 0.1/0.65], PMT, cart), coeff))
        cosine =1 - (np.array(cos)*np.array(base)).sum(-1)/np.linalg.norm(np.array(cos), axis=-1)/np.linalg.norm(np.array(base))
        ax1.plot(0.65*np.array(r)*1000, L - np.min(L), 'g-')
        ax1.axvline(0.65*r[17]*1000, c='k', ls='--', lw=1)
        p1 = ax1.axvline(0.65*r[119]*1000, c='k', ls='--', lw=1, label='Local Minimum')
        p2 = ax1.axvline(0.1*1000, c='r', ls='-', lw=1, label='Truth')
        
        ax1.tick_params(axis = 'both', which = 'major', labelsize = 25)
        ax1.set_xlabel('Vertex radius/mm', fontsize=30)
        ax1.set_ylabel('$-\log\mathcal{L}$', color='g', fontsize=30)

        ax1.set_yticks([])
        ax1.legend(handles=[p1, p2], fontsize=30, loc=2)
        #pp.savefig(fig)
    ax2 = ax1.twinx()
    ax2.set_yticks([])
    ax2.plot(0.65*np.array(r)*1000, cosine, 'b-')
    ax2.set_ylabel('Cosine distance', color='b')
    pp.savefig(fig)