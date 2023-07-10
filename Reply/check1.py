import sys, os
import matplotlib.pyplot as plt
import numpy as np
import h5py, uproot
from tqdm import *
from numba import njit

fileno = eval(sys.argv[1])
@njit
def legval_raw(x, c):
    if len(c) == 1:
        return c[0]
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1*(nd - 1))/nd
            c1 = tmp + (c1*x*(2*nd - 1))/nd
    return c0 + c1*x

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

trigger_fast = []
trigger_slow = []

ChannelID = []
Dt = np.loadtxt('/mnt/stage/douwei/PMTTimeCalib_Run257toRun262.txt')
dirname = '/mnt/eternity/Jinping_1ton_Data/01_RawData/run00000%d/' % fileno
filelist = os.listdir(dirname)
length = len(filelist) - 1
#length = 20

for i in trange(length):
    if i == 0:
        filename = '%s%d.h5' % (dirname+filelist[0][:31], fileno)
    else:
        filename = '%s%d_%d.h5' % (dirname+filelist[0][:31], fileno, i)
    with h5py.File(filename) as h:
        data = h['Readout/TriggerInfo'][:]
        time = data['Sec'] + 1e-9*data['NanoSec']
        diff = np.diff(time)
        index = (diff>1*1e-6) & (diff<10)
        trigger_fast.append(data['TriggerNo'][:-1][index])
        trigger_slow.append(data['TriggerNo'][1:][index])

PMT = np.loadtxt('/home/douwei/Recon1t/calib/PMT_1t.txt')

with h5py.File('/mnt/stage/douwei/JP_1t_paper/coeff/Legendre/Gather1~/Time/2/20/20.h5') as h:
    coef = h['coeff'][:]

trigger_fast = np.hstack(trigger_fast)
trigger_slow = np.hstack(trigger_slow)
'''
with h5py.File('/mnt/stage/douwei/Bi-Po-214/run%d_bg1.h5' % fileno) as h:
    #xx1s = h['xx1s'][:]
    #yy1s = h['yy1s'][:]
    #zz1s = h['zz1s'][:]
    #EE1s = h['data1s'][:]
    #time1s = h['time1s'][:]
    
    #xx1f = h['xx1f'][:]
    #yy1f = h['yy1f'][:]
    #zz1f = h['zz1f'][:]
    #EE1f = h['data1f'][:]
    #time1f = h['time1f'][:]
    
    xx2s = h['xx2s'][:]
    yy2s = h['yy2s'][:]
    zz2s = h['zz2s'][:]
    EE2s = h['data2s'][:]
    time2s = h['time2s'][:]
    
    xx2f = h['xx2f'][:]
    yy2f = h['yy2f'][:]
    zz2f = h['zz2f'][:]
    EE2f = h['data2f'][:]
    time2f = h['time2f'][:]
    
rr2 = np.sqrt((xx2s-xx2f)**2 + (yy2s-yy2f)**2 + (zz2s-zz2f)**2)
# all cut
index1 = (EE2f>0.73) & (EE2f<0.87) & (EE2s>0.6) & (EE2s<1) & ((time2s-time2f)>10*1e-6) & ((time2s-time2f)<1000*1e-6) & (rr2<0.3)

trigger_fast_ = trigger_fast[index1]
trigger_slow_ = trigger_slow[index1]

vf = np.vstack((xx2f[index1], yy2f[index1], zz2f[index1])).T
vs = np.vstack((xx2s[index1], yy2s[index1], zz2s[index1])).T
'''
time_f = []
weight_f = []
energy_f = []
time_s = []
weight_s = []
energy_s = []

for d in np.arange(0, length):
    with h5py.File('/mnt/eternity/charge/run00000%d/%d.h5' % (fileno, d)) as h:
        TriggerNo = h['AnswerWF']['TriggerNo']
        ChannelID = h['AnswerWF']['ChannelID']
        HitPosInWindow = h['AnswerWF']['HitPosInWindow']
        Charge = h['AnswerWF']['Charge']
    
    with h5py.File('/mnt/eternity/recon/run00000%d/%d.h5' % (fileno, d)) as h:
        rc = h['Recon'][:]
        ids = rc['Likelihood_in'] > rc['Likelihood_out']
        x = rc['x_sph_in']
        y = rc['y_sph_in']
        z = rc['z_sph_in']
        x[ids] = rc['x_sph_out'][ids]
        y[ids] = rc['y_sph_out'][ids]
        z[ids] = rc['z_sph_out'][ids]
        v = np.float64(np.vstack((x, y, z)).T)
        
    a1, b1 = np.unique(trigger_fast_, return_index = True)
    a2, b2 = np.unique(trigger_slow_, return_index = True)

    for i in tqdm(np.unique(TriggerNo)):
        if np.isin(i, trigger_fast_):
            idx = TriggerNo == i
            charge = Charge[idx]
            hit_time = HitPosInWindow[idx]
            pmtId = ChannelID[idx]
            resort = b1[a1 == i]
            r = np.linalg.norm(v[ii])
            theta = np.sum((PMT[pmtId] * v[ii]), axis=1) / r / np.linalg.norm(PMT[pmtId], axis=1)

            cut = len(coef)
            t_basis = legval_raw(np.cos(theta), np.eye(cut).reshape((cut,cut,1))).T
            # r_basis = legval_raw(rhof, coef.T.reshape(coef.shape[1], coef.shape[0],1)).T
            r_basis = legval_raw(r, coef.T.reshape(coef.shape[1], coef.shape[0],1)).T
            probe = (t_basis*r_basis).sum(-1)
            corr = hit_time - probe - Dt[pmtId,-2]
            corr = corr - weighted_quantile(corr, 0.15, charge)
            e_f = EE2f[resort] * np.ones_like(corr)
            time_f.append(corr)
            weight_f.append(charge)
            energy_f.append(e_f)
            
        if np.isin(i, trigger_slow_):
            idx = TriggerNo == i
            charge = Charge[idx]
            hit_time = HitPosInWindow[idx]
            pmtId = ChannelID[idx]
            resort = b2[a2 == i]
            r = np.linalg.norm(v[ii])
            theta = np.sum((PMT[pmtId] * v[ii]), axis=1) / r / np.linalg.norm(PMT[pmtId], axis=1)

            cut = len(coef)
            t_basis = legval_raw(np.cos(theta), np.eye(cut).reshape((cut,cut,1))).T
            # r_basis = legval_raw(rhof, coef.T.reshape(coef.shape[1], coef.shape[0],1)).T
            r_basis = legval_raw(r, coef.T.reshape(coef.shape[1], coef.shape[0],1)).T
            probe = (t_basis*r_basis).sum(-1)
            corr = hit_time - probe - Dt[pmtId,-2]
            corr = corr - weighted_quantile(corr, 0.15, charge)
            e_s = EE2s[resort] * np.ones_like(corr)
            time_s.append(corr)
            weight_s.append(charge)
            energy_s.append(e_s)
            
with h5py.File('%d.h5' % fileno, 'w') as h:
    h['time_f'] = np.hstack(time_f)
    h['time_s'] = np.hstack(time_s)
    h['weight_f'] = np.hstack(weight_f)
    h['weight_s'] = np.hstack(weight_s)
    h['energy_f'] = np.hstack(energy_f)
    h['energy_s'] = np.hstack(energy_s)
    
plt.figure(dpi=200)
plt.hist(np.hstack(time_f), weights = np.hstack(weight_f), bins=np.linspace(-30,230,261), density=True, histtype='step', label='Fast')
plt.hist(np.hstack(time_s), weights = np.hstack(weight_s), bins=np.linspace(-30,230,261), density=True, histtype='step', label='Slow')
plt.legend()
plt.xlabel('Time/ns')
plt.savefig('%d.pdf' % fileno)
