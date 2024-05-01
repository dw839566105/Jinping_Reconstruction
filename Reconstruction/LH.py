#!/usr/bin/env python3
'''
重建所需文件读入
'''
import numpy as np
from DetectorConfig import chs

def LogLikelihood_PE(E, pe_array, expect):
    '''
    PE 项似然函数
    '''
    expect[expect == 0] = 1E-6
    expect1  = expect * E
    lnL = pe_array * np.log(expect1) - expect1
    return lnL.sum()

def LogLikelihood_Time(Ti, time_array):
    lnL = LogLikelihood_quantile(time_array, Ti, 0.1, 3)
    return -lnL.sum()

def LogLikelihood_quantile(y, T_i, tau, ts):
    # less = T_i[y<T_i] - y[y<T_i]
    # more = y[y>=T_i] - T_i[y>=T_i]
    # R = (1-tau)*np.sum(less) + tau*np.sum(more)
    # since lucy ddm is not sparse, use PE as weight
    L = (T_i-y) * (y<T_i) * (1-tau) + (y-T_i) * (y>=T_i) * tau
    return L/ts

def genPE(Z):
    if 's0' in Z.columns:
        return Z.set_index('ch').reindex(range(chs), fill_value=0)['s0'].values
    elif 'PEt' in Z.columns:
        return np.array(Z['ch'].value_counts().reindex(range(chs), fill_value=0))
    else:
        raise Exception("Data Error")

def genPEt(Z, PEchain):
    if 'PEt' in Z.columns:
        return Z["PEt"].values + Z["offset"].values
    else:
        raise Exception("Data Error")

def genFired(Z):
    return Z["ch"].values

def LogLikelihood(vertex, Z, probe, time_mode, data_mode):
    pe_array = genPE(Z)
    expect = probe.callPE(vertex)
    L1 = LogLikelihood_PE(vertex[3], pe_array, expect)
    if time_mode == "OFF":
        return L1
    else:
        time_array = Z["PEt"].values + Z["offset"].values
        firedPMT = genFired(Z)
        Ti = probe.callT(vertex, firedPMT)
        return L1 + LogLikelihood_Time(Ti, time_array)
    

