#!/usr/bin/env python3
'''
重建所需文件读入
'''
import numpy as np
from DetectorConfig import tau, ts, dark, wavel, E0
import statsmodels.api as sm

# sm.families.Poisson 只允许 log 的 link，其他 link 都会被视为 unsafe 弹出 warning
import warnings
warnings.filterwarnings("ignore")

def quantile(y, T_i, tau, ts):
    # less = T_i[y<T_i] - y[y<T_i]
    # more = y[y>=T_i] - T_i[y>=T_i]
    # R = (1-tau)*np.sum(less) + tau*np.sum(more)
    L = (T_i-y) * (y<T_i) * (1-tau) + (y-T_i) * (y>=T_i) * tau
    return L/ts

def LogLikelihood(vertex, pe_array, zs, s0s, offsets, chs, probe):
    '''
    计算似然函数
    pe_array: 在当前抽样的 Z 下，各通道接收到光子数。长度为 chnums 的一维数组
    '''
    expect = probe.callPE(vertex)
    L1 = - np.sum((expect * vertex[3] / E0 + dark * wavel))
    Ti = probe.callT(chs) + vertex[-1]
    L2 = np.zeros(len(s0s))
    for i, s0 in enumerate(s0s):
        L2[i] = np.sum(np.log(callRt(zs[i][:s0] + offsets[i], Ti[i], tau, ts) * expect[chs[i]] * vertex[3] / E0 + dark))
    return L1 + L2.sum()
    
def callRt(t, t0, tau, ts):
    '''
    计算 Rt
    '''
    return tau * (1 - tau) / ts * np.exp(-quantile(t, t0, tau, ts))

def glm(x, y):
    '''
    对 E 做广义线性回归
    y ~ poisson(E * x + B)
    暗噪声来源于模拟，各通道一致
    '''
    B = np.ones_like(x) * dark * wavel
    # glm 回归
    poisson_model = sm.GLM(y, x, family=sm.families.Poisson(link=sm.families.links.identity()), offset=B).fit()
    return poisson_model.params
