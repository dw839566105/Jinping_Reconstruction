#!/usr/bin/env python3
'''
重建所需文件读入
'''
import numpy as np
from DetectorConfig import tau, ts, wavel, E0, chnums, tbin
import statsmodels.api as sm

# sm.families.Poisson 只允许 log 的 link，其他 link 都会被视为 unsafe 弹出 warning
import warnings
warnings.filterwarnings("ignore")

def quantile(y, T_i):
    # less = T_i[y<T_i] - y[y<T_i]
    # more = y[y>=T_i] - T_i[y>=T_i]
    # R = (1-tau)*np.sum(less) + tau*np.sum(more)
    L = (T_i-y) * (y<T_i) * (1-tau) + (y-T_i) * (y>=T_i) * tau
    return L/ts

def LogLikelihood(vertex, zs, s0s, offsets, chs, probe, darkrate, timecalib):
    '''
    计算似然函数
    pe_array: 在当前抽样的 Z 下，各通道接收到光子数。长度为 chnums 的一维数组
    '''
    expect = probe.callPE(vertex)
    L1 = - np.sum((expect * vertex[3] / E0 + darkrate * wavel))
    Ti = probe.callT(chs) + vertex[-1]
    L2 = np.zeros(len(s0s))
    for i, s0 in enumerate(s0s):
        L2[i] = np.sum(np.log(callRt(zs[i][:s0] + offsets[i] + timecalib[chs[i]], Ti[i]) * expect[chs[i]] * vertex[3] / E0 + darkrate[chs[i]]))
    return L1 + L2.sum()
    
def callRt(t, t0):
    '''
    计算 Rt
    '''
    return tau * (1 - tau) / ts * np.exp(-quantile(t, t0))

def glm(vertex, zs, s0s, offsets, chs, probe, darkrate):
    '''
    对 E 做广义线性回归
    y ~ poisson(E * x + B)
    暗噪声来源于模拟，各通道一致: B = darkrate * tbin
    '''
    bound = np.arange(0, wavel + tbin, tbin)
    t = 0.5 * bound[:-1] + 0.5 * bound[1:]
    lamb = np.zeros((chnums, len(t)))
    N = np.zeros((chnums, len(t)))
    B = np.ones((chnums, len(t)))
    expect = probe.callPE(vertex)
    Ti = probe.callT(range(chnums)) + vertex[-1]
    for i in range(chnums):
        # 生成 x : lambda
        lamb[i,:] = expect[i] * callRt(t, Ti[i]) * tbin
        B[i,:] = darkrate[i] * tbin
        # 生成 y : N
        if i < len(chs):
            N[chs[i],:], _ = np.histogram(zs[i][:s0s[i]] + offsets[i], bins = bound)
    # glm 回归
    x = lamb.reshape(-1)
    y = N.reshape(-1)
    B = B.reshape(-1)
    poisson_model = sm.GLM(y, x, family = sm.families.Poisson(link = sm.families.links.identity()), offset = B).fit()
    return poisson_model.params
