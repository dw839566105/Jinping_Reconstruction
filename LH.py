#!/usr/bin/env python3
'''
重建所需文件读入
'''
import numpy as np

def LogLikelihood_PE(E, pe_array, expect):
    '''
    PE 项似然函数
    '''
    expect[expect == 0] = 1E-6
    expect1  = expect * E
    lnL = pe_array * np.log(expect1) - expect1
    return lnL.sum()

def LogLikelihood_Time(Ti, time_array):
    '''
    Time 项似然函数
    Ti: vertex0 下各触发 PMT 接收到光子的时刻，根据 probe 计算。一维数组
    '''
    lnL = LogLikelihood_quantile(time_array, Ti, 0.1, 3)
    return -lnL.sum()

def LogLikelihood_quantile(y, T_i, tau, ts):
    # less = T_i[y<T_i] - y[y<T_i]
    # more = y[y>=T_i] - T_i[y>=T_i]
    # R = (1-tau)*np.sum(less) + tau*np.sum(more)
    L = (T_i-y) * (y<T_i) * (1-tau) + (y-T_i) * (y>=T_i) * tau
    return L/ts

def LogLikelihood(vertex, pe_array, time_array, chs, probe, time_mode):
    '''
    计算似然函数
    pe_array: 在当前抽样的 Z 下，各通道接收到光子数。长度为 chnums 的一维数组
    time_array: 在当前抽样的 Z 下，各触发 PMT 接收到光子的时刻。一维数组
    '''
    expect = probe.callPE(vertex)
    L1 = LogLikelihood_PE(vertex[3], pe_array, expect)
    if time_mode == "OFF":
        return L1
    else:
        Ti = probe.callT(vertex, chs)
        return L1 + LogLikelihood_Time(Ti, time_array)
    

