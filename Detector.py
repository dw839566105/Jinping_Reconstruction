#!/usr/bin/env python3
'''
重建所需文件读入
Detector: jinping_1ton
'''
from DetectorConfig import *
from polynomial import *
import numpy as np
import tables

def Boundary(vertex):
    '''
    判断晃动是否超出边界
    探测器相关: Jinping_1ton
    '''
    return np.sum(vertex[:3] ** 2) <= np.square(0.97)

class Probe:
    '''
    多项式拟合 probe
    '''
    def __init__(self, coeff_pe, coeff_time, pmt_pos):
        self.coeff_pe = coeff_pe
        self.coeff_time = coeff_time
        self.pmt_pos = pmt_pos
        self.NPE = np.zeros(chnums)
        self.base_r = None
        self.base_t = None
        self.Ti = None
        self.probe_type = "poly"

    def Calc_basis(self, rho, cos_theta, len1, len2):
        base_t = legval(cos_theta, len1)
        base_r = legval(np.array([rho,]), len2).flatten()
        return base_r, base_t

    def getbase(self, vertex):
        # boundary
        v = vertex[:3]
        rho = np.linalg.norm(v)
        rho = np.clip(rho, 0, 1)
        # calculate cos theta
        cos_theta = np.cos(np.arctan2(np.linalg.norm(np.cross(v, self.pmt_pos), axis=1), np.dot(v,self.pmt_pos.T)))
        self.base_r, self.base_t = self.Calc_basis(rho, cos_theta, 
            np.max([self.coeff_pe.shape[0], self.coeff_time.shape[0]]),
            np.max([self.coeff_pe.shape[1], self.coeff_time.shape[1]]))

    def callPE(self, vertex):
        # 计算 NPE
        self.getbase(vertex)
        self.NPE = np.exp(self.base_t[:self.coeff_pe.shape[0]].T @ self.coeff_pe @ self.base_r[:self.coeff_pe.shape[1]])
        return self.NPE

    def callT(self, firedPMT):
        # 计算 PEt
        # 调用 callT 都在 callPE 之后，base_r 和 base_t 已被更新，不再重复计算
        self.Ti = self.base_t[:self.coeff_time.shape[0], firedPMT].T @ self.coeff_time @ self.base_r[:self.coeff_time.shape[1]]
        return self.Ti

def Init(zs, s0s, offsets, chs, pmt_pos):
    '''
    计算初值
    pe_array: 在当前抽样的 Z 下，各通道接收到光子数。长度为 chnums 的一维数组
    time_array: 在当前抽样的 Z 下，各触发 PMT 接收到光子的时刻。一维数组
    '''
    pe_array = np.zeros(chnums)
    time_array = np.zeros(np.sum(s0s))
    j = 0
    for i, s0 in enumerate(s0s):
        pe_array[chs[i]] = s0
        time_array[j : j + s0] = zs[i, :s0] + offsets[i]
        j += s0
    vertex = np.zeros(5)
    x_ini = 1.5 * np.sum(np.atleast_2d(pe_array).T*pmt_pos, axis=0) / np.sum(pe_array)
    E_ini = np.sum(pe_array) / npe # npe PE/MeV
    t_ini = np.quantile(time_array, 0.1) # quantile 0.1
    vertex[-1] = t_ini - T0
    vertex[3] = E_ini
    vertex[:3] = x_ini / shell
    if Boundary(vertex) == False:
        vertex[0] = 0
        vertex[1] = 0
        vertex[2] = 0
    return vertex

def LoadProbe(file0, file1, file2, pmt_pos):
    '''
    加载 probe
    file0: histo probe
    file1: polynomial probe pe part
    file2: polynomial probe time part
    '''
    with tables.open_file(file1, 'r') as h:
        coeff_pe = h.root.coeff[:]
        pe_type = h.root.coeff.attrs['type']

    with tables.open_file(file2,'r') as h:
        coeff_time = h.root.coeff[:]
        time_type = h.root.coeff.attrs['type']
    probe = Probe(coeff_pe, coeff_time, pmt_pos)
    return probe


