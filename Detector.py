#!/usr/bin/env python3
'''
重建所需文件读入
Detector: jinping_1ton
'''
from DetectorConfig import *
from numba import njit
import numpy as np
import h5py

def Boundary(vertex):
    '''
    判断晃动是否超出边界
    探测器相关: Jinping_1ton
    '''
    return np.sum(np.square(vertex[:, :3]), axis=1) <= 1

@njit
def legval(x, n):
    res = np.zeros((n,) + x.shape)
    res[0] = 1
    res[1] = x
    for i in range(2, n):
        res[i] = ((2 * i - 1) * x * res[i - 1] - (i - 1) * res[i - 2]) / i
    return res

class Probe:
    '''
    多项式拟合 probe
    '''
    def __init__(self, coeff_pe, coeff_time, pmt_pos):
        '''
        coeff_pe: PE 项系数
        coeff_time: Timing 项系数
        pmt_pos: PMT 直角坐标位置
        '''
        self.coeff_pe = coeff_pe
        self.coeff_time = coeff_time
        self.pmt_pos = pmt_pos

    def genBase(self, vertex):
        '''
        生成多项式基底 (全通道)
        '''
        # boundary
        v = vertex[:, :3]
        rho = np.linalg.norm(v, axis=1)
        rho = np.clip(rho, 0, 1)
        # calculate cos theta
        cos_theta = np.cos(np.arctan2(np.linalg.norm(np.cross(v[:, None, :], self.pmt_pos), axis=-1), np.dot(v, self.pmt_pos.T)))
        base_t = legval(cos_theta, np.max([self.coeff_pe.shape[0], self.coeff_time.shape[0]]))
        base_r = legval(rho, np.max([self.coeff_pe.shape[1], self.coeff_time.shape[1]]))
        return base_t, base_r

    def genR(self, vertex, PEt, sum_mode = True):
        '''
        调用 probe (全通道)
        vertex: 顶点的位置能量时刻
        PEt: 波形分析得到的 PEt, 通过时间刻度修正
        sum_mode: 广义线性回归需要单位能量的 R, 且不需要 R 的积分，加开关以减少计算
        '''
        base_t, base_r = self.genBase(vertex)
        # 计算空间项
        NPE = np.exp(np.sum(np.matmul(base_t[:self.coeff_pe.shape[0]].T, self.coeff_pe) * base_r[:self.coeff_pe.shape[1]].T[None, :, :], axis=2))
        # 计算 R
        Ti = np.sum(np.matmul(base_t[:self.coeff_time.shape[0]].T, self.coeff_time) * base_r[:self.coeff_time.shape[1]].T[None, :, :], axis=2)
        t = PEt - Ti.T[:,:, None] - vertex[:, -1][:, None, None]
        R = tau * (1 - tau) / ts * np.exp(- np.where(t < 0, t * (tau - 1), t * tau) / ts) * NPE.T[:, :, None] / E0
        if not sum_mode:
            # 返回单位能量 R
            return R
        R *= vertex[:, 3][:, None, None]
        # 分类计算 R 的积分
        down = - Ti.T - vertex[:, -1][:, None]
        Rsum = np.zeros_like(down)
        index1 = down > 0
        Rsum[index1] = (tau - 1) * (np.exp(- tau * (down[index1] + wavel) / ts) - np.exp(- tau * down[index1] / ts))
        index2 = down < - wavel
        Rsum[index2] = tau * (np.exp((1- tau) * (down[index2] + wavel) / ts) - np.exp((1- tau) * down[index2] / ts))
        index3 = ~(index1 | index2)
        Rsum[index3] = 1 + (tau - 1) * np.exp(- tau * (down[index3] + wavel) / ts) - tau * np.exp((1- tau) * down[index3] / ts)
        return R, Rsum * NPE.T * vertex[:, 3][:, None] / E0

    def genBasech(self, vertex, chs):
        '''
        生成多项式基底 (单通道)
        '''
        # boundary
        v = vertex[:, :3]
        rho = np.linalg.norm(v, axis=1)
        rho = np.clip(rho, 0, 1)
        # calculate cos theta
        p = self.pmt_pos[chs]
        cos_theta = np.cos(np.arctan2(np.linalg.norm(np.cross(v, p), axis=-1), np.sum(v * p, axis=1)))
        base_t = legval(cos_theta, np.max([self.coeff_pe.shape[0], self.coeff_time.shape[0]]))
        base_r = legval(rho, np.max([self.coeff_pe.shape[1], self.coeff_time.shape[1]]))
        return base_t, base_r    

    def genRch(self, vertex, PEt, chs):
        '''
        调用 probe (单通道)
        对 z 采样只涉及单通道，输入变量相比 genR 少一个通道编号的维度
        chs: 各事例被抽取的通道编号
        '''
        base_t, base_r = self.genBasech(vertex, chs)
        # 计算空间项
        NPE = np.exp(np.sum((base_t[:self.coeff_pe.shape[0]].T @ self.coeff_pe) * base_r[:self.coeff_pe.shape[1]].T, axis=1))
        # 计算 R
        Ti = np.sum((base_t[:self.coeff_time.shape[0]].T @ self.coeff_time) * base_r[:self.coeff_time.shape[1]].T, axis=1)
        t = PEt - Ti.T[:, None] - vertex[:, -1][:, None]
        R = tau * (1 - tau) / ts * np.exp(- np.where(t < 0, t * (tau - 1), t * tau) / ts) * NPE.T[:, None] * vertex[:, 3][:, None] / E0
        # 分类计算 R 的积分
        down = - Ti - vertex[:, -1]
        Rsum = np.zeros_like(down)
        index1 = down > 0
        Rsum[index1] = (tau - 1) * (np.exp(- tau * (down[index1] + wavel) / ts) - np.exp(- tau * down[index1] / ts))
        index2 = down < - wavel
        Rsum[index2] = tau * (np.exp((1- tau) * (down[index2] + wavel) / ts) - np.exp((1- tau) * down[index2] / ts))
        index3 = ~(index1 | index2)
        Rsum[index3] = 1 + (tau - 1) * np.exp(- tau * (down[index3] + wavel) / ts) - tau * np.exp((1- tau) * down[index3] / ts)
        return R, Rsum * NPE * vertex[:, 3] / E0

def quantile_regression(arr):
    return np.quantile(arr[arr != 0], tau)

def Init(zs, s0s, offsets, pmt_pos, timecalib):
    '''
    计算初值
    '''
    vertex = np.zeros((s0s.shape[0], 5))
    s0sum = np.sum(s0s, axis=1)
    vertex[:, 3] = s0sum/ npe
    vertex[:, :3] = 1.5 * s0s @ pmt_pos / s0sum[:, None]
    PEt = zs.copy()
    index = np.arange(zs.shape[2])[None, None, :] < s0s[:, :, None]
    PEt = np.where(index, zs + offsets[:, :, None] + timecalib[None, :, None], 0)
    vertex[:, -1] = np.apply_along_axis(quantile_regression, axis=1, arr=PEt.reshape(PEt.shape[0], -1))
    return vertex

def LoadProbe(PEFile, TimeFile, PmtPos):
    '''
    加载 probe
    PEFile: polynomial probe pe part
    TimeFile: polynomial probe time part
    '''
    with h5py.File(PEFile, 'r') as h:
        PECoeff = h['coeff'][:]
    with h5py.File(TimeFile, 'r') as h:
        TimeCoeff = h['coeff'][:]
    probe = Probe(PECoeff, TimeCoeff, PmtPos)
    return probe
