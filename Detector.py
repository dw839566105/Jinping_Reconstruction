#!/usr/bin/env python3
'''
重建所需文件读入
Detector: jinping_1ton
'''
from DetectorConfig import *
import cupy as cp
import h5py
import numpy as np
from cupyx.scipy.special import lpmv

def Boundary(vertex):
    '''
    判断晃动是否超出边界
    探测器相关: Jinping_1ton
    '''
    return cp.linalg.norm(vertex[:, :3], axis=1) <= 1


int_rsum = cp.ElementwiseKernel(
    'T down',
    'T Rsum',
    f"""
        if (down > 0)
            Rsum = ({tau}-1)*(exp(-{tau}*(down+{wavel})/{ts}) - exp(-{tau}*down/{ts}));
        else if (down < -{wavel})
            Rsum = {tau}*(exp((1-{tau})*(down+{wavel})/{ts}) - exp((1-{tau})*down/{ts}));
        else
            Rsum = 1 + ({tau}-1)*exp(-{tau}*(down+{wavel})/{ts}) - {tau}*exp((1-{tau})*down/{ts});
    """,
    'int_rsum'
)


class Probe:
    '''
    多项式拟合 probe
    '''
    def __init__(self, coeff_pe, coeff_time, pmt_pos):
        '''
        初始化 probe
        coeff_pe: cp.ndarray
            PE 项系数
        coeff_time: cp.ndarray
            Timing 项系数
        pmt_pos: cp.ndarray
            PMT 直角坐标位置
        ordert:
            时间项阶数
        orderr:
            空间项阶数
        '''
        self.coeff_pe = cp.asarray(coeff_pe)
        self.coeff_time = cp.asarray(coeff_time)
        self.pmt_pos = pmt_pos
        self.ordert = max(self.coeff_pe.shape[0], self.coeff_time.shape[0])
        self.orderr = max(self.coeff_pe.shape[1], self.coeff_time.shape[1])

    def genBase(self, vertex):
        '''
        生成多项式基底 (全通道)
        '''
        # boundary
        v = vertex[:, :3]
        rho = cp.linalg.norm(v, axis=1)
        rho = cp.clip(rho, 0, 1)
        # calculate cos theta
        cos_theta = cp.cos(cp.arctan2(cp.linalg.norm(cp.cross(v[:, None, :], self.pmt_pos), axis=-1), cp.dot(v, self.pmt_pos.T)))
        base_t = lpmv(cp.zeros(self.ordert, dtype=cp.float16)[:, None, None], cp.arange(self.ordert, dtype=cp.int16)[:, None, None], cos_theta[None, :, :])
        base_r = lpmv(cp.zeros(self.orderr, dtype=cp.float16)[:, None], cp.arange(self.orderr, dtype=cp.int16)[:, None], rho[None, :])
        return base_t, base_r

    def genR(self, vertex, PEt, sum_mode = True):
        '''
        调用 probe (全通道)
        vertex: cp.ndarray
            顶点的位置能量时刻
        PEt: cp.ndarray
            波形分析得到的 PEt, 通过时间刻度修正
        sum_mode: 
            广义线性回归需要单位能量的 R, 且不需要 R 的积分，加开关以减少计算
        '''
        base_t, base_r = self.genBase(vertex)
        # 计算空间项
        NPE = cp.exp(cp.sum(cp.matmul(base_t[:self.coeff_pe.shape[0]].T, self.coeff_pe) * base_r[:self.coeff_pe.shape[1]].T[None, :, :], axis=2))
        # 计算 R
        Ti = cp.sum(cp.matmul(base_t[:self.coeff_time.shape[0]].T, self.coeff_time) * base_r[:self.coeff_time.shape[1]].T[None, :, :], axis=2)
        t = PEt - Ti.T[:,:, None] - vertex[:, -1][:, None, None]
        R = tau * (1 - tau) / ts * cp.exp(- cp.where(t < 0, t * (tau - 1), t * tau) / ts) * NPE.T[:, :, None] / E0
        if not sum_mode:
            # 返回单位能量 R
            return R
        R *= vertex[:, 3][:, None, None]
        # 分类计算 R 的积分
        down = - Ti.T - vertex[:, -1][:, None]
        Rsum = int_rsum(down)
        return R, Rsum * NPE.T * vertex[:, 3][:, None] / E0

    def genBasech(self, vertex, chs):
        '''
        生成多项式基底 (单通道)
        '''
        # boundary
        v = vertex[:, :3]
        rho = cp.linalg.norm(v, axis=1)
        rho = cp.clip(rho, 0, 1)
        # calculate cos theta
        p = self.pmt_pos[chs]
        cos_theta = cp.cos(cp.arctan2(cp.linalg.norm(cp.cross(v, p), axis=-1), cp.sum(v * p, axis=1)))
        base_t = lpmv(cp.zeros(self.ordert, dtype=cp.float16)[:, None], cp.arange(self.ordert, dtype=cp.int16)[:, None], cos_theta[None, :])
        base_r = lpmv(cp.zeros(self.orderr, dtype=cp.float16)[:, None], cp.arange(self.orderr, dtype=cp.int16)[:, None], rho[None, :])
        return base_t, base_r

    def genRch(self, vertex, PEt, chs):
        '''
        调用 probe (单通道)
        对 z 采样只涉及单通道，输入变量相比 genR 少一个通道编号的维度
        chs: 各事例被抽取的通道编号
        '''
        base_t, base_r = self.genBasech(vertex, chs)
        # 计算空间项
        NPE = cp.exp(cp.sum((base_t[:self.coeff_pe.shape[0]].T @ self.coeff_pe) * base_r[:self.coeff_pe.shape[1]].T, axis=1))
        # 计算 R
        Ti = cp.sum((base_t[:self.coeff_time.shape[0]].T @ self.coeff_time) * base_r[:self.coeff_time.shape[1]].T, axis=1)
        t = PEt - Ti.T[:, None] - vertex[:, -1][:, None]
        R = tau * (1 - tau) / ts * cp.exp(- cp.where(t < 0, t * (tau - 1), t * tau) / ts) * NPE.T[:, None] * vertex[:, 3][:, None] / E0
        # 分类计算 R 的积分
        down = - Ti - vertex[:, -1]
        Rsum = int_rsum(down)
        return R, Rsum * NPE * vertex[:, 3] / E0

def quantile_regression(arr):
    return cp.quantile(arr[arr != 0], tau)

def Init(PEt, s0s, pmt_pos):
    '''
    计算初值
    '''
    vertex = cp.zeros((s0s.shape[0], 5), dtype=cp.float16)
    s0sum = cp.sum(s0s, axis=1)
    vertex[:, 3] = s0sum / npe
    vertex[:, :3] = 1.5 * s0s @ pmt_pos / s0sum[:, None]
    index = cp.arange(PEt.shape[2])[None, None, :] < s0s[:, :, None]
    PEt = cp.where(index, PEt, 0)
    vertex[:, -1] = cp.apply_along_axis(quantile_regression, axis=1, arr=PEt.reshape(PEt.shape[0], -1))
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
