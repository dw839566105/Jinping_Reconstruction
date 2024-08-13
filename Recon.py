#!/usr/bin/env python3
'''
重建模拟数据的位置、能量
'''
import cupy as cp
import numpy as np
import h5py
import Detector
from config import *
from DetectorConfig import chnums, wavel
from fsmp_reader import Reader
import statsmodels.api as sm

# 重建数据储存类型
dtype = np.dtype([('EventID', '<i8'), ('step', '<i4'), ('x', '<f2'), ('y', '<f2'),
                  ('z', '<f2'), ('E', '<f2'), ('t', '<f2'), ('NPE', '<i4'),
                  ('Likelihood', '<f4'), ('acceptz', np.bool_), ('acceptr', np.bool_),
                  ('acceptE', np.bool_), ('acceptt', np.bool_)])

def LH(vertex, PEt, s0s, probe, darkrate):
    '''
    计算 Likelihood (全通道)
    PEt: cp.ndarray
        提前转化成 cupy.ndarray, 因为是 tvE 三者采样共用
    '''
    R, Rsum = probe.genR(vertex, PEt)
    index = cp.arange(R.shape[2])[None, None, :] < s0s[:, :, None]
    L1 = cp.sum(cp.log(R + darkrate[None, :, None]) * index, axis=(1,2))
    L2 = - cp.sum(Rsum, axis=1) # 省略了暗噪声积分的常数项
    return L1 + L2

def LHch(vertex, chs, PEt, s0s, probe, darkrate):
    '''
    计算 Likelihood (单通道)
    '''
    R, Rsum = probe.genRch(cp.asarray(vertex), cp.asarray(PEt), chs)
    index = cp.arange(R.shape[1])[None, :] < cp.asarray(s0s[:, None])
    L1 = cp.sum(cp.log(R + darkrate[chs][:, None]) * index, axis=1)
    return L1 + Rsum

def glm(data):
    length = len(data) // 3
    result = sm.GLM(data[length: 2 * length], data[:length], family=sm.families.Poisson(link=sm.families.links.identity()), offset=data[2 * length:]).fit().params[0]
    return result

def histogram(data, weights, bins):
    result = np.zeros((data.shape[0], data.shape[1], bins.shape[0] - 1))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            result[i, j], _ = np.histogram(data[i, j][weights[i,j]], bins=bins)
    return result

def Regression(vertex, offsets, zs, s0s, probe, darkrate, timecalib):
    '''
    回归能量
    '''
    index = np.arange(zs.shape[2])[None, None, :] < s0s[:, :, None]
    PEt = zs + offsets[:, :, None] + timecalib[None, :, None]
    bound = np.arange(0, wavel + tbin, tbin)
    tx = np.arange(tbin / 2, wavel + tbin / 2, tbin)
    tx = np.broadcast_to(tx, (zs.shape[0], zs.shape[1], tx.shape[0]))
    X = probe.genR(vertex, tx, False) * tbin
    Y = histogram(PEt, index, bound)
    darkrate_3d = np.expand_dims(darkrate, axis=(0, 2))
    B = np.broadcast_to(darkrate_3d, (X.shape)) * tbin
    data = np.column_stack((X.reshape(X.shape[0], -1), Y.reshape(Y.shape[0], -1), B.reshape(B.shape[0], -1)))
    E = np.apply_along_axis(glm, 1, data)
    return E

def Reconstruction(fsmp, sparsify, num, probe, pmt_pos, darkrate, timecalib, MC_step):
    '''
    reconstruction
    '''
    # 读入波形分析结果
    waveform = Reader(sparsify, fsmp, num, MC_step)
    # 预分配储存数组
    recon_step = np.zeros((len(waveform), MC_step), dtype=dtype)
    i = 0
    for eids, zs, meta_zs, samplers in waveform:
        zs = cp.asarray(zs, dtype=cp.float16)
        s0s = cp.asarray(meta_zs["s0"], dtype=cp.int32)
        nu_lcs = cp.asarray(meta_zs["nu_lc"], dtype=cp.float16)

        # 设定随机数
        cp.random.seed(eids[0] % 1000000) # 取第一个事例编号设定随机数种子
        u_gibbs = cp.log(cp.random.uniform(0, 1, (MC_step, len(eids), gibbs_variables)), dtype=cp.float32)
        u_V = cp.random.normal(0, 1, (MC_step, len(eids), V_variables), dtype=cp.float32)

        # 给出 vertex, LogLikelihhood 的初值并记录
        PEt = zs + timecalib[None, :, None]
        vertex0 = Detector.Init(PEt, s0s, pmt_pos)
        Likelihood_vertex0 = LH(vertex0, PEt, s0s, probe, darkrate)

        # gibbs iteration
        for step in range(MC_step):
            # 对 z 采样
            ch_s, z_s, meta_zc = next(samplers)
            ch_s = cp.asarray(ch_s, dtype=cp.int16)
            z_s = cp.asarray(z_s, dtype=cp.float16)
            s0_s = cp.asarray(meta_zc["s0"], dtype=cp.int16)
            nu_lc_s = cp.asarray(meta_zc["nu_lc"], dtype=cp.float16)
            ievch = cp.s_[cp.arange(len(zs)), ch_s]
            PEt_o = PEt[ievch]
            s0_o = s0s[ievch]
            nu_lc_o = nu_lcs[ievch]

            PEt_s = z_s + timecalib[ch_s][:, None]
            LH_z_sample = LHch(vertex0, ch_s, PEt_s, s0_s, probe, darkrate)
            LH_z_origin = LHch(vertex0, ch_s, PEt_o, s0_o, probe, darkrate)
            accept_z = (nu_lc_o - nu_lc_s + LH_z_sample - LH_z_origin) > u_gibbs[step, :, 0]
            # update z
            ievch_update = cp.s_[cp.arange(s0s.shape[0])[accept_z], ch_s[accept_z]]
            s0s[ievch_update] = s0_s[accept_z]
            nu_lcs[ievch_update] = nu_lc_s[accept_z]
            PEt[ievch_update] = PEt_s[accept_z]
            # update Likelihood
            Likelihood_vertex0[accept_z] = LH(vertex0[accept_z], PEt[accept_z], s0s[accept_z], probe, darkrate)

            # 对 r 采样
            vertex1 = vertex0.copy()
            vertex1[:, :3] = vertex1[:, :3] + r_sigma * u_V[step, :, :3]
            accept_rB = Detector.Boundary(vertex1)
            Likelihood_vertex1 = LH(vertex1, PEt, s0s, probe, darkrate)
            accept_rL = Likelihood_vertex1 - Likelihood_vertex0 > u_gibbs[step, :, 1]
            accept_r = accept_rL & accept_rB
            # update r
            vertex0[accept_r] = vertex1[accept_r]
            Likelihood_vertex0[accept_r] = Likelihood_vertex1[accept_r]

            # 对 E 采样
            vertex1 = vertex0.copy()
            vertex1[:, 3] = vertex1[:, 3] + E_sigma * u_V[step, :, 3]
            Likelihood_vertex1 = LH(vertex1, PEt, s0s, probe, darkrate)
            accept_E = Likelihood_vertex1 - Likelihood_vertex0 > u_gibbs[step, :, 2]
            # update E
            vertex0[accept_E] = vertex1[accept_E]
            Likelihood_vertex0[accept_E] = Likelihood_vertex1[accept_E]

            # 对 t 采样
            vertex1 = vertex0.copy()
            vertex1[:, 4] = vertex1[:, 4] + T_sigma * u_V[step, :, 4]
            Likelihood_vertex1 = LH(vertex1, PEt, s0s, probe, darkrate)
            accept_t = Likelihood_vertex1 - Likelihood_vertex0 > u_gibbs[step, :, 3]
            # update t
            vertex0[accept_t] = vertex1[accept_t]
            Likelihood_vertex0[accept_t] = Likelihood_vertex1[accept_t]

            # write into tables
            block_size = np.s_[i : i + len(eids), step]
            recon_step[block_size]['EventID'] = eids
            recon_step[block_size]['step'] = step
            recon_step[block_size]['x'] = vertex0[:, 0].get()
            recon_step[block_size]['y'] = vertex0[:, 1].get()
            recon_step[block_size]['z'] = vertex0[:, 2].get()
            recon_step[block_size]['E'] = vertex0[:, 3].get()
            recon_step[block_size]['t'] = vertex0[:, 4].get()
            recon_step[block_size]['NPE'] = cp.sum(s0s, axis=1).get()
            recon_step[block_size]['Likelihood'] = Likelihood_vertex0.get()
            recon_step[block_size]['acceptz'] = accept_z.get()
            recon_step[block_size]['acceptr'] = accept_r.get()
            recon_step[block_size]['acceptE'] = accept_E.get()
            recon_step[block_size]['acceptt'] = accept_t.get()
            i += len(eids)
    return recon_step



