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
from tqdm import tqdm
from fsmp_reader import FSMPreader
import statsmodels.api as sm
# sm.families.Poisson 只允许 log 的 link，其他 link 都会被视为 unsafe 弹出 warning
import warnings
warnings.filterwarnings("ignore")

# 重建数据储存类型
dtype = cp.dtype([('EventID', '<i8'), ('step', '<i4'), ('x', '<f2'), ('y', '<f2'),
                  ('z', '<f2'), ('E', '<f2'), ('t', '<f2'), ('NPE', '<i4'),
                  ('Likelihood', '<f4'), ('acceptz', '<i4'), ('acceptr', '<i4'),
                  ('acceptE', '<i4'), ('acceptt', '<i4')])

def LH(vertex, chs, offsets, zs, s0s, probe, darkrate, timecalib):
    '''
    计算 Likelihood (全通道)
    '''
    PEt = zs + offsets[:, :, None] + timecalib[None, :, None]
    R, Rsum = probe.genR(vertex, PEt)
    index = cp.arange(R.shape[2])[None, None, :] < cp.asarray(s0s[:, :, None])
    L1 = cp.sum(cp.log(R + darkrate[chs][:, :, None]) * index, axis=(1,2))
    index = s0s > 0
    L2 = - cp.sum(Rsum * index, axis=1)
    return L1 + L2

def LHch(vertex, chs, zs, s0s, offsets, probe, darkrate, timecalib):
    '''
    计算 Likelihood (单通道)
    '''
    PEt = zs + offsets[:, None] + timecalib[chs][:, None]
    R, Rsum = probe.genRch(vertex, PEt, chs)
    index = cp.arange(R.shape[1])[None, :] < cp.asarray(s0s[:, None])
    L1 = cp.sum(cp.log(R + darkrate[chs][:, None]) * index, axis=1)
    return L1 + Rsum

def glm(data):
    length = len(data) // 3
    result = sm.GLM(data[length: 2 * length], data[:length], family=sm.families.Poisson(link=sm.families.links.identity()), offset=data[2 * length:]).fit().params[0]
    return result

def histogram(data, weights, bins):
    result = cp.zeros((data.shape[0], data.shape[1], bins.shape[0] - 1))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            result[i, j], _ = cp.histogram(data[i, j][weights[i,j]], bins=bins)
    return result

def Regression(vertex, offsets, zs, s0s, probe, darkrate, timecalib):
    '''
    回归能量
    '''
    index = cp.arange(zs.shape[2])[None, None, :] < s0s[:, :, None]
    PEt = zs + offsets[:, :, None] + timecalib[None, :, None]
    bound = cp.arange(0, wavel + tbin, tbin)
    tx = cp.arange(tbin / 2, wavel + tbin / 2, tbin)
    tx = cp.broadcast_to(tx, (zs.shape[0], zs.shape[1], tx.shape[0]))
    X = probe.genR(vertex, tx, False) * tbin
    Y = histogram(PEt, index, bound)
    darkrate_3d = cp.expand_dims(darkrate, axis=(0, 2))
    B = cp.broadcast_to(darkrate_3d, (X.shape)) * tbin
    data = cp.column_stack((X.reshape(X.shape[0], -1), Y.reshape(Y.shape[0], -1), B.reshape(B.shape[0], -1)))
    E = cp.apply_along_axis(glm, 1, data)
    return E

def concat(iterator, Entries):
    '''
    将多个事例数组拼接
    eids.shape = (N, 1)
    chs.shape = (N, chnums)
    offsets.shape = (N, chnums)
    zs.shape = (N, chnums, y)
    s0s.shape = (N, chnums)
    nu_lcs.shape = (N, chnums)
    '''
    for iter, (eid, ch, offset, z, s0, nu_lc, _, sampler) in enumerate(iterator):
        i = iter % Entries
        # 取数后重置
        if i == 0:
            eids = cp.zeros(Entries, dtype=cp.int64)
            chs = cp.zeros((Entries, chnums), dtype=cp.int32)
            offsets = cp.zeros((Entries, chnums), dtype=cp.float32)
            zs = cp.zeros((Entries, chnums, 10), dtype=cp.float32)
            s0s = cp.zeros((Entries, chnums), dtype=cp.int32)
            nu_lcs = cp.zeros((Entries, chnums), dtype=cp.float32)
            samplers = []
        eids[i] = eid
        chs[i, :len(ch)] = ch
        offsets[i, ch] = offset
        s0s[i, ch] = s0
        nu_lcs[i, ch] = nu_lc
        # NPE 扩展
        if z.shape[1] > zs.shape[2]:
            supply = cp.zeros((Entries, zs.shape[1], z.shape[1] - zs.shape[2]), dtype=cp.float32)
            zs = cp.concatenate((zs, supply), axis=2)
        zs[i, ch, :z.shape[1]] = z
        samplers.append(sampler)
        # 取数
        if i == Entries - 1:
            yield eids, chs, offsets, zs, s0s, nu_lcs, samplers
    # 不足 Entries 个事例取完
    yield eids[:i], chs[:i], offsets[:i], zs[:i], s0s[:i], nu_lcs[:i], samplers[:i]
                    
def resampleZ(iterators, events, maxs):
    '''
    返回 z 的重采样结果
    ich.shape = (N, 1)
    z_extend.shape = (N, 1, y)
    s0.shape = (N, 1)
    nu_lc.shape = (N, 1)
    '''
    ich = cp.zeros((events), dtype=cp.int32)
    z_extend = cp.zeros((events, maxs), dtype=cp.float32)
    s0 = cp.zeros((events), dtype=cp.int32)
    nu_lc = cp.zeros((events), dtype=cp.float32)
    for i, iterator in enumerate(iterators):
        ich[i], z, s0[i], nu_lc[i] = next(iterator)
        z = cp.array(z)
        # NPE 扩展
        if s0[i] > z_extend.shape[1]:
            supply = cp.zeros((events, s0[i] - z_extend.shape[1]), dtype=cp.float32)
            z_extend = cp.concatenate((z_extend, supply), axis=1)
        z_extend[i, :s0[i]] = z[:s0[i]]
    return ich, z_extend, s0, nu_lc

def Reconstruction(fsmp, sparsify, Entries, output, probe, pmt_pos, darkrate, timecalib, MC_step):
    '''
    reconstruction
    '''
    # 创建输出文件
    opts = {"compression": "gzip", "shuffle": True}
    with h5py.File(output, "w") as opt:
        dataset = opt.create_dataset("Recon", shape=(0,), maxshape=(None,), dtype=dtype, **opts)

        for eids, chs, offsets, zs, s0s, nu_lcs, samplers in tqdm(concat(FSMPreader(sparsify, fsmp).rand_iter(MC_step), Entries)):
            # 预分配储存数组
            recon_step = np.zeros(len(eids), dtype=dtype)

            # 设定随机数
            cp.random.seed(eids[0].get() % 1000000) # 取第一个事例编号设定随机数种子
            u_gibbs = cp.log(cp.random.uniform(0, 1, (MC_step, len(eids), gibbs_variables)))
            u_V = cp.random.normal(0, 1, (MC_step, len(eids), V_variables))

            # 给出 vertex, LogLikelihhood 的初值并记录
            vertex0 = Detector.Init(zs, s0s, offsets, pmt_pos, timecalib)
            Likelihood_vertex0 = LH(vertex0, chs, offsets, zs, s0s, probe, darkrate, timecalib)

            # gibbs iteration
            for step in range(MC_step):
                # 对 z 采样
                ich_s, z_s, s0_s, nu_lc_s = resampleZ(samplers, len(eids), zs.shape[2])
                ch_s = chs[cp.arange(chs.shape[0]), ich_s]
                z_o = zs[cp.arange(zs.shape[0]), ch_s]
                s0_o = s0s[cp.arange(s0s.shape[0]), ch_s]
                nu_lc_o = nu_lcs[cp.arange(nu_lcs.shape[0]), ch_s]
                offset_s = offsets[cp.arange(offsets.shape[0]), ch_s]
                LH_z_sample = LHch(vertex0, ch_s, z_s, s0_s, offset_s, probe, darkrate, timecalib)
                LH_z_origin = LHch(vertex0, ch_s, z_o, s0_o, offset_s, probe, darkrate, timecalib)
                accept_z = (nu_lc_o - nu_lc_s + LH_z_sample - LH_z_origin) > u_gibbs[step, :, 0]
                # update z
                s0s[cp.arange(s0s.shape[0])[accept_z], ch_s[accept_z]] = s0_s[accept_z]
                nu_lcs[cp.arange(nu_lcs.shape[0])[accept_z], ch_s[accept_z]] = nu_lc_s[accept_z]
                if z_s.shape[1] > z_o.shape[1]:
                    supply = cp.zeros((zs.shape[0], zs.shape[1], z_s.shape[1] - z_o.shape[1]), dtype=cp.float32)
                    zs = cp.concatenate((zs, supply), axis=2)
                zs[cp.arange(zs.shape[0])[accept_z], ch_s[accept_z]] = z_s[accept_z]
                # update Likelihood
                Likelihood_vertex0[accept_z] = LH(vertex0[accept_z], chs[accept_z], offsets[accept_z], zs[accept_z], s0s[accept_z], probe, darkrate, timecalib)

                # 对 r 采样
                vertex1 = vertex0.copy()
                vertex1[:, :3] = vertex1[:, :3] + r_sigma * u_V[step, :, :3]
                accept_rB = Detector.Boundary(vertex1)
                Likelihood_vertex1 = LH(vertex1, chs, offsets, zs, s0s, probe, darkrate, timecalib)
                accept_rL = Likelihood_vertex1 - Likelihood_vertex0 > u_gibbs[step, :, 1]
                accept_r = accept_rL & accept_rB
                # update r
                vertex0[accept_r] = vertex1[accept_r]
                Likelihood_vertex0[accept_r] = Likelihood_vertex1[accept_r]

                # 对 E 采样
                vertex1 = vertex0.copy()
                vertex1[:, 3] = vertex1[:, 3] + E_sigma * u_V[step, :, 3]
                Likelihood_vertex1 = LH(vertex1, chs, offsets, zs, s0s, probe, darkrate, timecalib)
                accept_E = Likelihood_vertex1 - Likelihood_vertex0 > u_gibbs[step, :, 2]
                # update E
                vertex0[accept_E] = vertex1[accept_E]
                Likelihood_vertex0[accept_E] = Likelihood_vertex1[accept_E]

                # 对 t 采样
                vertex1 = vertex0.copy()
                vertex1[:, 4] = vertex1[:, 4] + T_sigma * u_V[step, :, 4]
                Likelihood_vertex1 = LH(vertex1, chs, offsets, zs, s0s, probe, darkrate, timecalib)
                accept_t = Likelihood_vertex1 - Likelihood_vertex0 > u_gibbs[step, :, 3]
                # update t
                vertex0[accept_t] = vertex1[accept_t]
                Likelihood_vertex0[accept_t] = Likelihood_vertex1[accept_t]

                # write into tables
                recon_step['EventID'] = cp.asnumpy(eids)
                recon_step['step'] = step
                recon_step['x'] = cp.asnumpy(vertex0[:, 0])
                recon_step['y'] = cp.asnumpy(vertex0[:, 1])
                recon_step['z'] = cp.asnumpy(vertex0[:, 2])
                recon_step['E'] = cp.asnumpy(vertex0[:, 3])
                recon_step['t'] = cp.asnumpy(vertex0[:, 4])
                recon_step['NPE'] = cp.asnumpy(cp.sum(s0s, axis=1))
                recon_step['Likelihood'] = cp.asnumpy(Likelihood_vertex0)
                recon_step['acceptz'] = cp.asnumpy(accept_z)
                recon_step['acceptr'] = cp.asnumpy(accept_r)
                recon_step['acceptE'] = cp.asnumpy(accept_E)
                recon_step['acceptt'] = cp.asnumpy(accept_t)
                current_size = dataset.shape[0]
                new_size = current_size + len(eids)
                dataset.resize((new_size,))
                dataset[current_size:new_size] = recon_step
