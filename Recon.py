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
dtype = np.dtype([('EventID', '<i8'), ('step', '<i4'), ('x', '<f2'), ('y', '<f2'),
                  ('z', '<f2'), ('E', '<f2'), ('t', '<f2'), ('NPE', '<i4'),
                  ('Likelihood', '<f4'), ('acceptz', '<i4'), ('acceptr', '<i4'),
                  ('acceptE', '<i4'), ('acceptt', '<i4')])

def LH(vertex, chs, offsets, zs, s0s, probe, darkrate, timecalib):
    '''
    计算 Likelihood (全通道)
    '''
    PEt = zs + offsets[:, :, None] + timecalib[None, :, None]
    R, Rsum = probe.genR(cp.asarray(vertex), cp.asarray(PEt))
    index = cp.arange(R.shape[2])[None, None, :] < cp.asarray(s0s[:, :, None])
    L1 = cp.sum(cp.log(R + darkrate[chs][:, :, None]) * index, axis=(1,2))
    L2 = - cp.sum(Rsum, axis=1) # 省略了暗噪声积分的常数项
    return cp.asnumpy(L1 + L2)

def LHch(vertex, chs, zs, s0s, offsets, probe, darkrate, timecalib):
    '''
    计算 Likelihood (单通道)
    '''
    PEt = zs + offsets[:, None] + timecalib[chs][:, None]
    R, Rsum = probe.genRch(cp.asarray(vertex), cp.asarray(PEt), chs)
    index = cp.arange(R.shape[1])[None, :] < cp.asarray(s0s[:, None])
    L1 = cp.sum(cp.log(R + darkrate[chs][:, None]) * index, axis=1)
    return cp.asnumpy(L1 + Rsum)

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

def concat(iterator, entries, zlength_max, num):
    '''
    将多个事例数组拼接, 并按 z 的长度排序分块
    entries:
        事例总数
    zlength_max:
        整个文件内 z 的最大长度
    num: 
        分块数量
    eids.shape = (entries,)
    chs.shape = (entries, chnums)
    offsets.shape = (entries, chnums)
    zs.shape = (entries, chnums, zlength)
    s0s.shape = (entries, chnums)
    nu_lcs.shape = (entries, chnums)
    '''
    eids = np.zeros(entries, dtype=np.int64)
    chs = np.zeros((entries, chnums), dtype=np.int32)
    offsets = np.zeros((entries, chnums), dtype=np.float32)
    zs = np.zeros((entries, chnums, zlength_max), dtype=np.float32)
    s0s = np.zeros((entries, chnums), dtype=np.int32)
    nu_lcs = np.zeros((entries, chnums), dtype=np.float32)
    samplers = np.empty(entries, dtype=object)
    zlength = np.zeros(entries, dtype=np.int32)
    for i, (eid, ch, offset, z, s0, nu_lc, _, sampler) in enumerate(iterator):
        eids[i] = eid
        chs[i, :len(ch)] = ch
        offsets[i, ch] = offset
        s0s[i, ch] = s0
        nu_lcs[i, ch] = nu_lc
        zs[i, ch, :z.shape[1]] = z
        samplers[i] = sampler
        zlength[i] = np.max(s0)
    l_sort = np.sort(zlength)
    index_sort = np.argsort(zlength)
    split_point = np.zeros(num, dtype=np.int32)
    l_total = np.sum(zlength)
    l_cumsum = np.cumsum(l_sort)
    for i in range(1, num):
        split_point[i] = np.abs(l_cumsum - l_total * i / num).argmin()
        block = index_sort[split_point[i-1]: split_point[i]]
        zlength_block = l_sort[index_sort[split_point[i]]]
        yield eids[block], chs[block], offsets[block], zs[block, :, :zlength_block], s0s[block], nu_lcs[block], samplers[block]
    block = index_sort[split_point[i]:]
    yield eids[block], chs[block], offsets[block], zs[block], s0s[block], nu_lcs[block], samplers[block]

def resampleZ(iterators, events, maxs):
    '''
    返回 z 的重采样结果
    ich.shape = (N, 1)
    z_extend.shape = (N, 1, y)
    s0.shape = (N, 1)
    nu_lc.shape = (N, 1)
    '''
    ich = np.zeros((events), dtype=np.int32)
    z_extend = np.zeros((events, maxs), dtype=np.float32)
    s0 = np.zeros((events), dtype=np.int32)
    nu_lc = np.zeros((events), dtype=np.float32)
    for i, iterator in enumerate(iterators):
        ich[i], z, s0[i], nu_lc[i] = next(iterator)
        z = np.array(z)
        # NPE 扩展
        if s0[i] > z_extend.shape[1]:
            supply = np.zeros((events, s0[i] - z_extend.shape[1]), dtype=np.float32)
            z_extend = np.concatenate((z_extend, supply), axis=1)
        z_extend[i, :s0[i]] = z[:s0[i]]
    return ich, z_extend, s0, nu_lc

def Reconstruction(fsmp, sparsify, num, output, probe, pmt_pos, darkrate, timecalib, MC_step):
    '''
    reconstruction
    '''
    # 创建输出文件
    opts = {"compression": "gzip", "shuffle": True}
    with h5py.File(output, "w") as opt:
        # 读入波形分析结果
        waveform = FSMPreader(sparsify, fsmp)
        entries, zlength_max = waveform.get_size()
        dataset = opt.create_dataset("Recon", shape=(entries * MC_step,), dtype=dtype, **opts)
        i = 0
        for eids, chs, offsets, zs, s0s, nu_lcs, samplers in tqdm(concat(waveform.rand_iter(MC_step), entries, zlength_max, num)):
            # 预分配储存数组
            recon_step = np.zeros(len(eids), dtype=dtype)

            # 设定随机数
            np.random.seed(eids[0] % 1000000) # 取第一个事例编号设定随机数种子
            u_gibbs = np.log(np.random.uniform(0, 1, (MC_step, len(eids), gibbs_variables)))
            u_V = np.random.normal(0, 1, (MC_step, len(eids), V_variables))

            # 给出 vertex, LogLikelihhood 的初值并记录
            vertex0 = Detector.Init(zs, s0s, offsets, pmt_pos, timecalib)
            Likelihood_vertex0 = LH(vertex0, chs, offsets, zs, s0s, probe, darkrate, timecalib)

            # gibbs iteration
            for step in range(MC_step):
                # 对 z 采样
                ich_s, z_s, s0_s, nu_lc_s = resampleZ(samplers, len(eids), zs.shape[2])
                ch_s = chs[np.arange(chs.shape[0]), ich_s]
                z_o = zs[np.arange(zs.shape[0]), ch_s]
                s0_o = s0s[np.arange(s0s.shape[0]), ch_s]
                nu_lc_o = nu_lcs[np.arange(nu_lcs.shape[0]), ch_s]
                offset_s = offsets[np.arange(offsets.shape[0]), ch_s]
                LH_z_sample = LHch(vertex0, ch_s, z_s, s0_s, offset_s, probe, darkrate, timecalib)
                LH_z_origin = LHch(vertex0, ch_s, z_o, s0_o, offset_s, probe, darkrate, timecalib)
                accept_z = (nu_lc_o - nu_lc_s + LH_z_sample - LH_z_origin) > u_gibbs[step, :, 0]
                # update z
                s0s[np.arange(s0s.shape[0])[accept_z], ch_s[accept_z]] = s0_s[accept_z]
                nu_lcs[np.arange(nu_lcs.shape[0])[accept_z], ch_s[accept_z]] = nu_lc_s[accept_z]
                if z_s.shape[1] > z_o.shape[1]:
                    supply = np.zeros((zs.shape[0], zs.shape[1], z_s.shape[1] - z_o.shape[1]), dtype=np.float32)
                    zs = np.concatenate((zs, supply), axis=2)
                zs[np.arange(zs.shape[0])[accept_z], ch_s[accept_z]] = z_s[accept_z]
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
                recon_step['EventID'] = eids
                recon_step['step'] = step
                recon_step['x'] = vertex0[:, 0]
                recon_step['y'] = vertex0[:, 1]
                recon_step['z'] = vertex0[:, 2]
                recon_step['E'] = vertex0[:, 3]
                recon_step['t'] = vertex0[:, 4]
                recon_step['NPE'] = np.sum(s0s, axis=1)
                recon_step['Likelihood'] = Likelihood_vertex0
                recon_step['acceptz'] = accept_z
                recon_step['acceptr'] = accept_r
                recon_step['acceptE'] = accept_E
                recon_step['acceptt'] = accept_t
                dataset[i: i + len(recon_step)] = recon_step
                i += len(recon_step)
