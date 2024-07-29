#!/usr/bin/env python3
'''
重建模拟数据的位置、能量
'''
import numpy as np
import tables
import Detector
from config import *
from DetectorConfig import chnums
from tqdm import tqdm
from fsmp_reader import FSMPreader

class DataType(tables.IsDescription):
    '''
    重建数据储存类型
    '''
    EventID = tables.Int64Col(pos=0)    # EventNo
    step = tables.Int32Col(pos=1)       # wave step
    x = tables.Float16Col(pos=2)        # x position
    y = tables.Float16Col(pos=3)        # y position
    z = tables.Float16Col(pos=4)        # z position
    E = tables.Float16Col(pos=5)        # Energy
    t = tables.Float16Col(pos=6)        # time
    NPE = tables.Int32Col(pos=7)        # NPE
    Likelihood = tables.Float32Col(pos=8)
    acceptz = tables.Int32Col(pos=9)
    acceptr = tables.Int32Col(pos=10)
    acceptt = tables.Int32Col(pos=11)

dtype = np.dtype([('EventID', '<i8'), ('step', '<i4'), ('x', '<f2'), ('y', '<f2'),
                  ('z', '<f2'), ('E', '<f2'), ('t', '<f2'), ('NPE', '<i4'),
                  ('Likelihood', '<f4'), ('acceptz', '<i4'), ('acceptr', '<i4'),
                  ('acceptt', '<i4')])

def LH(vertex, fires, chs, offsets, zs, s0s, probe, darkrate, timecalib):
    '''
    计算 Likelihood
    '''
    PEt = zs + offsets[:,:,None] - vertex[:, 4][:,None,None] + timecalib[chs][:,:,None]
    R = probe.genR(vertex, PEt, chs, fires)
    index = np.arange(R.shape[2])[None, None, :] < s0s[:, :, None]
    L1 = np.sum(np.log(R + darkrate[chs][:,:,None]) * index, axis=(1,2))
    Rsum = probe.genRsum(vertex, chs, fires)
    index = np.arange(Rsum.shape[1])[None, :] < fires[:, None]
    L2 = - np.sum(Rsum * index, axis=1)
    return L1 + L2

# TODO
def LHch():
    pass

# TODO
def Regression(vertex, fires, chs, offsets, zs, s0s, probe, darkrate, timecalib):
    pass

def concat(iterator, Entries):
    '''
    将多个事例数组拼接
    eids.shape = (N, 1)
    fires.shape = (N, 1)
    chs.shape = (N, x)
    offsets.shape = (N, x)
    zs.shape = (N, x, y)
    s0s.shape = (N, x)
    nu_lcs.shape = (N, x)
    '''
    eids = np.zeros(Entries, dtype=np.int64)
    fires = np.zeros(Entries, dtype=np.int32)
    chs = np.zeros((Entries, chnums), dtype=np.int32)
    offsets = np.zeros((Entries, chnums), dtype=np.float32)
    zs = np.zeros((Entries, chnums, 10), dtype=np.float32)
    s0s = np.zeros((Entries, chnums), dtype=np.int32)
    nu_lcs = np.zeros((Entries, chnums), dtype=np.float32)
    samplers = []
    for iter, (eid, ch, offset, z, s0, nu_lc, _, sampler) in enumerate(iterator):
        i = iter % Entries
        # 取数后重置
        if i == 0:
            eids = np.zeros(Entries, dtype=np.int64)
            fires = np.zeros(Entries, dtype=np.int32)
            chs = np.zeros((Entries, chnums), dtype=np.int32)
            offsets = np.zeros((Entries, chnums), dtype=np.float32)
            zs = np.zeros((Entries, chnums, 10), dtype=np.float32)
            s0s = np.zeros((Entries, chnums), dtype=np.int32)
            nu_lcs = np.zeros((Entries, chnums), dtype=np.float32)
            samplers = []
        # 数据类型更正
        s0 = s0.astype('int32')
        eids[i] = eid
        fires[i] = len(ch)
        # 通道数量扩展
        ch_increment = fires[i] - chs.shape[1]
        if ch_increment > 0:
            supply = np.zeros((Entries, ch_increment), dtype=np.int32)
            chs = np.concatenate((chs, supply), axis=1)
            s0s = np.concatenate((s0s, supply), axis=1)
            supply = np.zeros((Entries, ch_increment), dtype=np.float32)
            offsets = np.concatenate((offsets, supply), axis=1)
            nu_lcs = np.concatenate((nu_lcs, supply), axis=1)
            supply = np.zeros((Entries, ch_increment, zs.shape[2]), dtype=np.float32)
            zs = np.concatenate((zs, supply), axis=1)
        chs[i, :fires[i]] = ch
        offsets[i, :fires[i]] = offset
        s0s[i, :fires[i]] = s0
        nu_lcs[i, :fires[i]] = nu_lc
        # NPE 扩展
        if z.shape[1] > zs.shape[2]:
            supply = np.zeros((Entries, zs.shape[1], z.shape[1] - zs.shape[2]), dtype=np.float32)
            zs = np.concatenate((zs, supply), axis=2)
        zs[i, :z.shape[0], :z.shape[1]] = z
        samplers.append(sampler)
        # 取数
        if i == Entries - 1:
            yield eids, fires, chs, offsets, zs, s0s, nu_lcs, samplers
    # 不足 Entries 个事例取完
    yield eids[:i], fires[:i], chs[:i], offsets[:i], zs[:i], s0s[:i], nu_lcs[:i], samplers[:i]
                    
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
        # NPE 扩展
        if s0[i] > z_extend.shape[1]:
            supply = np.zeros((events, s0[i] - z_extend.shape[1]), dtype=np.float32)
            z_extend = np.concatenate((z_extend, supply), axis=1)
        z_extend[i, :s0[i]] = z[:s0[i]]
    return ich, z_extend, s0, nu_lc

def Reconstruction(fsmp, sparsify, Entries, output, probe, pmt_pos, darkrate, timecalib, MC_step):
    '''
    reconstruction
    '''
    # create tables
    h5file = tables.open_file(output, mode="w", title="Detector", filters = tables.Filters(complevel=9))
    group = "/"
    ReconTable = h5file.create_table(group, "Recon", DataType, "Recon")
    for eids, fires, chs, offsets, zs, s0s, nu_lcs, samplers in tqdm(concat(FSMPreader(sparsify, fsmp).rand_iter(MC_step), Entries)):
        # 设定随机数
        np.random.seed(eids[0] % 1000000) # 取第一个事例编号设定随机数种子
        u_gibbs = np.random.uniform(0, 1, (MC_step, len(eids), gibbs_variables))
        u_V = np.random.normal(0, 1, (MC_step, len(eids), V_variables))
        
        # test
        Likelihood_vertex0 = LH(np.zeros((len(eids), 5)), fires, chs, offsets, zs, s0s, probe, darkrate, timecalib)

        # 给出 vertex, LogLikelihhood 的初值并记录
        vertex0 = Detector.Init(zs, s0s, offsets, chs, pmt_pos, timecalib) 
        Likelihood_vertex0 = LH(vertex0, fires, chs, offsets, zs, s0s, probe, darkrate, timecalib)

        # gibbs iteration
        for step in range(MC_step):
            # 对 z 采样
            ich_s, z_s, s0_s, nu_lc_s = resampleZ(samplers, len(eids), zs.shape[2])
            ch_s = chs[np.arange(chs.shape[0]), ich_s]
            z_o = zs[np.arange(zs.shape[0]), ich_s]
            s0_o = s0s[np.arange(s0s.shape[0]), ich_s]
            nu_lc_o = nu_lcs[np.arange(nu_lcs.shape[0]), ich_s]
            offset_s = offsets[np.arange(offsets.shape[0]), ich_s]
            LH_z_sample = LHch(vertex0, ch_s, z_s, s0_s, offset_s, probe, darkrate, timecalib)
            LH_z_origin = LHch(vertex0, ch_s, z_o, s0_o, offset_s, probe, darkrate, timecalib)
            accept_z = (nu_lc_o - nu_lc_s + LH_z_sample - LH_z_origin) > u_gibbs[step, :, 0]
            # update z
            s0s[np.arange(s0s.shape[0])[accept_z], ich_s[accept_z]] = s0_s[accept_z]
            nu_lcs[np.arange(nu_lcs.shape[0])[accept_z], ich_s[accept_z]] = nu_lc_s[accept_z]
            if z_s.shape[2] > z_o.shape[2]:
                supply = np.zeros((zs.shape[0], zs.shape[1], z_s.shape[2] - z_o.shape[2]), dtype=np.float32)
                zs = np.concatenate((zs, supply), axis=2)
            zs[np.arange(zs.shape[0])[accept_z], ich_s[accept_z]] = z_s[accept_z]
            # update Likelihood
            Likelihood_vertex0[accept_z] = LH(vertex0, fires[accept_z], chs[accept_z], offsets[accept_z], zs[accept_z], s0s[accept_z], probe, darkrate, timecalib)

            # 对 r 采样
            vertex1 = vertex0.copy()
            vertex1[:, :3] = vertex1[:, :3] + r_sigma * u_V[step, :, :3]
            if Detector.Boundary(vertex1):
                # regression on E
                vertex1[:, 3] = Regression(vertex1, fires, chs, offsets, zs, s0s, probe, darkrate, timecalib)
                Likelihood_vertex1 = LH(vertex1, fires, chs, offsets, zs, s0s, probe, darkrate, timecalib)
                accept_r = Likelihood_vertex1 - Likelihood_vertex0 > u_gibbs[step, :, 1]
                # update r
                vertex0[accept_r] = vertex1[accept_r]
                Likelihood_vertex0[accept_r] = Likelihood_vertex1[accept_r]

            # 对 t 采样
            vertex2 = vertex0.copy()
            vertex2[:, 4] = vertex2[:, 4] + T_sigma * u_V[step, :, 3]
            Likelihood_vertex2 = LH(vertex2, fires, chs, offsets, zs, s0s, probe, darkrate, timecalib)
            accept_t = Likelihood_vertex2 - Likelihood_vertex0 > u_gibbs[step, :, 2]
            # update t
            vertex0[accept_t] = vertex2[accept_t]
            Likelihood_vertex0[accept_t] = Likelihood_vertex2[accept_t]

            # write into tables
            result_step = np.hstack((eids, np.repeat(step, len(eids)), vertex0, np.sum(s0s, axis=1), Likelihood_vertex0, accept_z, accept_r, accept_t)).astype(dtype)
            ReconTable.append(result_step)
    
    ReconTable.flush()
    h5file.close()








