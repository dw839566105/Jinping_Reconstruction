#!/usr/bin/env python3
'''
重建模拟数据的位置、能量
'''
import h5py
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
import tables
import mcmc
import Detector
from config import *
from DetectorConfig import E0, chnums, tau, ts
from tqdm import tqdm
import LH
from fsmp_reader import FSMPreader

class DataType(tables.IsDescription):
    '''
    重建数据储存类型
    '''
    EventID = tables.Int64Col(pos=0)    # EventNo
    step = tables.Int64Col(pos=1)       # wave step
    x = tables.Float32Col(pos=2)        # x position
    y = tables.Float32Col(pos=3)        # y position
    z = tables.Float32Col(pos=4)        # z position
    E = tables.Float32Col(pos=5)        # Energy
    t = tables.Float32Col(pos=6)        # time
    Likelihood = tables.Float64Col(pos=7)
    acceptz = tables.Float32Col(pos=8)
    acceptr = tables.Float32Col(pos=9)
    acceptE = tables.Float32Col(pos=10)
    acceptt = tables.Float32Col(pos=11)

def genPE(chs, s0s):
    '''
    取出所有通道的 NPE
    '''
    pe_array = np.zeros(chnums)
    for i in range(len(chs)):
        pe_array[chs[i]] = s0s[i]
    return pe_array

def genTime(zs, s0s, offsets, T_i):
    '''
    取出所有通道的 PEt，展开为一维数组
    '''
    time_array = np.zeros(np.sum(s0s))
    j = 0
    for i, s0 in enumerate(s0s):
        time_array[j : j + s0] = zs[i, :s0] + offsets[i] - T_i[i]
        j += s0
    return time_array

def reconstruction(fsmp, sparsify, Entries, output, probe, pmt_pos, MC_step, time_mode, record_mode):
    '''
    reconstruction
    '''
    # Create the output file, the group, tables
    h5file = tables.open_file(output, mode="w", title="OSIRISDetector", filters = tables.Filters(complevel=9))
    group = "/"
    InitTable = h5file.create_table(group, "Init", DataType, "Init")
    init = InitTable.row
    ReconTable = h5file.create_table(group, "Recon", DataType, "Recon")
    recon = ReconTable.row
    if record_mode == "ON":
        SampleTable = h5file.create_table(group, "Sample", DataType, "Sample")
        sample = SampleTable.row

    # start reconstruction
    eid_start = 0
    for eid, chs, offsets, zs, s0s, nu_lcs, mu0s, sampler in tqdm(FSMPreader(sparsify, fsmp).rand_iter(MC_step)):
        # 将 s0s 转换为 int16
        s0s = s0s.astype('int16')

        # 设定随机数
        np.random.seed(eid % 1000000)
        u = np.random.uniform(0, 1, (MC_step, variables))
    
        # 给出 vertex, LogLikelihhood 的初值
        pe_array = genPE(chs, s0s)
        time_array = genTime(zs, s0s, offsets, np.zeros(len(s0s)))
        vertex0 = Detector.Init(pe_array, time_array, pmt_pos, time_mode)
        Likelihood_vertex0 = LH.LogLikelihood(vertex0, pe_array, zs, s0s, offsets, chs, probe, time_mode)
        init['x'], init['y'], init['z'], init['E'], init['t'] = vertex0
        init['EventID'] = eid
        init.append()

        for recon_step in range(MC_step):
            recon['EventID'] = eid
            recon['step'] = recon_step

            # 对 z 采样
            expect = probe.callPE(vertex0)
            i_ch, z, s0, nu_lc = next(sampler)
            s0 = np.int16(s0)
            # 以 count 为权重，从所有 channel 随机采样的组合
            offset = offsets[i_ch]
            mu0 = mu0s[i_ch]
            if time_mode == "ON":
                T_i = probe.callT(vertex0, ch)
                log_ratio_time = np.sum(LH.LogLikelihood_quantile(z[:s0] + offset, T_i, tau, ts)) - np.sum(LH.LogLikelihood_quantile(zs[i_ch][:s0s[i_ch]] + offset, T_i, tau, ts))
                log_ratio = (s0 - s0s[i_ch]) * np.log(expect[chs[i_ch]] * vertex0[3]) + nu_lcs[i_ch] - nu_lc + log_ratio_time
            else:
                log_ratio = (s0 - s0s[i_ch]) * np.log(expect[chs[i_ch]] * vertex0[3] / mu0)
            if log_ratio > np.log(u[recon_step, 0]):
                s0s[i_ch] = s0
                nu_lcs[i_ch] = nu_lc
                zs[i_ch] = z
                Likelihood_vertex0 = LH.LogLikelihood(vertex0, pe_array, zs, s0s, offsets, chs, probe, time_mode)
                recon['acceptz'] = 1

            # 对位置采样
            vertex1 = mcmc.Perturb_pos(vertex0, u[recon_step, 1:4], r_max)

            # EM 计算 E
            expect = probe.callPE(vertex1)
            expect[expect == 0] = 1E-6
            pe_array = genPE(chs, s0s)
            vertex1[3] = np.sum(pe_array) / np.sum(expect) * E0

            # EM 计算 t
            if time_mode == "ON":
                T_i = probe.callT(vertex0, chs)
                time_array = genTime(zs, s0s, offsets, T_i)
                vertex1[-1] = np.quantile(time_array, tau)
            
            if Detector.Boundary(vertex1): ## 边界检查
                Likelihood_vertex1 = LH.LogLikelihood(vertex1, pe_array, zs, s0s, offsets, chs, probe, time_mode)
                if record_mode == "ON":
                    sample['EventID'] = eid
                    sample['step'] = recon_step
                    sample['x'], sample['y'], sample['z'], sample['E'], sample['t'] = vertex1
                    sample['Likelihood'] = Likelihood_vertex1
                    sample.append()

                if ((Likelihood_vertex1 - Likelihood_vertex0) > np.log(u[recon_step, 4])):
                    vertex0 = vertex1
                    Likelihood_vertex0 = Likelihood_vertex1
                    recon['acceptr'] = 1

            recon['x'], recon['y'], recon['z'], recon['E'], recon['t'] = vertex0
            recon['Likelihood'] = Likelihood_vertex0
            recon.append()

        # 如果 Entries 不为 0，只重建 Entries 个事例
        eid_start += 1
        if Entries != 0:
            if eid_start > Entries - 1:
                break

    # Flush into the output file
    InitTable.flush()
    ReconTable.flush() # 重建结果
    if record_mode == "ON":
        SampleTable.flush()
    h5file.close()

