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
import Read
import Detector
from config import *
from DetectorConfig import E0
from tqdm import tqdm
import LH

def reconstruction(data, output, probe, pmt_pos, MC_step, sampling_mode, data_mode, time_mode):
    '''
    reconstruction
    '''
    # Create the output file, the group, tables
    h5file = tables.open_file(output, mode="w", title="OSIRISDetector", filters = tables.Filters(complevel=9))
    group = "/"
    InitTable = h5file.create_table(group, "Init", Read.DataType, "Init")
    init = InitTable.row
    ReconTable = h5file.create_table(group, "Recon", Read.DataType, "Recon")
    recon = ReconTable.row

    # start reconstruction
    for eid, data_eid in tqdm(data.groupby('eid')):
        print(f"Start processing eid-{eid}")
        data_eid = data_eid.reset_index(drop=True)

        # 设定随机数
        np.random.seed(eid % 1000000)
        u = np.random.uniform(0, 1, (MC_step, variables))

        # 给出 Z 的初值
        if data_mode == "raw":
            Z0 = data_eid.loc[data_eid.groupby(['ch', 'offset'])['count'].idxmax()]
            # 给出 z 抽样的随机数
            index_sampled = np.random.choice(data_eid.index, size=MC_step, p=data_eid['count']/data_eid['count'].sum())
            u_z = np.random.uniform(0, 1, MC_step)
        else:
            Z0 = data_eid.copy()

        # 给出 vertex, LogLikelihhood 的初值
        vertex0 = Detector.Init(Z0, pmt_pos, time_mode)
        Likelihood_vertex0 = LH.LogLikelihood(vertex0, Z0, probe, time_mode, data_mode)
        init['x'], init['y'], init['z'], init['E'], init['t'] = vertex0
        init['EventID'] = eid
        init.append()

        # 根据能量调整 r 晃动步长
        r_max_E = r_max / np.clip(np.sqrt(vertex0[3]), 1, None)

        for recon_step in range(MC_step):
            recon['EventID'] = eid
            recon['step'] = recon_step

            if data_mode == "raw":
                # 对 z 采样 
                expect = probe.callPE(vertex0)
                expect[expect == 0] = 1E-6
                s0_sampled = data_eid['s0'].values[index_sampled[recon_step]]
                ch_sampled = data_eid['ch'].values[index_sampled[recon_step]]
                mu0_sampled = data_eid['mu0'].values[index_sampled[recon_step]]
                nu_lc_sampled = data_eid['nu_lc'].values[index_sampled[recon_step]]
                Z0_sampled_index = Z0.query(f'ch == {ch_sampled}').index
                if time_mode == "ON":
                    log_ratio = (s0_sampled - Z0.loc[Z0_sampled_index]['s0'].values) * np.log(expect[ch_sampled] * vertex0[3]) + Z0.loc[Z0_sampled_index]['nu_lc'].values - nu_lc_sampled
                else:
                    log_ratio = (s0_sampled - Z0.loc[Z0_sampled_index]['s0'].values) * np.log(expect[ch_sampled] * vertex0[3] / mu0_sampled)
                if log_ratio > np.log(u_z[recon_step]):
                    Z0.loc[Z0_sampled_index] = data_eid.loc[index_sampled[recon_step]].values
                    recon['acceptz'] = 1

            # 对位置采样
            vertex1 = mcmc.Perturb_pos(vertex0, u[recon_step, :3], r_max_E)
            ## 边界检查
            if Detector.Boundary(vertex1):
                Likelihood_vertex1 = LH.LogLikelihood(vertex1, Z0, probe, time_mode, data_mode)
                if ((Likelihood_vertex1 - Likelihood_vertex0) > np.log(u[recon_step, 3])):
                    vertex0[:3] = vertex1[:3]
                    vertex0[-1] = vertex1[-1]
                    if sampling_mode == "EM":
                        expect = probe.callPE(vertex0)
                        expect[expect == 0] = 1E-6
                        pe_array = LH.genPE(Z0)
                        vertex0[3] = np.sum(pe_array) / np.sum(expect) * E0
                    Likelihood_vertex0 = Likelihood_vertex1
                    recon['acceptr'] = 1
                    
            if sampling_mode == "Gibbs":
                # 对能量采样
                vertex2 = mcmc.Perturb_energy(vertex0, u[recon_step, 4])
                ## 边界检查
                if vertex2[3] > 0:
                    Likelihood_vertex2 = LH.LogLikelihood(vertex2, Z0, probe, time_mode, data_mode)
                    if ((Likelihood_vertex2 - Likelihood_vertex0) > np.log(u[recon_step, 5])):
                        vertex0[3] = vertex2[3]
                        Likelihood_vertex0 = Likelihood_vertex2
                        recon['acceptE'] = 1

            recon['x'], recon['y'], recon['z'], recon['E'], recon['t'] = vertex0
            recon['Likelihood'] = Likelihood_vertex0
            recon.append()
        print(f"Recon eid-{eid} Done!")

    # Flush into the output file
    InitTable.flush()
    ReconTable.flush() # 重建结果
    h5file.close()

