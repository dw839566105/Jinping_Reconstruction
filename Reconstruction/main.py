# recon range: [-1,1], need * detector radius
import numpy as np
import tables
import pyarrow.parquet as pq
import pandas as pd
import argparse
from argparse import RawTextHelpFormatter
from scipy.optimize import minimize
from zernike import RZern
from MCMC import perturbation
import pub
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(precision=3, suppress=True)

def process_group(group):
    '''
    Reset index_step in waveform analysis(FSMP)
    '''
    cumul = group.drop_duplicates(subset=['step']).copy()
    cumul = cumul.loc[cumul.index.repeat(cumul['count'])]
    cumul['cumulation'] = range(len(cumul))
    result = pd.merge(group, cumul[['step', 'cumulation']], on='step')
    return result

def Recon(filename, output):
    '''
    reconstruction

    filename: parquet reference file
    output: output file
    '''

    # Create the output file and the group
    h5file = tables.open_file(output, mode="w", title="OneTonDetector",
                            filters = tables.Filters(complevel=9))
    group = "/"
    # Create tables
    ReconBCTable = h5file.create_table(group, "ReconBC", pub.Recon, "Recon")
    reconbc = ReconBCTable.row
    ReconTable = h5file.create_table(group, "Recon", pub.Recon, "Recon")
    recon = ReconTable.row
    # Loop for event
    f = pq.read_table(filename).to_pandas()

    # few events test
    if args.num > 0:
        events = args.num
        eid_list = f['eid'].unique()[:events]
        f = f[f['eid'].isin(eid_list)]

    grouped = f.groupby("eid")
    for sid, group_eid in grouped:
        # mcmc 重建, 波形分析返回 2500 step，对应每个采样结果进行 MC_step 次 mcmc 晃动
        np.random.seed(sid)
        u = np.random.uniform(0, 1, 2500 * MC_step)
        grouped = group_eid.groupby(['ch', 'offset'])
        group_reset = grouped.apply(process_group)
        group_reset = group_reset.reset_index(drop=True)
        # 给定初值
        time_array = group_reset["PEt"].values + group_reset["offset"].values
        pe_array = np.array(group_reset['ch'].value_counts().reindex(range(len(PMT_pos)), fill_value=0))
        # reconbc 记录的是归一化的坐标
        reconbc['E'], reconbc['x'], reconbc['y'], reconbc['z'], reconbc['t'] = pub.Initial.FitGrid(pe_array, Mesh.mesh, Mesh.tpl, time_array)
        x0 = np.array([reconbc['x'], reconbc['y'], reconbc['z'], reconbc['t']])
        reconbc['EventID'] = sid

        for step in range(2500):
            data = group_reset[group_reset["cumulation"] == step]
            fired_PMT = data["ch"].values
            time_array = data["PEt"].values + data["offset"].values
            pe_array = np.array(data['ch'].value_counts().reindex(range(len(PMT_pos)), fill_value=0))
            event_parameter = (PMT_pos, fired_PMT, time_array, pe_array, coeff_pe, coeff_time, cart)
            Likelihood_x0 = - LH.Likelihood(x0, *event_parameter, expect = False)
            E0 = LH.Likelihood(x0, *event_parameter, expect = True)
            breakpoint()
            # 进行 MCMC 晃动
            for recon_step in range(MC_step):
                Likelihood_x0 = - LH.Likelihood(x0, *event_parameter, expect = False)
                E0 = LH.Likelihood(x0, *event_parameter, expect = True)
                x1, Likelihood_x1 = mcmc(x0, event_parameter)
                E1 = LH.Likelihood(x1, *event_parameter, expect = True)
                record_step = step * MC_step + recon_step
                recon['EventID'] = sid
                recon['step'] = record_step
                # Likelihood 是负值
                if ((Likelihood_x1 - Likelihood_x0) > np.log(u[record_step])):
                    recon['E'] = E1
                    recon['x'], recon['y'], recon['z'] = x1[0:3]*shell
                    recon['t'] = x1[3]
                    recon['Likelihood'] = Likelihood_x1
                    x0 = x1
                    recon['accept'] = 1
                else:
                    recon['E'] = E0
                    recon['x'], recon['y'], recon['z'] = x0[0:3]*shell
                    recon['t'] = x0[3]
                    recon['Likelihood'] = Likelihood_x0
                    recon['accept'] = 0
                recon.append()
        # echo
        '''
        print('-'*60)
        print('%d vertex: [%+.2f, %+.2f, %+.2f] radius: %+.2f, energy: %.2f, Likelihood: %+.6f' 
            % (sid, reconbc['x']*shell, reconbc['y']*shell, reconbc['z']*shell, 
                np.sqrt(reconbc['x']**2 + reconbc['y']**2 + reconbc['z']**2)*shell, reconbc['E'], reconbc['Likelihood']))
        '''
        reconbc.append()

    # Flush into the output file
    ReconBCTable.flush()
    ReconTable.flush() # 重建结果
    h5file.close()

parser = argparse.ArgumentParser(description='Process Reconstruction construction', formatter_class=RawTextHelpFormatter)
parser.add_argument('-f', '--filename', dest='filename', metavar='filename[*.parquet]', type=str,
                    help='The filename [*Q.parquet] to read')

parser.add_argument('-o', '--output', dest='output', metavar='output[*.h5]', type=str,
                    help='The output filename [*.h5] to save')

parser.add_argument('--pe', dest='pe', metavar='PECoeff[*.h5]', type=str, 
                    default=r'./coeff/PE_coeff.h5',
                    help='The pe coefficients file [*.h5] to be loaded')

parser.add_argument('--time', dest='time', metavar='TimeCoeff[*.h5]', type=str,
                    default=r'./coeff/Time_coeff.h5',
                    help='The time coefficients file [*.h5] to be loaded')

parser.add_argument('--PMT', dest='PMT', metavar='PMT[*.txt]', type=str, default=r'./PMT.txt',
                    help='The PMT file [*.txt] to be loaded')

parser.add_argument('-n', '--num', dest='num', type=int, default=10,
                    help='test event nums')

parser.add_argument('-m', '--MCstep', dest='MCstep', type=int, default=10,
                    help='mcmc step per PEt')

parser.add_argument('--init', dest='init', type=str, default=None,
                    help='init vertex method')

args = parser.parse_args()

shell = 0.65
Gain = 164
sigma = 40
MC_step = args.MCstep
PMT_pos = np.loadtxt(args.PMT)
coeff_pe, coeff_time, pe_type, time_type = pub.load_coeff.load_coeff_Single(PEFile = args.pe, TimeFile = args.time)
cart = None
if pe_type=='Zernike':
    LH = pub.LH_Zer
    Mesh = pub.construct_Leg(coeff_pe, PMT_pos, np.linspace(0.01, 0.95, 3))
elif pe_type == 'Legendre':
    LH = pub.LH_Leg
    Mesh = pub.construct_Leg(coeff_pe, PMT_pos, np.linspace(0.01, 0.95, 30))

def mcmc(init, parameter):
    Likelihood_init = - LH.Likelihood(init, *parameter, expect = False)
    result = perturbation(init)
    # 判断是否超出边界
    if np.sum(result[0:3] ** 2) >= np.square(0.97):
        return init, Likelihood_init
    else:
        Likelihood_result = - LH.Likelihood(result, *parameter, expect = False)
        return result, Likelihood_result

Recon(args.filename, args.output)
