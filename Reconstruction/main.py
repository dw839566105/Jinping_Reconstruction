# recon range: [-1,1], need * detector radius
import numpy as np
import tables
import pyarrow.parquet as pq
import argparse
from argparse import RawTextHelpFormatter
from scipy.optimize import minimize
from zernike import RZern
from MCMC import perturbation
import pub
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(precision=3, suppress=True)

# boundaries
shell = 0.65
Gain = 164
sigma = 40
MC_step = 5 # 进行有效 mcmc 晃动步数
MC_maxfind = 1000 # 寻找一步有效 mcmc 晃动时执行的最多步数


def Recon(filename, output):
    '''
    reconstruction

    filename: root reference file
    output: output file
    '''

    # Create the output file and the group
    h5file = tables.open_file(output, mode="w", title="OneTonDetector",
                            filters = tables.Filters(complevel=9))
    group = "/"
    # Create tables
    ReconWATable = h5file.create_table(group, "ReconWA", pub.Recon, "Recon")
    reconwa = ReconWATable.row
    # 暂时把 reconin 和 reconout 记录下来。（用来比较收敛性，后续删去）
    ReconInTable = h5file.create_table(group, "ReconIn", pub.Recon, "Recon")
    reconin = ReconInTable.row
    ReconOutTable = h5file.create_table(group, "ReconOut", pub.Recon, "Recon")
    reconout = ReconOutTable.row
    ReconMCMCTable = h5file.create_table(group, "ReconMCMC", pub.Recon, "Recon")
    reconmcmc = ReconMCMCTable.row
    # Loop for event
    f = pq.read_table(filename).to_pandas()
    f = f[f['step'] > 2500] # burn 前 2500 步

    # single event test
    if args.event:
        f = f[f['eid'] == args.event]

    grouped = f.groupby("eid")
    for sid, group_eid in grouped:
        # 电荷重心法给出每个事例初值
        minstep = np.min(group_eid["step"].values)
        condition = (group_eid['step'] == minstep)
        fired_PMT_init = group_eid[condition]["ch"].values
        time_array_init = group_eid[condition]["PEt"].values
        pe_array_init, cid = np.histogram(fired_PMT_init, bins=np.arange(len(PMT_pos)+1))
        x0_in = pub.Initial.FitGrid(pe_array_init, MeshIn.mesh, MeshIn.tpl, time_array_init)
        x0_out = pub.Initial.FitGrid(pe_array_init, MeshOut.mesh, MeshOut.tpl, time_array_init)

        reconwa['E'], reconwa['x'], reconwa['y'], reconwa['z'], reconwa['t'] = x0_in # x,y,z是归一化距离
        reconwa['EventID'] = sid

        for step, group_step in group_eid.groupby("step"):
            fired_PMT = group_step["ch"].values
            time_array = group_step["PEt"].values
            pe_array, cid = np.histogram(fired_PMT, bins=np.arange(len(PMT_pos)+1))

            if np.sum(pe_array)==0:
                continue
            
            event_parameter = (PMT_pos, fired_PMT, time_array, pe_array, coeff_pe, coeff_time, cart)

            # 进行MCMC晃动
            for recon_step in range(MC_step):
                x0_in[1:], Likelihood_x0_in = mcmc(x0_in[1:], event_parameter)
                x0_out[1:], Likelihood_x0_out = mcmc(x0_out[1:], event_parameter)
                E_in = LH.Likelihood(x0_in[1:], *event_parameter, expect = True)
                E_out = LH.Likelihood(x0_out[1:], *event_parameter, expect = True)

                reconmcmc['EventID'] = sid
                reconmcmc['step'] = recon_step
                if Likelihood_x0_out > Likelihood_x0_in:
                    reconmcmc['E'] = E_out
                    reconmcmc['x'], reconmcmc['y'], reconmcmc['z'] = x0_out[1:4]*shell
                    reconmcmc['Likelihood'] = Likelihood_x0_out
                else:
                    reconmcmc['E'] = E_in
                    reconmcmc['x'], reconmcmc['y'], reconmcmc['z'] = x0_in[1:4]*shell
                    reconmcmc['Likelihood'] = Likelihood_x0_in        
                
                # 记录 reconin 和 reconout （用来比较收敛性，后续删去）
                reconin['EventID'] = sid
                reconin['step'] = recon_step
                reconin['E'] = E_in
                reconin['x'], reconin['y'], reconin['z'] = x0_in[1:4]*shell
                reconin['Likelihood'] = Likelihood_x0_in
                reconout['EventID'] = sid
                reconout['step'] = recon_step
                reconout['E'] = E_out
                reconout['x'], reconout['y'], reconout['z'] = x0_out[1:4]*shell
                reconout['Likelihood'] = Likelihood_x0_out

                reconwa.append()
                reconin.append()
                reconout.append()
                reconmcmc.append()

    # Flush into the output file
    ReconWATable.flush() # reconin 的电荷重心法初值
    ReconInTable.flush() # 初值在全反射区域内得到的重建结果
    ReconOutTable.flush() # 初值在全反射区域外得到的重建结果
    ReconMCMCTable.flush() # 重建结果
    h5file.close()

parser = argparse.ArgumentParser(description='Process Reconstruction construction', formatter_class=RawTextHelpFormatter)
parser.add_argument('-f', '--filename', dest='filename', metavar='filename[*.pqf]', type=str,
                    help='The filename [*Q.pqf] to read')

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

parser.add_argument('--event', dest='event', type=int, default=None,
                    help='test event')

args = parser.parse_args()

PMT_pos = np.loadtxt(args.PMT)
coeff_pe, coeff_time, pe_type, time_type = pub.load_coeff.load_coeff_Single(PEFile = args.pe, TimeFile = args.time)
cart = RZern(30)
if pe_type=='Zernike':
    LH = pub.LH_Zer
    MeshIn = pub.construct_Zer(coeff_pe, PMT_pos, np.linspace(0.01, 0.92, 3), cart)
    MeshOut = pub.construct_Zer(coeff_pe, PMT_pos, np.linspace(0.92, 1, 3), cart)
elif pe_type == 'Legendre':
    LH = pub.LH_Leg
    MeshIn = pub.construct_Leg(coeff_pe, PMT_pos, np.linspace(0.01, 0.92, 30))
    MeshOut = pub.construct_Leg(coeff_pe, PMT_pos, np.linspace(0.92, 1, 30))

def mcmc(init, parameter):
    Likelihood_init = LH.Likelihood(init, *parameter, expect = False)
    result = perturbation(init)
    # 判断是否超出边界
    if np.sum(result[0:3] ** 2) >= np.square(1):
        return init, Likelihood_init
    else:
        Likelihood_result = LH.Likelihood(result, *parameter, expect = False)
        # 通过似然函数判断晃动是否有效
        if Likelihood_result > Likelihood_init:
            return result, Likelihood_result
    return init, Likelihood_init

Recon(args.filename, args.output)
