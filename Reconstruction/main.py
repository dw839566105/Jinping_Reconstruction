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
MC_step = 1000


def Recon(filename, output):
    '''
    reconstruction

    filename: root reference file
    output: output file
    '''
    # Create the output file and the group
    print(filename) # filename
    # Create the output file and the group
    h5file = tables.open_file(output, mode="w", title="OneTonDetector",
                            filters = tables.Filters(complevel=9))
    group = "/"
    # Create tables
    ReconWATable = h5file.create_table(group, "ReconWA", pub.Recon, "Recon")
    reconwa = ReconWATable.row
    ReconMCMCTable = h5file.create_table(group, "ReconMCMC", pub.Recon, "Recon")
    reconmcmc = ReconMCMCTable.row
    # Loop for event
    f = pq.read_table(filename).to_pandas()
    f = f[(f['eid'] == 36350) & (f['step'] > 2500)] # burn 前 2500 步
    grouped = f.groupby("eid")
    for sid, group_eid in grouped:
        for step, group_step in group_eid.groupby("step"):
            fired_PMT = group_step["ch"].values
            time_array = group_step["PEt"].values
            pe_array, cid = np.histogram(fired_PMT, bins=np.arange(len(PMT_pos)+1))

            if np.sum(pe_array)==0:
                continue
            
            ############################################################################
            ###############               inner recon                  #################
            ############################################################################
            x0_in = pub.Initial.FitGrid(pe_array, MeshIn.mesh, MeshIn.tpl, time_array)
            reconwa['step'] = step
            reconwa['E'], reconwa['x'], reconwa['y'], reconwa['z'], reconwa['t'] = x0_in # x,y,z是归一化距离
            reconwa['EventID'] = sid
            Likelihood_x0_in = LH.Likelihood(x0_in[1:],
                    *(PMT_pos, fired_PMT, time_array, pe_array, coeff_pe, coeff_time, cart),
                    expect = False)

            # 将梯度下降方法更改为一步有效的 MCMC 晃动
            for iter in range(MC_step):
                result_in = perturbation(x0_in[1:])
                # 判断是否超出边界
                if result_in[-1] <= 0 or np.sum(result_in[0:3] ** 2) >= np.square(1):
                    continue
                else:
                    Likelihood_result_in = LH.Likelihood(result_in,
                        *(PMT_pos, fired_PMT, time_array, pe_array, coeff_pe, coeff_time, cart),
                        expect = False)
                    # 找到一步有效晃动退出
                    if Likelihood_result_in > Likelihood_x0_in:
                        x0_in[1:] = result_in
                        Likelihood_x0_in = Likelihood_result_in
                        break

            E_in = LH.Likelihood(x0_in[1:],
                *(PMT_pos, fired_PMT, time_array, pe_array, coeff_pe, coeff_time, cart),
                expect = True)
            
            ############################################################################
            ###############               outer recon                  #################
            ############################################################################

            x0_out = pub.Initial.FitGrid(pe_array, MeshOut.mesh, MeshOut.tpl, time_array)
            Likelihood_x0_out = LH.Likelihood(x0_out[1:],
                    *(PMT_pos, fired_PMT, time_array, pe_array, coeff_pe, coeff_time, cart),
                    expect = False)

            # 将梯度下降方法更改为一步有效的 MCMC 晃动
            for iter in range(MC_step):
                result_out = perturbation(x0_out[1:])
                # 判断是否超出边界
                if result_out[-1] <= 0 or np.sum(result_out[0:3] ** 2) >= np.square(1):
                    continue
                else:
                    Likelihood_result_out = LH.Likelihood(result_out,
                        *(PMT_pos, fired_PMT, time_array, pe_array, coeff_pe, coeff_time, cart),
                        expect = False)
                    # 找到一步有效晃动退出
                    if Likelihood_result_out > Likelihood_x0_out:
                        x0_out[1:] = result_out
                        Likelihood_x0_out = Likelihood_result_out
                        break
                        
            E_out = LH.Likelihood(x0_out[1:],
                *(PMT_pos, fired_PMT, time_array, pe_array, coeff_pe, coeff_time, cart),
                expect = True)
            
            reconmcmc['EventID'] = sid
            reconmcmc['step'] = step
            if Likelihood_x0_out > Likelihood_x0_in:
                reconmcmc['E'] = E_out
                reconmcmc['x'], reconmcmc['y'], reconmcmc['z'] = x0_out[1:4]*shell
                reconmcmc['Likelihood'] = Likelihood_x0_out
            else:
                reconmcmc['E'] = E_in
                reconmcmc['x'], reconmcmc['y'], reconmcmc['z'] = x0_in[1:4]*shell
                reconmcmc['Likelihood'] = Likelihood_x0_in               
            
            # print recon result
            print('-'*60)
            print('%d %d vertex: [%+.2f, %+.2f, %+.2f] radius: %+.2f, energy: %.2f, Likelihood: %+.6f' 
                % (sid, step, reconmcmc['x'], reconmcmc['y'], reconmcmc['z'], 
                    np.sqrt(reconmcmc['x']**2 + reconmcmc['y']**2 + reconmcmc['z']**2), reconmcmc['E'], reconmcmc['Likelihood']))

            reconmcmc.append()

    # Flush into the output file
    ReconMCMCTable.flush()
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

args = parser.parse_args()
print(args.filename)
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

Recon(args.filename, args.output)
