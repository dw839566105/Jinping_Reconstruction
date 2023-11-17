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
MC_step = 10 # 进行 mcmc 晃动步数，后续根据 Gelman-Rubin 确定

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
    ReconTable = h5file.create_table(group, "Recon", pub.Recon, "Recon")
    recon = ReconTable.row
    # Loop for event
    f = pq.read_table(filename).to_pandas()

    # single event test
    if args.event:
        f = f[f['eid'] == args.event]

    grouped = f.groupby("eid")
    for sid, group_eid in grouped:
        for step, group_step in group_eid.groupby("step"):
            # 初值为探测器中央
            x0 = np.array([0, 0 ,0 ,0])
            fired_PMT = group_step["ch"].values
            time_array = group_step["PEt"].values
            pe_array = group_step['ch'].value_counts().reindex(range(len(PMT_pos)), fill_value=0)        
            event_parameter = (PMT_pos, fired_PMT, time_array, pe_array, coeff_pe, coeff_time, cart)
            Likelihood_x0 = LH.Likelihood(x0, *event_parameter, expect = False)
            E0 = LH.Likelihood(x0, *event_parameter, expect = True)

            # 进行MCMC晃动
            for recon_step in range(MC_step):
                x1, Likelihood_x1 = mcmc(x0, event_parameter)
                E1 = LH.Likelihood(x1, *event_parameter, expect = True)
                np.random.seed(recon_step)
                u = np.random.uniform(0,1)

                recon['EventID'] = sid
                recon['wavestep'] = step
                recon['reconstep'] = recon_step
                recon['count'] = group_step["count"].values[0]
                if Likelihood_x1 / Likelihood_x0 > u:
                    recon['E'] = E1
                    recon['x'], recon['y'], recon['z'] = x1[0:3]*shell
                    recon['Likelihood'] = Likelihood_x1
                else:
                    recon['E'] = E0
                    recon['x'], recon['y'], recon['z'] = x0[0:3]*shell
                    recon['Likelihood'] = Likelihood_x0
            
                # echo
                print('-'*60)
                print('%d vertex: [%+.2f, %+.2f, %+.2f] radius: %+.2f, energy: %.2f, Likelihood: %+.6f' 
                    % (sid, recon['x'], recon['y'], recon['z'], 
                        np.sqrt(recon['x']**2 + recon['y']**2 + recon['z']**2), recon['E'], recon['Likelihood']))

                recon.append()

    # Flush into the output file
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

parser.add_argument('--event', dest='event', type=int, default=None,
                    help='test event')

args = parser.parse_args()

PMT_pos = np.loadtxt(args.PMT)
coeff_pe, coeff_time, pe_type, time_type = pub.load_coeff.load_coeff_Single(PEFile = args.pe, TimeFile = args.time)
cart = None
if pe_type=='Zernike':
    LH = pub.LH_Zer
elif pe_type == 'Legendre':
    LH = pub.LH_Leg

def mcmc(init, parameter):
    Likelihood_init = LH.Likelihood(init, *parameter, expect = False)
    result = perturbation(init)
    # 判断是否超出边界
    if np.sum(result[0:3] ** 2) >= np.square(1):
        return init, Likelihood_init
    else:
        Likelihood_result = LH.Likelihood(result, *parameter, expect = False)
        return result, Likelihood_result

Recon(args.filename, args.output)
