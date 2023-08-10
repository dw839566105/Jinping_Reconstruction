# recon range: [-1,1], need * detector radius
import numpy as np
import tables
import pyarrow.parquet as pq
import awkward as ak
import argparse
from argparse import RawTextHelpFormatter
from scipy.optimize import minimize
from zernike import RZern
import pub
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(precision=3, suppress=True)

# boundaries
shell = 0.65
Gain = 164
sigma = 40


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
    ReconInTable = h5file.create_table(group, "ReconIn", pub.Recon, "Recon")
    reconin = ReconInTable.row
    ReconOutTable = h5file.create_table(group, "ReconOut", pub.Recon, "Recon")
    reconout = ReconOutTable.row
    # Loop for event
    f = pq.read_table(filename).to_pandas()
    #暂时只取迭代的最后一步
    #filtered_f = f.groupby(['eid', 'ch']).apply(lambda x: x[x['step'] == x['step'].max()])
    #f = filtered_f.reset_index(drop=True)

    grouped = f.groupby("eid")
    for sid, group_f in grouped:
        steps = group_f.groupby('step')['count'].first().sum()
        group_f = group_f.reindex(group_f.index.repeat(group_f['count']))
        fired_PMT = group_f["ch"].values
        time_array = group_f["PEt"].values
        #pe_array = group_f["e"].values

        pe_array, cid = np.histogram(fired_PMT, bins=np.arange(len(PMT_pos)+1)) / steps

        #count = group_f["count"].values

        #取最后1个step做先验
        #prior_f = group_f.groupby(['ch']).apply(lambda x: x[x['step'] == x['step'].max()])
        #prior_f = prior_f.reset_index(drop=True)
        #prior_fired_PMT = prior_f["ch"].values
        #prior_pe_array, cid = np.histogram(prior_fired_PMT, bins=np.arange(len(PMT_pos)+1))
        #prior_time_array = prior_f["PEt"].values

        if np.sum(pe_array)==0:
            continue
        
        ############################################################################
        ###############               inner recon                  #################
        ############################################################################
        x0_in = pub.Initial.FitGrid(pe_array, MeshIn.mesh, MeshIn.tpl, time_array)
        reconwa['E'], reconwa['x'], reconwa['y'], reconwa['z'], reconwa['t'] = x0_in
        reconwa['EventID'] = sid
        result_in = minimize(LH.Likelihood, x0_in[1:], method='SLSQP', 
            bounds=((-1, 1), (-1, 1), (-1, 1), (None, None)),
            args = (PMT_pos, fired_PMT, time_array, pe_array, coeff_pe, coeff_time, cart))
        E_in = LH.Likelihood(result_in.x,
            *(PMT_pos, fired_PMT, time_array, pe_array, coeff_pe, coeff_time, cart),
            expect = True)

        reconin['EventID'] = sid
        reconin['E'] = E_in
        reconin['x'], reconin['y'], reconin['z'] = result_in.x[:3]*shell
        reconin['success'] = result_in.success
        reconin['Likelihood'] = result_in.fun
        
        ############################################################################
        ###############               outer recon                  #################
        ############################################################################

        x0_out = pub.Initial.FitGrid(pe_array, MeshOut.mesh, MeshOut.tpl, time_array)
        result_out = minimize(LH.Likelihood, x0_out[1:], method='SLSQP', 
            bounds=((-1, 1), (-1, 1), (-1, 1), (None, None)),
            args = (PMT_pos, fired_PMT, time_array, pe_array, coeff_pe, coeff_time, cart))
        E_out = LH.Likelihood(result_out.x,
            *(PMT_pos, fired_PMT, time_array, pe_array, coeff_pe, coeff_time, cart),
            expect = True)
        
        reconout['EventID'] = sid
        reconout['E'] = E_out
        reconout['x'], reconout['y'], reconout['z'] = result_out.x[:3]*shell
        reconout['success'] = result_out.success
        reconout['Likelihood'] = result_out.fun
        
        print('-'*60)
        print('inner')
        print('%d vertex: [%+.2f, %+.2f, %+.2f] radius: %+.2f, energy: %.2f, Likelihood: %+.6f' 
            % (sid, reconin['x'], reconin['y'], reconin['z'], 
                np.sqrt(reconin['x']**2 + reconin['y']**2 + reconin['z']**2), E_in, result_in.fun))
        print('outer')
        print('%d vertex: [%+.2f, %+.2f, %+.2f] radius: %+.2f, energy: %.2f, Likelihood: %+.6f' 
            % (sid, reconout['x'], reconout['y'], reconout['z'], 
                np.sqrt(reconout['x']**2 + reconout['y']**2 + reconout['z']**2), E_out, result_out.fun))
        reconin.append()
        reconout.append()

    # Flush into the output file
    ReconInTable.flush()
    ReconOutTable.flush()
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
