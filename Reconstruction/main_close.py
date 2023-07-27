# recon range: [-1,1], need * detector radius
import numpy as np
import tables
import uproot3 as uproot
import awkward as ak
import argparse
from argparse import RawTextHelpFormatter
from scipy.optimize import minimize
from zernike import RZern
import pub_close as pub
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

    fid: root reference file convert to .h5
    fout: output file
    '''
    # Create the output file and the group
    print(filename) # filename
    # Create the output file and the group
    h5file = tables.open_file(output, mode="w", title="OneTonDetector",
                            filters = tables.Filters(complevel=9))
    group = "/"
    # Create tables
    TruthTable = h5file.create_table(group, "Truth", pub.Recon, "Recon")
    truth = TruthTable.row
    ReconWATable = h5file.create_table(group, "ReconWA", pub.Recon, "Recon")
    reconwa = ReconWATable.row
    ReconInTable = h5file.create_table(group, "ReconIn", pub.Recon, "Recon")
    reconin = ReconInTable.row
    ReconOutTable = h5file.create_table(group, "ReconOut", pub.Recon, "Recon")
    reconout = ReconOutTable.row
    # Loop for event
    f = uproot.open(filename)
    data = f['SimTriggerInfo']

    PMTId = data['PEList.PMTId'].array()
    # Time = data['PEList.HitPosInWindow'].array()
    Time = data['PEList.PulseTime'].array()
    Charge = data['PEList.Charge'].array()   
    SegmentId = ak.to_numpy(ak.flatten(data['truthList.SegmentId'].array()))
    VertexId = ak.to_numpy(ak.flatten(data['truthList.VertexId'].array()))
    x = ak.to_numpy(ak.flatten(data['truthList.x'].array()))
    y = ak.to_numpy(ak.flatten(data['truthList.y'].array()))
    z = ak.to_numpy(ak.flatten(data['truthList.z'].array()))
    E = ak.to_numpy(ak.flatten(data['truthList.EkMerged'].array()))
    
    for pmt, time_array, pe_array, sid, xt, yt, zt, Et in zip(PMTId, Time, Charge, SegmentId, x, y, z, E):
        truth['x'], truth['y'], truth['z'], truth['E'], truth['EventID'] = xt, yt, zt, Et, sid
        fired_PMT = ak.to_numpy(pmt)
        time_array = ak.to_numpy(time_array)
        # PMT order: 0-29
        # PE /= Gain
        # For charge info
        # pe_array, cid = np.histogram(pmt, bins=np.arange(31)-0.5, weights=PE)
        # For hit info
        pe_array, cid = np.histogram(fired_PMT, bins=np.arange(len(PMT_pos)+1)) 
        # For very rough estimate from charge to PE
        # pe_array = np.round(pe_array)

        if np.sum(pe_array)==0:
            continue
        
        ############################################################################
        ###############               inner recon                  #################
        ############################################################################
        x0_in = pub.Initial.FitGrid(pe_array, MeshIn.mesh, MeshIn.tpl, time_array)
        # x0_in = np.ones(5)*0.1
        reconwa['E'], reconwa['x'], reconwa['y'], reconwa['z'], reconwa['t'] = x0_in
        reconwa['EventID'] = sid
        index = np.arange(30)!=11
        result_in = minimize(LH.Likelihood, x0_in[1:], method='SLSQP', 
            bounds=((-1, 1), (-1, 1), (-1, 1), (None, None)),
            args = (PMT_pos[index], fired_PMT, time_array, pe_array[index], coeff_pe, coeff_time, cart))
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
        # x0_out = np.ones(5)*0.1
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
        print('Truth:', np.array((xt, yt, zt))/1000)
        print('inner')
        print('%d vertex: [%+.2f, %+.2f, %+.2f] radius: %+.2f, energy: %.2f, Likelihood: %+.6f' 
            % (sid, reconin['x'], reconin['y'], reconin['z'], 
                np.sqrt(reconin['x']**2 + reconin['y']**2 + reconin['z']**2), E_in, result_in.fun))
        print('outer')
        print('%d vertex: [%+.2f, %+.2f, %+.2f] radius: %+.2f, energy: %.2f, Likelihood: %+.6f' 
            % (sid, reconout['x'], reconout['y'], reconout['z'], 
                np.sqrt(reconout['x']**2 + reconout['y']**2 + reconout['z']**2), E_out, result_out.fun))
        truth.append()
        reconin.append()
        reconout.append()

    # Flush into the output file
    ReconInTable.flush()
    ReconOutTable.flush()
    TruthTable.flush()
    h5file.close()

parser = argparse.ArgumentParser(description='Process Reconstruction construction', formatter_class=RawTextHelpFormatter)
parser.add_argument('-f', '--filename', dest='filename', metavar='filename[*.h5]', type=str,
                    help='The filename [*Q.h5] to read')

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
if pe_type=='Zernike':
    cart = RZern(30)
    LH = pub.LH_Zer
    MeshIn = pub.construct_Zer(coeff_pe, PMT_pos, np.linspace(0.01, 0.92, 3), cart)
    MeshOut = pub.construct_Zer(coeff_pe, PMT_pos, np.linspace(0.92, 1, 3), cart)
elif pe_type == 'Legendre':
    LH = pub.LH_Leg
    MeshIn = pub.construct_Leg(coeff_pe, PMT_pos, np.linspace(0.01, 0.92, 30))
    MeshOut = pub.construct_Leg(coeff_pe, PMT_pos, np.linspace(0.92, 1, 30))
Recon(args.filename, args.output)
