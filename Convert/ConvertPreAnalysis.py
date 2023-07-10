'''
For later analysis of real data
'''

import numpy as np
import scipy, h5py
import scipy.stats as stats
import math
import ROOT
import os,sys
import tables
import uproot, argparse
import scipy.io as scio
from scipy.optimize import minimize
from scipy import interpolate
from numpy.polynomial import legendre as LG
from scipy import special


def Convert(fid, fout):
    '''
    convert root files into h5

    fid: root reference file
    fout: output file
    '''
    global event_count
    # Create the output file and the group
    rootfile = ROOT.TFile(fid)
    class RawData(tables.IsDescription):
        EventID = tables.Int64Col(pos=0)
        ChannelID = tables.Int64Col(pos=1)
        PE = tables.Int64Col(pos=2)
        Time = tables.Int64Col(pos=3)

    # Create the output file and the group
    h5file = tables.open_file(fout, mode="w", title="OneTonDetector",
                    filters = tables.Filters(complevel=9))
    group = "/"
    # Create tables
    RawTable = h5file.create_table(group, "RawData", RawData, "RawData")
    rawdata = RawTable.row
    # Loop for event
    f = uproot.open(fid)
    a = f['SimpleAnalysis']
    
    for tot, chl, PEl, Pkl in zip(a.array("TotalPE"),  # total pe in an event
                a.array("ChannelInfo.ChannelId"),       # PMT fired seq
                a.array('ChannelInfo.PE'),              # Hit info number on PMT
                a.array('ChannelInfo.PeakLoc')):         # Time info on PMT
        pe_array = np.zeros(30) # Photons on each PMT (PMT size * 1 vector)
        event_count = event_count + 1
        time_array = np.zeros(0, dtype=int)    # Time info (Hit number)
        for ch, pe, pk in zip(chl, PEl, Pkl):
            '''
            rawdata['EventID'] = event_count
            rawdata['ChannelID'] = ch
            rawdata['PE'] = pe
            rawdata.append()
            '''
            for i in np.arange(0,np.size(pk)):
                rawdata['EventID'] = event_count
                rawdata['ChannelID'] = ch
                rawdata['PE'] = pe
                rawdata['Time'] = pk[i]
                rawdata.append()
            
            #rawdata['Time'] = pk
            '''
            pe_array[ch] = pe
            time_array = np.hstack((time_array, pk))
            fired_PMT = np.hstack((fired_PMT, ch*np.ones(np.size(pk))))
            '''
        '''
        fired_PMT = fired_PMT.astype(int)
        
        # initial result
        result_vertex = np.empty((0,6)) # reconstructed vertex
        # initial value x[0] = [1,6]
        rawdata['EventID'] = event_count
        rawdata['ChannelID'] = fired_PMT
        rawdata['PE'] = pe_array
        rawdata['Time'] = time_array
        '''
    # Flush into the output file
    RawTable.flush()
    h5file.close()

# Automatically add multiple root files created a program with max tree size limitation.
#if len(sys.argv)!=2:
#    print("Wront arguments!")
#    print("Usage: python Recon.py MCFileName[.root] outputFileName[.h5]")
#    sys.exit(1)
# Read PMT position
global event_count
event_count = 0
#ROOT.PyConfig.IgnoreComman:dLineOptions = True
# Reconstruction
fid = sys.argv[1] # input file .root
fout = sys.argv[2] # output file .h5
Convert(fid, fout)
