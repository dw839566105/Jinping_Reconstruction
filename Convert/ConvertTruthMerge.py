'''
# Convert ROOT file to HDF5 file, For simulation files
'''
import numpy as np
import ROOT
import sys
import os
import tables

# Define the database columns
class TruthData(tables.IsDescription):
    E = tables.Float64Col(pos=0)
    x = tables.Float64Col(pos=1)
    y = tables.Float64Col(pos=2)
    z = tables.Float64Col(pos=3)
    px = tables.Float64Col(pos=4)
    py = tables.Float64Col(pos=5)
    pz = tables.Float64Col(pos=6)

class GroundTruthData(tables.IsDescription):
    EventID    = tables.Int64Col(pos=0)
    ChannelID  = tables.Int64Col(pos=1)
    PETime     = tables.Float64Col(pos=2)
    photonTime = tables.Float64Col(pos=3)
    PulseTime = tables.Float64Col(pos=4)
    dETime = tables.Float64Col(pos=5)

class PETruthData(tables.IsDescription):
    Q = tables.Int64Col(pos=1)

# Automatically add multiple root files created a program with max tree size limitation.
if len(sys.argv)!=3:
    print("Wront arguments!")
    print("Usage: python ConvertTruth.py MCFileName outputFileName")
    sys.exit(1)

baseFileName = sys.argv[1]
outputFileName = sys.argv[2]

ROOT.PyConfig.IgnoreCommandLineOptions = True

FileNo = 0

# Create the output file and the group
h5file = tables.open_file(outputFileName, mode="w", title="OneTonDetector",
                          filters = tables.Filters(complevel=9))
group = "/"

# Create tables

GroundTruthTable = h5file.create_table(group, "GroundTruth", GroundTruthData, "GroundTruth")
groundtruth = GroundTruthTable.row
TruthData = h5file.create_table(group, "TruthData", TruthData, "TruthData")
truthdata = TruthData.row
PETruthData = h5file.create_table(group, "PETruthData", PETruthData, "PETruthData")
PEtruthdata = PETruthData.row
# Loop for ROOT files. 
t = ROOT.TChain("Readout")
tTruth = ROOT.TChain("SimTriggerInfo")
base = baseFileName[0:-5]
for count in np.arange(30):
    if (count == 0):
        t.Add(base + '.root')
        tTruth.Add(base + '.root')
    else:
        try:
            t.Add(base + '_%d.root' % count)
            tTruth.Add(base + '_%d.root' % count)
            print(base + '_%d.root' % count)
        except:
            break
# Loop for event
cnt = 0
for event in tTruth:
    if(len(event.PEList)==0):
        pass
    else:
        for truthinfo in event.truthList:
            truthdata['E'] =  truthinfo.EkMerged
            truthdata['x'] =  truthinfo.x
            truthdata['y'] =  truthinfo.y
            truthdata['z'] =  truthinfo.z
            for px in truthinfo.PrimaryParticleList:
                truthdata['px'] = px.px
                truthdata['py'] = px.py
                truthdata['pz'] = px.pz
        truthdata.append()
        Q = []

        for PE in event.PEList:
            Q.append(PE.PMTId)
            groundtruth['EventID'] = event.TriggerNo
            groundtruth['ChannelID'] = PE.PMTId
            groundtruth['PETime'] =  PE.HitPosInWindow
            groundtruth['photonTime'] = PE.photonTime
            groundtruth['PulseTime'] = PE.PulseTime
            groundtruth['dETime'] = PE.dETime
            groundtruth.append()
        PEs = np.zeros(30)
        PEs[0:np.max(Q)+1] = np.bincount(np.array(Q))
        #print(PE[0:np.max(Q)+1].shape)
        #print(np.bincount(np.array(Q)).shape)
        for i in np.arange(30):
            PEtruthdata['Q'] = PEs[i]
            PEtruthdata.append()
# Flush into the output file
GroundTruthTable.flush()

h5file.close()
