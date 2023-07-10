'''
# Convert ROOT file to HDF5 file, For simulation files
'''
import numpy as np
import ROOT
import sys
import os
import tables
import uproot
print(sys.argv)
baseFilename = sys.argv[1]

WindowSize=1029
# Define the database columns
class WaveformTable(tables.IsDescription):
    EventID = tables.Int64Col(pos=0)
    ChannelID = tables.Int16Col(pos=1)
    Waveform = tables.Col.from_type('int16', shape=WindowSize, pos=2)

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

WaveformTable = h5file.create_table(group, "WaveformTable", WaveformTable, "WaveformTable")
waveform = WaveformTable.row
#
fip = uproot.open(baseFilename)
ReadoutTree = fip["Readout"]
Waveform = np.array(ReadoutTree["Waveform"].array(flatten=True), dtype=np.int16)
ChannelID = np.array(ReadoutTree["ChannelId"].array(flatten=True), dtype=np.int16)
EventID = np.array(ReadoutTree["TriggerNo"].array(), dtype=np.int64)
print(Waveform.shape, ChannelID.shape, EventID.shape)

nWave = len(ChannelID)
Waveform = Waveform.reshape(nWave, WindowSize)
nChannels = Len(np.array(ReadoutTree["ChannelId"].array()))

for i in range(nWave) :
    if(ChannelID[i] == 17) :
        waveform['EventID'] = EventID[i]
        waveform['ChannelID'] = ChannelID[i]
        waveform['Waveform'] = Waveform[i]
        waveform.append()

# Flush into the output file
WaveformTable.flush()

h5file.close()
