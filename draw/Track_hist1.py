import tables
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import uproot
from tqdm import *

plt.figure(dpi=300)
path = '/mnt/stage/douwei/JP_1t_paper/track/'

file = 'point/z/2/0.62.h5'
filename = path + file
with tables.open_file(filename) as h:
    Step = pd.DataFrame(h.root.SimTriggerInfo.Track.StepPoint[:])
    TrackList = pd.DataFrame(h.root.SimTriggerInfo.Track.TrackList[:])
    Truth = pd.DataFrame(h.root.SimTruth.SimTruth[:])

select = TrackList[['SegmentId', 'TrackId']][TrackList['bDetectedPhoton']==1].values

th0 = []
cth = []
for i in trange(len(select)):
    block = (Step['SegmentId'] == select[i,0]) & (Step['TrackId'] == select[i,1])
    event = Step[block]
    x0 = event.iloc[0][['fX', 'fY', 'fZ']].values
    x1 = event.iloc[-1][['fX', 'fY', 'fZ']].values
    th = (x0*x1).sum() / np.linalg.norm(x0) / np.linalg.norm(x1)
    if (th > 0.75) & (th < 0.95):
        x1 = event.iloc[1][['fX', 'fY', 'fZ']].values
        if np.linalg.norm(x1) > 644:
            if (np.linalg.norm(event[['fX', 'fY', 'fZ']].values, axis=1)<760).all():
                th = (x0*x1).sum() / np.linalg.norm(x0) / np.linalg.norm(x1)
                cth.append(th)

cth = np.hstack(cth)
plt.figure()
# plt.hist(cth, np.linspace(0.98,1,50), histtype='step', alpha=0.9, label='Outlet')
plt.hist(np.arccos(cth), np.linspace(0, np.pi/15, 50), histtype='step', alpha=0.9, label='Outlet')

file = 'shell/2/0.62.h5'
filename = path + file
with tables.open_file(filename) as h:
    Step = pd.DataFrame(h.root.SimTriggerInfo.Track.StepPoint[:])
    TrackList = pd.DataFrame(h.root.SimTriggerInfo.Track.TrackList[:])
    Truth = pd.DataFrame(h.root.SimTruth.SimTruth[:])

select = TrackList[['SegmentId', 'TrackId']][TrackList['bDetectedPhoton']==1].values

cth = []
for i in trange(len(select)):
    block = (Step['SegmentId'] == select[i,0]) & (Step['TrackId'] == select[i,1])
    event = Step[block]
    x0 = event.iloc[0][['fX', 'fY', 'fZ']].values
    x1 = event.iloc[-1][['fX', 'fY', 'fZ']].values
    th = (x0*x1).sum() / np.linalg.norm(x0) / np.linalg.norm(x1)
    if (th > 0.75) & (th < 0.95):
        th0.append(x1)
        if np.linalg.norm(x1) > 644:
            if (np.linalg.norm(event[['fX', 'fY', 'fZ']].values, axis=1)<760).all():
                x1 = event.iloc[1][['fX', 'fY', 'fZ']].values
                th = (x0*x1).sum() / np.linalg.norm(x0) / np.linalg.norm(x1)
                cth.append(th)

cth = np.hstack(cth)
# plt.hist(cth, np.linspace(0.98,1,50), histtype='step', alpha=0.9, label='General')
plt.hist(np.arccos(cth), np.linspace(0, np.pi/15, 50), histtype='step', alpha=0.9, label='General')
#plt.axvline(np.cos(np.arcsin(40/650)), alpha=0.3,color='k', linestyle='dashed', label='Outlet region')
#plt.axvline(np.cos(np.arcsin(80/650)), alpha=0.3,color='k', linestyle='dashed', label='Outlet region')
#plt.axvline(np.arcsin(40/650), alpha=0.3, color='k', linestyle='dashed', label='Outlet region')
plt.axvline(np.arcsin(60/650), alpha=0.3, color='k', linestyle='dashed')
#plt.xlim(0.98,1)
plt.xlim(0, 0.15)
plt.semilogy()
plt.legend(loc=6)
plt.xticks([0,0.05,0.10,0.15])
plt.xlabel(r'$\theta_\mathrm{acr}/\mathrm{rad}$')
plt.ylabel(r'Photon Counts')
plt.savefig('track_hist.pdf')