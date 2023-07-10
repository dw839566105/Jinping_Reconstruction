import tables
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import uproot

plt.figure(dpi=300)
path = '/mnt/stage/douwei/JP_1t_paper/track/'

file = 'point/z/2/0.62.h5'
filename = path + file
with tables.open_file(filename) as h:
    Step = pd.DataFrame(h.root.SimTriggerInfo.Track.StepPoint[:])
    TrackList = pd.DataFrame(h.root.SimTriggerInfo.Track.TrackList[:])
    Truth = pd.DataFrame(h.root.SimTruth.SimTruth[:])

rs = np.sqrt(Step['fX']**2 + Step['fY']**2 + Step['fZ']**2)
#index = (((rs > 655)[1:]).values & ((rs < 655)[:-1]).values)
index = ((rs[:-3]<645).values & (rs[1:-2]>640).values & (rs[2:-1]<651).values & (rs[3:]> 660).values)
v = Step[['SegmentId', 'TrackId']].values
index_ = (np.hstack((1, np.diff(v, axis=0).sum(1)))!=0)
df_ = pd.merge(pd.merge(Step[2:-1][index], Step[index_],
                       on=['SegmentId','TrackId', 'VertexId', 'RunNo']), TrackList,
                   on=['SegmentId','TrackId', 'VertexId', 'RunNo'])
x = df_['fX_x'][2:-1]
y = df_['fY_x'][2:-1]
z = df_['fZ_x'][2:-1]
'''
plt.hist(z/np.sqrt(x**2+y**2+z**2), color='k', 
         bins = np.linspace(0.95, 1, 100), 
         histtype='step', 
         # alpha=0.3,
         weights=np.full_like(z, 1/len(np.unique(Step['SegmentId']))),
        label='Outlet(detect + undetect)')
'''
x = df_['fX_x'][2:-1][df_['bDetectedPhoton']==1]
y = df_['fY_x'][2:-1][df_['bDetectedPhoton']==1]
z = df_['fZ_x'][2:-1][df_['bDetectedPhoton']==1]

cth = (df_['fX_x'][3:] * df_['fX_x'][:-3] + 
df_['fY_x'][3:] * df_['fY_x'][:-3] +
df_['fZ_x'][3:] * df_['fZ_x'][:-3]) / np.sqrt(df_['fX_x'][3:]**2 + df_['fY_x'][3:]**2 + df_['fZ_x'][3:]**2) / np.sqrt(df_['fX_x'][:-3] + df_['fY_x'][:-3]**2 + df_['fZ_x'][:-3]**2)

breakpoint()
plt.hist(z/np.sqrt(x**2+y**2+z**2), color='g', 
         histtype='step',
         bins = np.linspace(0.98, 1, 100), 
         alpha=1,
         weights=np.full_like(z, 1/len(np.unique(df_['SegmentId'][df_['bDetectedPhoton']==1]))),
        label='Outlet')
#plt.axvline(np.arcsin(80/650), color='k', label='supprot')
#plt.axvline(np.pi - np.arcsin(100/650), color='k', label='base')

file = 'shell/2/0.62.h5'
filename = path + file

with tables.open_file(filename) as h:
    Step1 = pd.DataFrame(h.root.SimTriggerInfo.Track.StepPoint[:])
    TrackList1 = pd.DataFrame(h.root.SimTriggerInfo.Track.TrackList[:])
    Truth1 = pd.DataFrame(h.root.SimTruth.SimTruth[:])

rs = np.sqrt(Step1['fX']**2 + Step1['fY']**2 + Step1['fZ']**2)
# index1 = (((rs > 655)[1:]).values & ((rs < 655)[:-1]).values)
index1 = ((rs[:-3]<645).values & (rs[1:-2]>640).values & (rs[2:-1]<651).values & (rs[3:]> 660).values)
v = Step1[['SegmentId', 'TrackId']].values
index2 = (np.hstack((1, np.diff(v, axis=0).sum(1)))!=0)
df_test = pd.merge(pd.merge(Step1[2:-1][index1], Step1[index2],
                       on=['SegmentId','TrackId', 'VertexId', 'RunNo']), TrackList1,
                   on=['SegmentId','TrackId', 'VertexId', 'RunNo'])

x1 = df_test['fX_x']
y1 = df_test['fY_x']
z1 = df_test['fZ_x']
x2 = df_test['fX_y']
y2 = df_test['fY_y']
z2 = df_test['fZ_y']
cth = np.clip((x1*x2+y1*y2+z1*z2)/np.sqrt(x1**2+y1**2+z1**2)/np.sqrt(x2**2+y2**2+z2**2), -1, 1)
'''
plt.hist(cth, bins=np.linspace(0.95, 1, 100), color='b',
         histtype='step', 
         #alpha=0.3,
         weights=np.full_like(z2, 1/len(np.unique(Step1['SegmentId']))),
        label='General(detect + undetect)')
'''
x1 = df_test['fX_x'][df_test['bDetectedPhoton']==1]
y1 = df_test['fY_x'][df_test['bDetectedPhoton']==1]
z1 = df_test['fZ_x'][df_test['bDetectedPhoton']==1]
x2 = df_test['fX_y'][df_test['bDetectedPhoton']==1]
y2 = df_test['fY_y'][df_test['bDetectedPhoton']==1]
z2 = df_test['fZ_y'][df_test['bDetectedPhoton']==1]
cth = np.clip((x1*x2+y1*y2+z1*z2)/np.sqrt(x1**2+y1**2+z1**2)/np.sqrt(x2**2+y2**2+z2**2), -1, 1)

plt.hist(cth, bins=np.linspace(0.98,1,100), color='r',
         histtype='step',
         alpha=1,
         weights=np.full_like(z2, 1/len(np.unique(df_test['SegmentId'][df_test['bDetectedPhoton']==1]))),
        label='General')

plt.axvline(np.cos(np.arcsin(80/650)), alpha=0.3,color='k', linestyle='dashed', label='Outlet region')
#plt.fill_between([0,np.arcsin(80/650)], [0.05,0.05], [200,200], alpha=0.3,color='red', label='Outlet region')
#plt.fill_between([np.pi - np.arcsin(100/650), np.pi], [0.05,0.05], [200,200], alpha=0.1,color='red')

plt.xlim(0.98,1)
#plt.ylim(0.05,200)
plt.semilogy()

plt.legend()
plt.xlabel(r'$\cos\theta_\mathrm{acr}$')
plt.ylabel(r'Photon Counts')
plt.savefig('track_hist.pdf')