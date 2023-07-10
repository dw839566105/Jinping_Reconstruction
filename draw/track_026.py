import tables
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import uproot
from tqdm import *
from matplotlib import gridspec

path = '/mnt/stage/douwei/JP_1t_paper/track/'

file = 'shell/2/0.26.h5'
filename = path + file

with tables.open_file(filename) as h:
    Step1 = pd.DataFrame(h.root.SimTriggerInfo.Track.StepPoint[:])
    TrackList1 = pd.DataFrame(h.root.SimTriggerInfo.Track.TrackList[:])
    Truth1 = pd.DataFrame(h.root.SimTruth.SimTruth[:])

data = []
data_test = []
df = pd.merge(Step1, TrackList1,
                   on=['SegmentId','TrackId', 'VertexId', 'RunNo'])
for seg in tqdm(np.unique(df['SegmentId'])):
    df_tmp = df[df['SegmentId'] == seg]
    for tc in np.unique(df_tmp['TrackId']):
        df_sub = df_tmp[df_tmp['TrackId'] == tc]
        A = df_sub[['fX', 'fY', 'fZ']].values
        cth = np.clip((A*A[0]).sum(1)/np.linalg.norm(A, axis=1)/np.linalg.norm(A[0]), -1, 1)
        if cth[-1] < -0.99:
            data.append(np.hstack((A[0], A[1], A[-1])))
        if cth[-1] > 0.99:
            data_test.append(np.hstack((A[0], A[1], A[-1])))

dt = np.vstack((data))
cth = (dt[:, 3:6] * dt[:,:3]).sum(1) / \
    np.linalg.norm(dt[:,3:6], axis=1) / np.linalg.norm(dt[:,:3], axis=1)

fig = plt.figure(dpi=300, figsize=(6,4.5))


spec = gridspec.GridSpec(ncols=1, nrows=2,
                         height_ratios=[2, 1.5])
                         
ax0 = fig.add_subplot(spec[1])
dt = np.vstack((data))
cth = (dt[:, 3:6] * dt[:,:3]).sum(1) / \
    np.linalg.norm(dt[:,3:6], axis=1) / np.linalg.norm(dt[:,:3], axis=1)
plt.hist(cth, bins=100, color='k',
         histtype='step', label=r'PMT1')

dt = np.vstack((data_test))
cth = (dt[:, 3:6] * dt[:,:3]).sum(1) / \
    np.linalg.norm(dt[:,3:6], axis=1) / np.linalg.norm(dt[:,:3], axis=1)
ax0.hist(cth, bins=100, color='b', 
         histtype='step', label=r'PMT2')
ax0.legend(loc=9, fontsize=16)
ax0.semilogy()
ax0.set_xlabel(r'$\cos\theta_\mathrm{acr}$')

ax1 = fig.add_subplot(spec[0])
x = np.linspace(0, np.pi, 100)
ax1.plot(-np.cos(x), np.sin(x), color='k', ls='--', label='shell')
ax1.set_aspect(1)
v0 = np.array((0, 0.26/0.65))
v1 = np.array((0, 0.83/0.65))
ax1.scatter(v0[1], v0[0], color='r', s=100, marker='*', label='vertex')
ax1.scatter(-v1[1], v1[0], marker='^', s=50, color='k', label='PMT1')
ax1.scatter(v1[1], v1[0], marker='^', s=50, color='b', label='PMT2')
for i in np.arange(5, 30, 5):
    if i < 25:
        alpha=0.2
    else:
        alpha=0.6
    ax1.plot([v0[1], np.cos(x[i])], [v0[0], np.sin(x[i])], color='k', alpha=alpha)
    ax1.plot([np.cos(x[i]), -v1[1]], [np.sin(x[i]), v1[0]], color='k', alpha=alpha)
    
ax1.plot([np.cos(x[i]), 0.75], [np.sin(x[i]), 1], color='k', alpha=alpha)

ax1.plot([v0[1], -v1[1]], [v0[0], v1[0]], color='k', alpha = 1, label='track')
ax1.plot([v0[1], v1[1]], [v0[0], v1[0]], color='b', alpha = 1)
ax1.plot(0.15*np.cos(x[:25]), 0.15*np.sin(x[:25]), color='k', lw=1, ls='-')
ax1.plot([0, np.cos(x[25])], ([0, np.sin(x[25])]), color='k', lw=1, ls='-')
ax1.text(0.1, 0.1, r'$\theta_{\mathrm{acr}}$',fontsize=20)
ax1.legend(loc=1, fontsize=18)
ax1.set_xlabel(r'$x/R_\mathrm{LS}$')
ax1.set_ylabel(r'$y/R_\mathrm{LS}$')
ax1.set_xticks([-1,0,1],[-1,0,1])
plt.xlim(-1.5,2.8)
plt.ylim(-0.1,1.1)
plt.tight_layout()
plt.savefig('track_026_1.pdf')