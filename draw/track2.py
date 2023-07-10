import tables
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

path = '/mnt/stage/douwei/JP_1t_paper/track/'

with PdfPages('track2.pdf') as pp:
    for index, file in enumerate(['point/z/2/0.62.h5', 'shell/2/0.62.h5', 'shell/2/0.26.h5']):
        filename = path + file
        with tables.open_file(filename) as h:
            Step = pd.DataFrame(h.root.SimTriggerInfo.Track.StepPoint[:])
            TrackList = pd.DataFrame(h.root.SimTriggerInfo.Track.TrackList[:])
            Truth = pd.DataFrame(h.root.SimTruth.SimTruth[:])

        df = pd.merge(pd.merge(Step, TrackList,
                               on=['SegmentId','TrackId', 'VertexId', 'RunNo']), Truth, 
                      on=['SegmentId', 'VertexId'])
        if (index == 2) or (index == 0):
            fig = plt.figure(dpi=200, tight_layout=True)
            ax = plt.subplot(1,1,1, projection='polar', theta_offset=np.pi/2)
            ax.set_ylim(0, 800)
        for j in np.unique(df['SegmentId']):
            if index == 2:
                if j > 10:
                    break
            else:
                if j > 5:
                    break
            for i in np.unique(df['TrackId']):
                data = df[(df['SegmentId'] == i) & (df['TrackId']==j) & (df['PdgId']==0)]
                if len(data) > 0:
                    x = data.fX.values/1000
                    y = data.fY.values/1000
                    z = data.fZ.values/1000
                    vec = np.vstack((x, y, z)).T
                    cth = np.clip((vec*vec[0]).sum(axis=1)/np.linalg.norm(vec, axis=1) / np.linalg.norm(vec[0]), -1, 1)
                    rs = np.linalg.norm(vec, axis=1)

                    if index == 2:
                        if (rs[-2] < 0.65) & (rs[-2] > 0.64) & (cth[-2]<-0.99):
                            p0, = ax.plot(np.arccos(cth), rs*1000, 
                                          'ro-',  markersize=3, alpha=0.5, label='track')
                            p0, = ax.plot(np.pi*2 - np.arccos(cth), rs*1000, 
                                          'ro-',  markersize=3, alpha=0.5, label='track')
                    else:
                        if rs[-1]>0.70:
                            if index == 0:
                                p0, = ax.plot(np.arccos(cth), rs*1000, 'ro-',  markersize=3, alpha=0.1, label='track')
                            elif index == 1:
                                p0, = ax.plot(np.pi*2 - np.arccos(cth), rs*1000, 'ro-',  markersize=3, alpha=0.1, label='track')

        if index == 0:
            p2, = ax.plot([np.arcsin(80/650), np.arcsin(80/800)], [650, 800], 
                color='blue', alpha=0.5, linestyle='--', label='outlet')

        ns = np.linspace(-np.pi, np.pi, 100)
        p3, = ax.plot(ns, np.full_like(ns, 650),
               color='blue', alpha=0.5, linestyle='-', label=r'$R_\mathrm{LS}$')
        p1, = ax.plot(ns, np.full_like(ns, 650),
               color='blue', alpha=0.5, linestyle='-', label=r'$R_\mathrm{PMT}$')
        if index == 2:
            p4 = ax.scatter(0, 260, marker='*', s=100,  
                       color='black', label='vertex')
        else:
            p4 = ax.scatter(0, 620, marker='*', s=100,  
                       color='black', label='vertex')
        if index == 0:
            leg = ax.legend(handles=[p0, p1, p2, p3, p4], bbox_to_anchor=(1.9, 0.27), loc = 'center right')
        else:
            leg = ax.legend(handles=[p0, p1, p3, p4], bbox_to_anchor=(1.9, 0.23), loc = 'center right')
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
        
        if (index == 1) or (index == 2):
            pp.savefig(fig)
