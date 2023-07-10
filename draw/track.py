import tables
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + np.nan_to_num(kmat.dot(kmat) * ((1 - c) / (s ** 2)))
    return rotation_matrix

path = '/mnt/stage/douwei/JP_1t_paper/track/'

with PdfPages('track.pdf') as pp:
    for index, file in enumerate(['point/z/2/0.62.h5', 'shell/2/0.62.h5', 'shell/2/0.26.h5']):
        filename = path + file
        with tables.open_file(filename) as h:
            Step = pd.DataFrame(h.root.SimTriggerInfo.Track.StepPoint[:])
            TrackList = pd.DataFrame(h.root.SimTriggerInfo.Track.TrackList[:])
            Truth = pd.DataFrame(h.root.SimTruth.SimTruth[:])

        df = pd.merge(pd.merge(Step, TrackList,
                               on=['SegmentId','TrackId', 'VertexId', 'RunNo']), Truth, 
                      on=['SegmentId', 'VertexId'])

        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(1, 1, 1)
        t = np.linspace(0, np.pi, 100)
        vec0 = np.array((0,0,1))
        for i in np.unique(df['TrackId']):
            if index == 2:
                if i == 200:
                    break
            else:
                if i == 50:
                    break
            
            for j in np.unique(df['SegmentId']):
                data = df[(df['SegmentId'] == i) & (df['TrackId']==j) & (df['PdgId']==0)]
                if len(data) > 0:
                    x = data.fX.values/1000
                    y = data.fY.values/1000
                    z = data.fZ.values/1000
                    r = np.sqrt(x[-1]**2 + y[-1]**2 +z[-1]**2)
                    vec1 = np.array((data.x.values[0], data.y.values[0], data.z.values[0]))
                    rot = rotation_matrix_from_vectors(vec0, vec1)
                    step = np.vstack((x, y, z)).T.dot(rot)
                    if index == 2:
                        if ((step[-2, -1]>-0.65) & (step[-2, -1]<-0.64) & (np.sqrt(step[-2, 0]**2 + step[-2, 1]**2) < 0.05)):
                        
                            if ((r<0.80) & (r>0.70)):
                                p0, = ax.plot(1000*np.sqrt(step[:,0]**2 + step[:,1]**2), 
                                         1000*step[:,2], 'ro-',  markersize=3, alpha=0.5, label='track')
                                p1 = ax.scatter(1000*np.sqrt(step[-1,0]**2 + step[-1,1]**2), 
                                         1000*step[-1,2], c='b', marker='o', s=3, alpha=0.75, label='PMT')
                    else:
                        if ((r<0.80) & (r>0.70)):
                            p0, = ax.plot(1000*np.sqrt(step[:,0]**2 + step[:,1]**2), 
                                     1000*step[:,2], 'ro-',  markersize=3, alpha=0.1, label='track')
                            p1 = ax.scatter(1000*np.sqrt(step[-1,0]**2 + step[-1,1]**2), 
                                     1000*step[-1,2], c='b', marker='o', s=3, alpha=0.35, label='PMT')                    
        ax = plt.gca()
        ax.set_aspect(1)
        ax.set_xlabel(r'$\sqrt{x^2+y^2}$/mm')
        ax.set_ylabel('$z$/mm')
        ax.tick_params(axis = 'both', which = 'major')
        ax.set_xlim([-0.05*1000, 0.8*1000])
        ax.set_ylim([-0.83*1000, 0.83*1000])
        if index == 0:
            p2, = ax.plot([0.08*1000, 0.08*1000], [0.64*1000, 0.83*1000], color='blue', alpha=0.5, linestyle='--', label='outlet')

        p3, = ax.plot(np.sin(t)*650, np.cos(t)*650, alpha=0.5, linewidth=1, 
                       color='blue', linestyle='-', label='acrylic')
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
            
        pp.savefig(fig)
