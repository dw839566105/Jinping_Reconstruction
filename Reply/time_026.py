import matplotlib as mpl
import seaborn as sns
mpl.use('pdf')
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as LG

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
import tables
import numpy as np
from tqdm import tqdm
import sys
output = sys.argv[1]

plt.rcParams['lines.markersize'] = 5

fig = plt.gcf()
ax1 = plt.gca()

vec0 = np.array((0,0,260))

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
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

import ROOT
t = ROOT.TChain("SimTriggerInfo")
path = '/mnt/stage/douwei/JP_1t_paper/add/shell/2/0.26/'
for i in np.arange(0,19):
    #if i == 0:
    #    t.Add(path + '0.26.root')
    #else:
    #    t.Add(path + '0.26_%d.root' % i)
    # t.add(path + '%03d.root' % i)
        
x_total2 = []
y_total2 = []
z_total2 = []
t_total2 = []
track_total2 = []
for event in tqdm(t):
    for info in event.truthList:        
        if not (info.SegmentId)%100:
            print(info.SegmentId)
        x = []
        y = []
        z = []
        t = []
        trackid = []
        for track in info.trackList:               
            for step in track.StepPoints:
                if step.bDetectedPhoton:
                    trackid.append(track.nTrackId)
                    x.append(step.fX)
                    y.append(step.fY)
                    z.append(step.fZ)
                    t.append(step.fTime)
        x_total2.append(x)
        y_total2.append(y)
        z_total2.append(z)
        t_total2.append(t)
        track_total2.append(trackid)

radius = []
time = []
for i in tqdm(np.arange(len(x_total2))):
    for j in np.unique(track_total2[i]):
        index = (np.array(track_total2[i]) == j)
        block = np.vstack((np.array(x_total2[i])[index], np.array(y_total2[i])[index], np.array(z_total2[i])[index])).T
        vec1 = np.array((np.array(x_total2[i])[index][0], np.array(y_total2[i])[index][0], np.array(z_total2[i])[index][0]))
        vec2 = np.array((np.array(x_total2[i])[index][-1], np.array(y_total2[i])[index][-1], np.array(z_total2[i])[index][-1]))
        tmp = t_total2[i]
        theta = np.sum(vec1*vec2)/np.linalg.norm(vec1)/np.linalg.norm(vec2)
        if((theta<-0.99) & (np.linalg.norm(vec2) > 700) & (np.linalg.norm(vec2) < 800)):
            rd = np.linalg.norm(block, axis=1)
            radius.append(((rd>644) & (rd<646)).sum())
            time.append(tmp[-1])
breakpoint()
#plt.axvline(80,linewidth=2, alpha=0.5)
ax1.set_aspect(1)
breakpoint()
thetas = np.linspace(-np.pi, np.pi, 100)
p1, = ax1.plot(np.sin(thetas)*650, np.cos(thetas)*650, alpha=0.5, linewidth=0.5, 
               color='blue', linestyle='-', label='acrylic')
p2 = plt.scatter(0, 260, marker='*', s=100,  
               color='black', label='vertex')

ax1.set_xlabel(r'$\sqrt{x^2+y^2}$/mm')
ax1.set_ylabel('$z$/mm')
ax1.tick_params(axis = 'both', which = 'major')
ax1.set_xlim([-20,800])
ax1.set_ylim([-750,750])
#ax1.tick_params(which='both', width=200)
#ax1.tick_params(which='major', length=200, color='r')
#ax1.tick_params(which='minor', length=200, color='r')
ax1.legend(handles = [p2, p20, p21, p1])
plt.tight_layout()
plt.savefig(output)
plt.close()