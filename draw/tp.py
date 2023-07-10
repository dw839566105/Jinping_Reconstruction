import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as LG
import matplotlib.pyplot as plt
import tables
import numpy as np

import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import sys
output = sys.argv[1]

fig = plt.figure(constrained_layout=True, figsize=(10,4))

spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, wspace=0.05)
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[0, 1])

# example of read 1 file
def main(path):
    
    x_recon = np.empty(0)
    y_recon = np.empty(0)
    z_recon = np.empty(0)
    x_truth = np.empty(0)
    y_truth = np.empty(0)
    z_truth = np.empty(0)
    
    for i,file in enumerate(np.arange(0.25,0.258,0.01)):
        
        for j in np.arange(25):
            if j == 0:
                h = tables.open_file('%s/1t_%+.3f.h5' % (path, file),'r')
            else:
                try:
                    h = tables.open_file('%s/1t_%+.3f_%d.h5' % (path, file, j),'r')
                except:
                    break
            recondata = h.root.Recon
            E1 = recondata[:]['E_sph_in']
            x1 = recondata[:]['x_sph_in']
            y1 = recondata[:]['y_sph_in']
            z1 = recondata[:]['z_sph_in']
            L1 = recondata[:]['Likelihood_in']
            s1 = recondata[:]['success_in']

            E2 = recondata[:]['E_sph_out']
            x2 = recondata[:]['x_sph_out']
            y2 = recondata[:]['y_sph_out']
            z2 = recondata[:]['z_sph_out']
            L2 = recondata[:]['Likelihood_out']
            s2 = recondata[:]['success_out']

            data = np.zeros((np.size(x1),3))

            data[:,0] = x1
            data[:,1] = y1
            data[:,2] = z1

            index = L2<L1
            data[index,0] = x2[index]
            data[index,1] = y2[index]
            data[index,2] = z2[index]

            x = data[:,0]
            y = data[:,1]
            z = data[:,2]

            x_recon = np.hstack((x_recon, x))
            y_recon = np.hstack((y_recon, y))
            z_recon = np.hstack((z_recon, z))
            h.close()
        h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/point_10/1t_%+.3f_total.h5' % file)
        x = h.root.TruthData[:]['x']
        y = h.root.TruthData[:]['y']
        z = h.root.TruthData[:]['z']
        h.close()
        x_truth = np.hstack((x_truth, x))
        y_truth = np.hstack((y_truth, y))
        z_truth = np.hstack((z_truth, z))
    return x_recon, y_recon, z_recon, x_truth, y_truth, z_truth

x_recon, y_recon, z_recon, x_truth, y_truth, z_truth = main('/home/douwei/Recon1t/Recon1tonSim/result_1t_point_10_Recon_1t_new2')

r = np.sqrt(x_recon**2 + y_recon**2 + z_recon**2)
# index = (r<0.5) & (r>0.01)
viridis = cm.get_cmap('jet', 256)
newcolors = viridis(np.linspace(0, 1, 65536))
wt = np.array([1, 1, 1, 1])
newcolors[:25, :] = wt
newcmp = ListedColormap(newcolors)


theta = (np.arctan2(y_recon, x_recon))
phi = (z_recon/(r+1e-6))

cmin = 0
cmax = 50
vmin = 0
vmax = 50
index = (r<0.5) & (r>0.1)
im1 = ax1.hist2d(theta[index], phi[index], cmin=cmin, cmax=cmax, vmin=vmin, vmax=vmax, 
                 bins=80, cmap=newcmp)
ax1.set_xlabel(r'$\phi$')
ax1.set_ylabel(r'$\cos\theta$')
ax1.tick_params(axis = 'both', which = 'major')

x_recon, y_recon, z_recon, x_truth, y_truth, z_truth = main('/home/douwei/Recon1t/Recon1tonSim/result_1t_point_10_Recon_1t_closed_new_PE')
r = np.sqrt(x_recon**2 + y_recon**2 + z_recon**2)
# index = (r<0.5) & (r>0.01)
theta = (np.arctan2(y_recon, x_recon))
phi = (z_recon/(r+1e-6))
index = (r<0.5) & (r>0.1)
im2 = ax2.hist2d(theta[index], phi[index], cmin=cmin, cmax=cmax, vmin=vmin, vmax=vmax, 
                 bins=80, cmap=newcmp)
ax2.set_xlabel(r'$\phi$', fontsize=20)
ax2.set_ylabel(r'$\cos\theta$', fontsize=20)
ax2.tick_params(axis = 'both', which = 'major', labelsize = 15)

ax1.set_title('Recon with all PMT', fontsize=20)
ax2.set_title('Recon with a closed PMT', fontsize=20)

PMT_pos = np.loadtxt('/home/douwei/Recon1t/calib/PMT_1t.txt')
ax1.scatter(np.arctan2(PMT_pos[:,1], PMT_pos[:,0]), PMT_pos[:,2]/0.83, s=200, marker='o', label='PMT',
           facecolors='none', edgecolors='r',linewidth=2.5)
#ax1.scatter(np.arctan2(PMT_pos[3,1], PMT_pos[3,0]), PMT_pos[3,2]/0.83, s=200, marker='o', label='PMT',
#           facecolors='none', edgecolors='r')
ax1.legend(loc=3)
index = np.hstack((np.arange(0,3), np.arange(4,30)))
ax2.scatter(np.arctan2(PMT_pos[index,1], PMT_pos[index,0]), PMT_pos[index,2]/0.83, s=200, marker='o', label='PMT',
           facecolors='none', edgecolors='r', linewidth=2.5)
ax2.scatter(np.arctan2(PMT_pos[3,1], PMT_pos[3,0]), PMT_pos[3,2]/0.83, s=200, marker='^', label=' closed PMT',
           facecolors='none', edgecolors='r', linewidth=2.5)
ax2.legend(loc=3)
ax1.locator_params(nbins=8, axis='x')
ax1.locator_params(nbins=8, axis='y')
ax2.locator_params(nbins=8, axis='x')
ax2.locator_params(nbins=8, axis='y')

ax1.axvline(-1.95 + np.pi,color = 'r')
ax1.axvline(-1.8 + np.pi,color= 'r')
ax1.axhline(-0.25,color = 'r')
ax1.axhline(-0.34,color= 'r')

ax2.axvline(-1.95 + np.pi,color = 'r')
ax2.axvline(-1.8 + np.pi,color= 'r')
ax2.axhline(-0.25,color = 'r')
ax2.axhline(-0.34,color= 'r')

#cbaxes = fig.add_axes([0.92, 0.03, 0.1, 0.9])
#cbar = fig.colorbar(im1[3], ax=[ax1, ax2], aspect=40, shrink = 1, cax = cbaxes)
cbar = fig.colorbar(im1[3], ax=[ax1, ax2], aspect=60, shrink = 1, pad = 0.02)
cbar.ax.tick_params(labelsize=15)
fig.savefig(output)
plt.close()