import tables
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages

def cb():
    viridis = cm.get_cmap('jet', 256)
    colors = viridis(np.linspace(0, 1, 65536))
    wt = np.array([1, 1, 1, 1])
    colors[:25, :] = wt
    cmp = ListedColormap(colors)
    return cmp

def load(file):
    h = tables.open_file('/mnt/stage/douwei/JPSimData/recon/shell/%.2f.h5' % file)
    df_in = pd.DataFrame(h.root.ReconIn[:])
    df_out = pd.DataFrame(h.root.ReconOut[:])
    df_truth = pd.DataFrame(h.root.Truth[:])
    h.close()
    index = df_in['Likelihood'] < df_out['Likelihood']
    df = pd.concat([df_in[index], df_out[~index]]).sort_values('EventID')
    return df, df_truth

def load_truth(file):
    h = tables.open_file('/mnt/stage/douwei/JPSimData/recon/shell/%.2f.h5' % file)
    df_in = pd.DataFrame(h.root.ReconIn[:])
    df_out = pd.DataFrame(h.root.ReconOut[:])
    
    h.close()
    index = df_in['Likelihood'] < df_out['Likelihood']
    df = pd.concat([df_in[index], df_out[~index]]).sort_values('EventID')
    return df

newcmp = cb()   
PMT = np.loadtxt('../PMT.txt')
data = []
with PdfPages('recon_shell.pdf') as pdf:
    '''
    fig = plt.figure(dpi=300)
    df = load(0.26)
    index = np.linalg.norm(df[['x', 'y', 'z']], axis=1) < 0.57
    plt.hist2d(np.arctan2(df['x'], df['y'])[index], 
        (df['z'] / np.linalg.norm(df[['x', 'y', 'z']], axis=1))[index],
        cmap = newcmp,  
        bins=50)
    pdf.savefig(fig)
    '''
    df = pd.concat([load(ra)[0] for ra in np.arange(0.35, 0.45, 0.01)])
    idx = np.linalg.norm(df[['x', 'y', 'z']], axis=1) > 0.55
    for index in [idx, ~idx]:
        fig = plt.figure()
        plt.hist2d(np.arctan2(df['y'], df['x'])[index], 
            (df['z'] / np.linalg.norm(df[['x', 'y', 'z']], axis=1))[index], 
            cmap = newcmp,
            bins=50)
        plt.scatter(np.arctan2(PMT[:,1], PMT[:,0]),
            PMT[:,2]/np.linalg.norm(PMT, axis=1),
            c = 'red',
            marker = '*')
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'$\cos\theta$')
        pdf.savefig(fig)

    df = pd.concat([load(ra)[0] for ra in np.arange(0.55, 0.60, 0.01)])
    df_truth = pd.concat([load(ra)[1] for ra in np.arange(0.55, 0.60, 0.01)])
    idx = np.linalg.norm(df[['x', 'y', 'z']], axis=1) > 0.48
    for index in [idx, ~idx]:
        fig = plt.figure()
        plt.hist2d(np.arctan2(df_truth['y'], df_truth['x'])[index], 
            (df_truth['z']/1000 / np.linalg.norm(df_truth[['x', 'y', 'z']]/1000, axis=1))[index], 
            cmap = newcmp,
            bins=50)
        plt.scatter(np.arctan2(PMT[:,1], PMT[:,0]),
            PMT[:,2]/np.linalg.norm(PMT, axis=1),
            c = 'red',
            marker = '*')
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'$\cos\theta$')
        pdf.savefig(fig)