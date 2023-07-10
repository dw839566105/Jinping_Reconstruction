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

newcmp = cb()   

data = []
with PdfPages('recon_axis.pdf') as pdf:
    for axis in ['x','y','z']:
        recon = []
        truth = []
        wa = []
        ra = np.arange(-0.64, 0.65, 0.01)
        for i in ra:
            h = tables.open_file('/mnt/stage/douwei/JPSimData/recon/point/%s/%.2f.h5' % (axis, i))
            df_in = pd.DataFrame(h.root.ReconIn[:])
            df_out = pd.DataFrame(h.root.ReconOut[:])
            df_wa = pd.DataFrame(h.root.ReconWA[:])
            df_truth = pd.DataFrame(h.root.Truth[:])
            h.close()
            index = df_in['Likelihood'] < df_out['Likelihood']
            df = pd.concat([df_in[index], df_out[~index]]).sort_values('EventID')
            recon.append(df)
            truth.append(df_truth)
            wa.append(df_wa)

        df_recon = pd.concat(recon)
        df_truth = pd.concat(truth)
        wa = pd.concat(wa)

        index = np.sqrt(df_recon['x']**2 + df_recon['y']**2 + df_recon['z']**2) < 0.62
        for key in ['x', 'y', 'z']:
            fig = plt.figure(constrained_layout=False, figsize=(6,6))
            plt.hist2d(df_recon[key][index], 
                df_truth[key][index]/1000, 
                bins=ra - 0.005,
                cmap=newcmp)
            plt.xlabel(f'recon ${key}$/m')
            plt.ylabel(f'truth ${key}$/m')
            plt.title(f'Truth on {axis} axis')
            pdf.savefig(fig)  # saves the current figure into a pdf page
            plt.close()
            if key == axis:
                data.append([df_recon[key][index], df_truth[key][index]])

        data_sgl = pd.concat([df_recon, df_truth.add_suffix('_truth')], axis=1)
        for calc in ['mean', 'std']:
            fig = plt.figure(dpi=300)
            for key in ['x','y','z']:
                cmd = f'data_sgl.groupby(\'{axis}_truth\').{calc}()[\'{key}\'] - ra' if ((key == axis) and (calc == 'mean')) else \
                    f'data_sgl.groupby(\'{axis}_truth\').{calc}()[\'{key}\']'
                plt.scatter(ra, 
                    eval(cmd).values,
                    s = 1,
                    label=key)
            plt.legend()
            plt.title(f'Truth on {axis} axis')
            plt.xlabel('radius /m')
            plt.ylabel(f'{calc}/m' if calc == 'std' else 'bias/m')
            pdf.savefig(fig)
            plt.close()

    fig = plt.figure(dpi=300)
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, hspace=0.4, wspace=0.4)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax3 = fig.add_subplot(spec[1, 0])

    for ax, name, xvx in zip([ax1, ax2, ax3],['x', 'y', 'z'], data):
        im = ax.hist2d(xvx[0], 
            xvx[1]/1000,
            bins=(ra - 0.005),
            cmap=newcmp)
        ax.set_xlabel(f'{name}/m')
        ax.set_ylabel(f'{name}/m')
        ax.set_title(f'Truth on {name} axis')
    fig.colorbar(im[3], ax=[ax1, ax2, ax3], aspect=40, shrink = 1)
    pdf.savefig(fig)
    plt.close()