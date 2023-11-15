import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import tables
from tqdm import tqdm
from numba import njit
from zernike import RZern
from matplotlib.colors import LogNorm
from polynomial import *

mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 'x-large'
mpl.rcParams['legend.frameon'] = False

text_kwargs = dict(ha='right', va='bottom', fontsize=20, color='C1')
@njit
def polyval(p, x):
    y = np.zeros((p.shape[1], x.shape[0]))
    for i in range(len(p)):
        y = y * x + p[i]
    return y


def radial(coefnorm, rhotab, k, rho):
    return coefnorm[k, np.newaxis] * polyval(rhotab.T[:, k, np.newaxis], rho)


@njit
def angular(m, theta):
    return np.cos(m.reshape(-1, 1) * theta)

def loadh5(filename):
    with tables.open_file(filename) as h:
        coef_ = h.root.coeff[:]
        coef_type = h.root.coeff.attrs.type
    return coef_, coef_type

def calc(xx, yy, theta):
    
    
    v0 = np.array((0, 1))
    v1 = np.array((np.sin(theta), np.cos(theta)))

    d1 = (xx - v0[0])**2 + (yy-v0[1])**2
    d2 = (xx - v1[0])**2 + (yy-v1[1])**2

    cth1 = (v0[0]*(xx-v0[0]) + v0[1]*(yy-v0[1]))/np.sqrt((xx-v0[0])**2 + (yy-v0[1])**2)
    cth2 = (v1[0]*(xx-v1[0]) + v1[1]*(yy-v1[1]))/np.sqrt((xx-v1[0])**2 + (yy-v1[1])**2)

    ratio0 = (1/d1) / (1/d2)
    ratio1 = ((1/d1) * cth1) / ((1/d2) * cth2)
    ratio0[xx**2 + yy**2>1] = np.nan
    ratio1[xx**2 + yy**2>1] = np.nan
    return ratio0, ratio1

def calc_cart(x_, y_, t):
    v1 = np.array((np.sin(t), np.cos(t)))
    xx_, yy_ = np.meshgrid(x_, y_, sparse=False)
    rr_ = np.sqrt(xx_.flatten()**2 + yy_.flatten()**2)
    zs_r_ = cart.coefnorm[zo, np.newaxis] * polyval(cart.rhotab.T[:, zo, np.newaxis], rr_)

    theta = (xx_.flatten()*v1[0] + yy_.flatten()*v1[1])/rr_/np.linalg.norm(v1)
    zs_a_ = angular(cart.mtab[zo].reshape(-1, 1), np.arccos(theta))
    p_ = np.matmul((zs_r_ * zs_a_).T, coeff).reshape(xx_.shape)
    p_[xx_**2 + yy_**2 > 1] = np.nan
    return xx_, yy_, p_
        
N = 300
r = np.linspace(0, 1, N)
t = np.linspace(0, np.pi*2, N)
rr, tt = np.meshgrid(r, t, sparse=False)
        
xx = rr*np.sin(tt)
yy = rr*np.cos(tt)

theta = np.pi/3*2
ratio0, ratio1 = calc(xx, yy, theta)

angle = np.deg2rad(23.5)

with PdfPages('equiv.pdf') as pdf:
    for idx, ratio in enumerate([ratio0, ratio1]):
        ratio[xx**2 + yy**2 > 1] = np.nan
        
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1, projection="polar", theta_offset = np.pi / 2)
        
        Nc = 2.5
        
        CS1 = ax.contour(tt, rr, np.log(ratio), levels=np.linspace(-Nc, Nc, 11), linestyles='solid', linewidths=1.5, colors='k')
        CS2 = ax.contour(np.pi*2 - tt, rr, np.log(ratio), levels=np.linspace(-Nc, Nc, 11), linestyles='dotted', linewidths=1.5, colors='k')
        fl = ax.fill_between(np.linspace(-np.pi/3, np.pi/3, 100), np.zeros(100), np.ones(100), color='grey', alpha=0.2)
        h1, _ = CS1.legend_elements()
        h2, _ = CS2.legend_elements()  

        sc = ax.scatter(0, 1, marker='^', c='k', s=90, label='PMT')
        ax.text(0, 1.05, 'PMT 1', **text_kwargs)
        ax.scatter(theta, 1, marker='^', c='k', s=90)
        ax.text(theta, 1.2, 'PMT 2', **text_kwargs)
        ax.scatter(-theta, 1, marker='^', c='k', s=90)
        ax.text(-theta, 1.4, 'PMT 3', **text_kwargs)
        ax.set_yticks([])
        ax.set_xticks(np.linspace(0,2 * np.pi,9)[:-1], [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$'])
        # plt.axis('equal')
        tc = np.linspace(-np.pi, np.pi, 100)
        pl = plt.plot(tc, np.ones_like(tc), c='k', label='Detector', linestyle='dashed', linewidth=1, alpha=0.5)
        
        if(idx==1):
            ax.plot(np.pi/4.6, 0.5, 'o', ms=20, markerfacecolor="None",
                 markeredgecolor='red', markeredgewidth=2)
            ax.plot(np.pi/4.6, 0.85, 'o', ms=20, markerfacecolor="None",
                 markeredgecolor='red', markeredgewidth=2)
        ax.set_ylim([0, 0.835/0.65 + 0.1])
        pdf.savefig(fig)
        plt.close()

    theta = np.pi/101*2
    xx_ = np.linspace(-0.3, 0.3, N)
    yy_ = np.linspace(0.9, 1, N)
    xx_, yy_ = np.meshgrid(xx_, yy_, sparse=False)
    ratio0, ratio1 = calc(xx_, yy_, theta)   
    for idx, ratio in enumerate([ratio0, ratio1]):
        ratio[xx**2 + yy**2 > 1] = np.nan
        
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        
        Nc = 2.5
        
        CS1 = ax.contour(-xx_, yy_, np.log(ratio), levels=np.linspace(-Nc, Nc, 3), linestyles='solid', linewidths=1.5, colors='k')
        CS2 = ax.contour(xx_, yy_, np.log(ratio), levels=np.linspace(-Nc, Nc, 3), linestyles='dotted', linewidths=1.5, colors='k')

        h1,_ = CS1.legend_elements()
        h2,_ = CS2.legend_elements()

        sc = ax.scatter(0, 1, marker='^', c='k', s=90, label='PMT')
        ax.text(0, 1.01, 'PMT 1', **text_kwargs)
        ax.scatter(np.sin(theta), np.cos(theta), marker='^', c='k', s=90)
        ax.text(np.sin(theta), np.cos(theta) + 0.01, 'PMT 3', **text_kwargs)
        ax.scatter(-np.sin(theta), np.cos(theta), marker='^', c='k', s=90)
        ax.text(-np.sin(theta), np.cos(theta) + 0.01, 'PMT 2', **text_kwargs)
        ax.set_xticks(np.linspace(0,2 * np.pi,9)[:-1], [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$'])
        ax.tick_params(labelsize=20)

        tc = np.linspace(-np.pi, np.pi, 100)
        pl = plt.plot(np.linspace(-1, 1, N), np.sqrt(1-np.linspace(-1, 1, N)**2), c='k', label='Detector', linestyle='dashed', linewidth=1, alpha=0.5)
        
        xsmall = np.linspace(-np.tan(np.pi/100), 0, 100) 
        fl = ax.fill_between(xsmall, -xsmall * (1/np.tan(np.pi/100)), np.sqrt(1-xsmall**2), color='grey', alpha=0.2)
        fl = ax.fill_between(-xsmall, -xsmall * (1/np.tan(np.pi/100)), np.sqrt(1-xsmall**2), color='grey', alpha=0.2)
        if(idx==1):
            ax.plot(-0.03, 0.975, 'o', ms=20, markerfacecolor="None",
                 markeredgecolor='red', markeredgewidth=2)
            ax.plot(-0.03, 0.995, 'o', ms=20, markerfacecolor="None",
                 markeredgecolor='red', markeredgewidth=2)
              
        ax.set_ylim([0.9, 1.03])
        ax.set_xlim([-0.1, +0.1])
        ax.set_xlabel('$x/r_\mathrm{LS}$')
        ax.set_ylabel('$y/r_\mathrm{LS}$')
        pdf.savefig(fig)
        plt.close()
        
    for i in range(2):
        # if using zernike 
        filename = '/home/douwei/ReconJP/coeff/Zernike/PE/2/20.h5'
        coef_PE, coef_type = loadh5(filename)
        with tables.open_file(filename) as h:
            coeff = h.root.coeff[:]
        cart = RZern(20)
        zo = cart.mtab>=0
        zs_radial = cart.coefnorm[zo, np.newaxis] * polyval(cart.rhotab.T[:, zo, np.newaxis], rr.flatten())
        zs_angulars = angular(cart.mtab[zo].reshape(-1, 1), tt.flatten())
        probe1 = np.matmul((zs_radial * zs_angulars).T, coeff)

        filename = '/mnt/stage/douwei/JP_1t_github/coeff/Legendre/Gather/PE/2/80/40.h5'
        with tables.open_file(filename) as h:
            coef = h.root.coeff[:]
        cut = len(coef)
        t_basis = legval_raw(np.cos(tt.flatten()), np.eye(cut).reshape((cut,cut,1))).T
        r_basis = legval_raw(rr.flatten(), coef.T.reshape(coef.shape[1], coef.shape[0],1)).T
        probe = (r_basis*t_basis).sum(-1)

        shift = np.int16(N/10)
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1, projection="polar", theta_offset = np.pi / 2)

        data = np.exp(probe).reshape(rr.shape)
        Nc = 1
        if i == 0:
            CS1 = ax.contour(tt, rr, -np.log(np.roll(data, shift, axis=0)/np.roll(data, 2*shift+1, axis=0)), levels=np.linspace(-Nc, 0, 5), cmap='jet', linewidths=1.5)
            CS2 = ax.contour(np.pi*2-tt, rr, np.log(np.roll(data, shift+1, axis=0)/np.roll(data, -shift, axis=0)), levels=np.linspace(-Nc, 0, 5), cmap='jet',  linestyles='dotted', linewidths=1.5)
            ax.scatter(np.pi*2/10, 0.83/0.65, marker='^', color='k', s=90)
            ax.scatter(-np.pi*2/10, 0.83/0.65, marker='^', color='k', s=90)
            ax.scatter(2*np.pi*2/10, 0.83/0.65, marker='^', color='k', s=90, label='PMT')
            ax.text(np.pi*2/10, 0.83/0.65 - 0.3, 'PMT 1', **text_kwargs)
            ax.text(-np.pi*2/10, 0.83/0.65 - 0.2, 'PMT 3', **text_kwargs)
            ax.text(2*np.pi*2/10, 0.83/0.65 - 0.3, 'PMT 2', **text_kwargs)
            fl = ax.fill_between(np.linspace(0, np.pi*3/10,100), np.zeros(100), np.ones(100), color='grey', alpha=0.2)
            ax.set_ylim([0, 0.835/0.65])
            
            xx_, yy_, p1 = calc_cart(np.linspace(-0.5, 0.5, 300), np.linspace(0, 1, 300), 1*np.pi*2/10)
            xx_, yy_, p2 = calc_cart(np.linspace(-0.5, 0.5, 300), np.linspace(0, 1, 300), -1*np.pi*2/10)
            xx_, yy_, p3 = calc_cart(np.linspace(-0.5, 0.5, 300), np.linspace(0, 1, 300), 2*np.pi*2/10)

            axins = ax.inset_axes([0.0, 0.0, 0.4, 0.4])
            axins.contour(-xx_ , yy_ , p3 - p1, levels=[0,],
                          linestyles='solid', linewidths=1.5)
            axins.contour(xx_ , yy_ , p1 - p2, levels=[0,], 
                          linestyles='dotted', linewidths=1.5)
            axins.plot(0.0, 0.0, 'o', ms=20, markerfacecolor="None",
                 markeredgecolor='red', markeredgewidth=2)
            axins.plot(0.0, 0.92, 'o', ms=20, markerfacecolor="None",
                 markeredgecolor='red', markeredgewidth=2)
            xsmall = np.linspace(-0.5, 0, 100) 
            axins.fill_between(xsmall, -xsmall*np.tan(2/10*np.pi), np.sqrt(1-xsmall**2),
                              color='grey', alpha=0.2)            
        else:
            CS1 = ax.contour(tt, rr, np.log(np.roll(data, shift, axis=0)/np.roll(data, 0, axis=0)), levels=np.linspace(-3*Nc, 0, 9), cmap='jet', linewidths=1.5)
            CS2 = ax.contour(tt, rr, np.log(np.roll(data, -shift, axis=0)/np.roll(data, 0, axis=0)), levels=np.linspace(-3*Nc, 0, 9), cmap='jet',  linestyles='dotted', linewidths=1.5)
            ax.scatter(np.pi*2/10, 0.83/0.65, marker='^', color='k', s=90)
            ax.scatter(-np.pi*2/10, 0.83/0.65, marker='^', color='k', s=90)
            ax.scatter(0, 0.83/0.65, marker='^', color='k', s=90, label='PMT')
            ax.text(0, 0.83/0.65 - 0.3, 'PMT 1', **text_kwargs)
            ax.text(np.pi*2/10, 0.83/0.65-0.3, 'PMT 2', **text_kwargs)
            ax.text(-np.pi*2/10, 0.83/0.65-0.2, 'PMT 3', **text_kwargs)
            ax.set_xticks(np.linspace(0,2 * np.pi,9)[:-1], [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$'])
            fl = ax.fill_between(np.linspace(-np.pi/10, np.pi/10, 100), np.zeros(100), np.ones(100), color='grey', alpha=0.2)
            ax.set_ylim([0, 0.835/0.65])
            xx_, yy_, p1 = calc_cart(np.linspace(-0.5, 0.5, 300), np.linspace(0, 1, 300), 0)
            xx_, yy_, p2 = calc_cart(np.linspace(-0.5, 0.5, 300), np.linspace(0, 1, 300), np.pi*2/10)
            xx_, yy_, p3 = calc_cart(np.linspace(-0.5, 0.5, 300), np.linspace(0, 1, 300), -np.pi*2/10)
            
            axins = ax.inset_axes([0.0, 0.0, 0.4, 0.4])
            axins.contour(xx_ , yy_ , p1 - p3, levels=[0,], linestyles='solid', linewidths=1.5)
            axins.contour(xx_ , yy_ , p1 - p2, levels=[1.5,], linestyles='dotted', linewidths=1.5)
            xsmall = np.linspace(-1, 0, 100) * np.tan(1/10*np.pi) 
            axins.fill_between(xsmall, -np.linspace(-1, 0, 100) * np.tan(2.4/10*np.pi), np.sqrt(1-xsmall**2), color='grey', alpha=0.2)
            axins.fill_between(-xsmall, -np.linspace(-1, 0, 100) * np.tan(2.4/10*np.pi), np.sqrt(1-xsmall**2), color='grey', alpha=0.2)
            axins.plot(-0.250, 0.750, 'o', ms=40, markerfacecolor="None",
                 markeredgecolor='red', markeredgewidth=2)
            
        dt = np.linspace(-0.5, 0.5, 100)
        axins.plot(dt , np.sqrt(1-dt**2) , c='k', label='Detector', linestyle='dashed', linewidth=1, alpha=0.5)
        axins.tick_params(axis='both', which='major', labelsize=10)
        axins.set_xlabel('$x/r_\mathrm{LS}$')
        axins.set_ylabel('$y/r_\mathrm{LS}$')
        
        h1,_ = CS1.legend_elements()
        h2,_ = CS2.legend_elements()
        tc = np.linspace(-np.pi, np.pi, 100)
        pl = plt.plot(tc, np.ones_like(tc), c='k', label='Detector', linestyle='dashed', linewidth=1, alpha=0.5)
        ax.set_ylim([0, 0.83/0.65 + 0.1])
        ax.set_yticks([])
        pdf.savefig(fig)
        plt.close()
        
    for i in range(2):
        # filename = '/home/douwei/ReconJP/coeff/Zernike/PE/2/20.h5'
        # with tables.open_file(filename) as h:
        #    coeff = h.root.coeff[:]
        # cart = RZern(20)
        # zo = cart.mtab>=0
        # zs_radial = cart.coefnorm[zo, np.newaxis] * polyval(cart.rhotab.T[:, zo, np.newaxis], rr.flatten())
        # zs_angulars = angular(cart.mtab[zo].reshape(-1, 1), tt.flatten())
        # probe = np.matmul((zs_radial * zs_angulars).T, coeff)
        
        filename = '/mnt/stage/douwei/JP_1t_github/coeff/Legendre/Gather/PE/2/80/40.h5'
        with tables.open_file(filename) as h:
            coef = h.root.coeff[:]
        cut = len(coef)
        t_basis = legval_raw(np.cos(tt.flatten()), np.eye(cut).reshape((cut,cut,1))).T
        r_basis = legval_raw(rr.flatten(), coef.T.reshape(coef.shape[1], coef.shape[0],1)).T
        probe = (r_basis*t_basis).sum(-1)

        shift = np.int16(N/10)
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1, projection="polar", theta_offset = np.pi / 2)
        data = np.exp(probe).reshape(rr.shape)
        Nc = 1
        if i == 0:
            CS1 = ax.contour(tt, rr, -np.log(np.roll(data, shift, axis=0)/np.roll(data, 2*shift+1, axis=0)), levels=np.linspace(-Nc, 0, 5)[-1:], cmap='jet', linewidths=1.5)
            CS2 = ax.contour(np.pi*2-tt, rr, np.log(np.roll(data, shift+1, axis=0)/np.roll(data, -shift, axis=0)), levels=np.linspace(-Nc, 0, 5)[-1:], cmap='jet',  linestyles='dotted', linewidths=1.5)

            ax.scatter(np.pi*2/10, 0.83/0.65, marker='^', color='k', s=90)
            ax.scatter(-np.pi*2/10, 0.83/0.65, marker='^', color='k', s=90)
            ax.scatter(2*np.pi*2/10, 0.83/0.65, marker='^', color='k', s=90, label='PMT')
            ax.text(np.pi*2/10, 0.83/0.65 - 0.3, 'PMT 1', **text_kwargs)
            ax.text(-np.pi*2/10, 0.83/0.65 - 0.1, 'PMT 3', **text_kwargs)
            ax.text(2*np.pi*2/10, 0.83/0.65 - 0.2, 'PMT 2', **text_kwargs)
            ax.set_xticks(np.linspace(0,2 * np.pi,9)[:-1], [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$'])
            ax.set_rlabel_position(180-22.5)
            fl = ax.fill_between(np.linspace(0, np.pi*3/10,100), np.zeros(100), np.ones(100), color='grey', alpha=0.2)
            ax.set_ylim([0, 0.835/0.65])
            
            xx_, yy_, p1 = calc_cart(np.linspace(-0.5, 0.5, 300), np.linspace(0, 1, 300), 1*np.pi*2/10)
            xx_, yy_, p2 = calc_cart(np.linspace(-0.5, 0.5, 300), np.linspace(0, 1, 300), -1*np.pi*2/10)
            xx_, yy_, p3 = calc_cart(np.linspace(-0.5, 0.5, 300), np.linspace(0, 1, 300), 2*np.pi*2/10)

            axins = ax.inset_axes([0.0, 0.0, 0.4, 0.4])
            axins.contour(-xx_ , yy_ , p3 - p1, levels=[0,],
                          linestyles='solid', linewidths=1.5)
            axins.contour(xx_ , yy_ , p1 - p2, levels=[0,], 
                          linestyles='dotted', linewidths=1.5)
            axins.plot(0.0, 0.0, 'o', ms=20, markerfacecolor="None",
                 markeredgecolor='red', markeredgewidth=2)
            axins.plot(0.0, 0.92, 'o', ms=20, markerfacecolor="None",
                 markeredgecolor='red', markeredgewidth=2)
            xsmall = np.linspace(-0.5, 0, 100) 
            axins.fill_between(xsmall, -xsmall*np.tan(2/10*np.pi), np.sqrt(1-xsmall**2),
                              color='grey', alpha=0.2)
        else:
            CS1 = ax.contour(tt, rr, np.log(np.roll(data, shift, axis=0)/np.roll(data, 0, axis=0)), levels=np.linspace(-3*Nc, 0, 9)[-1:], cmap='jet', linewidths=1.5)
            CS2 = ax.contour(tt, rr, np.log(np.roll(data, -shift, axis=0)/np.roll(data, 0, axis=0)), levels=np.linspace(-3*Nc, 0, 9)[4:5], cmap='jet',  linestyles='dotted', linewidths=1.5)
            ax.scatter(np.pi*2/10, 0.83/0.65, marker='^', color='k', s=90)
            ax.scatter(-np.pi*2/10, 0.83/0.65, marker='^', color='k', s=90)
            ax.scatter(0, 0.83/0.65, marker='^', color='k', s=90, label='PMT')
            ax.text(0, 0.83/0.65 - 0.3, 'PMT 1', **text_kwargs)
            ax.text(np.pi*2/10, 0.83/0.65-0.3, 'PMT 2', **text_kwargs)
            ax.text(-np.pi*2/10, 0.83/0.65-0.1, 'PMT 3', **text_kwargs)
            ax.set_xticks(np.linspace(0,2 * np.pi,9)[:-1], [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$'])
            ax.set_rlabel_position(180-22.5)
            fl = ax.fill_between(np.linspace(-np.pi/10, np.pi/10, 100), np.zeros(100), np.ones(100), color='grey', alpha=0.2)
            ax.set_ylim([0, 0.835/0.65])
            xx_, yy_, p1 = calc_cart(np.linspace(-0.5, 0.5, 300), np.linspace(0, 1, 300), 0)
            xx_, yy_, p2 = calc_cart(np.linspace(-0.5, 0.5, 300), np.linspace(0, 1, 300), np.pi*2/10)
            xx_, yy_, p3 = calc_cart(np.linspace(-0.5, 0.5, 300), np.linspace(0, 1, 300), -np.pi*2/10)
            axins = ax.inset_axes([0.0, 0.0, 0.4, 0.4])
            axins.contour(xx_ , yy_ , p1 - p3, levels=[0,], linestyles='solid', linewidths=1.5)
            axins.contour(xx_ , yy_ , p1 - p2, levels=[1.5,], linestyles='dotted', linewidths=1.5)
            xsmall = np.linspace(-1, 0, 100) * np.tan(1/10*np.pi) 
            axins.fill_between(xsmall, -np.linspace(-1, 0, 100) * np.tan(2.4/10*np.pi), np.sqrt(1-xsmall**2), color='grey', alpha=0.2)
            axins.fill_between(-xsmall, -np.linspace(-1, 0, 100) * np.tan(2.4/10*np.pi), np.sqrt(1-xsmall**2), color='grey', alpha=0.2)
            axins.plot(-0.250, 0.750, 'o', ms=40, markerfacecolor="None",
                 markeredgecolor='red', markeredgewidth=2)
            
        dt = np.linspace(-0.5, 0.5, 100)
        axins.plot(dt , np.sqrt(1-dt**2) , c='k', label='Detector', linestyle='dashed', linewidth=1, alpha=0.5)
        axins.tick_params(axis='both', which='major', labelsize=10)
        axins.set_xlabel('$x/r_\mathrm{LS}$')
        axins.set_ylabel('$y/r_\mathrm{LS}$')
        
        h1,_ = CS1.legend_elements()
        h2,_ = CS2.legend_elements()
        tc = np.linspace(-np.pi, np.pi, 100)
        pl = plt.plot(tc, np.ones_like(tc), c='k', label='Detector', linestyle='dashed', linewidth=1, alpha=0.5)
        ax.set_ylim([0, 0.83/0.65 + 0.1])
        ax.set_yticks([])
        pdf.savefig(fig)
        plt.close()

    plt.figure()
    x = np.arange(5.0)
    y = np.arange(0,5,2*np.sqrt(3))
    for y_ in y:
        plt.scatter(x, np.full_like(x, y_), marker='^', c='k', s=50)

    x = np.arange(0.5,4.5)
    y = np.arange(np.sqrt(3),6,2*np.sqrt(3))
    for y_ in y:
        c1 = plt.scatter(x, np.full_like(x, y_), marker='^', c='k', s=50, label='PMT')

    plt.plot([1.5,2.5], [np.sqrt(3), np.sqrt(3)], 'k--', lw=1)
    plt.plot([1.5,2], [np.sqrt(3), 2 * np.sqrt(3)], 'k--', lw=1)
    plt.plot([2,2.5], [2 * np.sqrt(3), np.sqrt(3)], 'k--', lw=1)
    plt.text(2, 1.2, '$d$', ha='center', va='center', fontsize=25)
    plt.text(3, 2.5, '$S=\sqrt{3}d^2/4$', ha='center', va='center', fontsize=25)
    plt.axis('off')
    plt.legend(handles=[c1,], loc=1, frameon=True)
    pdf.savefig()
    
    im = []

    rs = np.arange(0.85, 1.55, 0.05)
    ts = np.arange(1.0, 1.50, 0.05)
    
    for ra in rs:
        for idx in ts:
            if ra == 0.85:
                filename = '/tarski/tjjk/douwei/Sim1/coeff/%.2f/%.3f/coeff.h5' % (idx, 0.832)
            else:
                filename = '/home/douwei/Sim/Sim1/coeff_old1/%.2f/%.2f/coeff.h5' % (idx, ra)
            coef_PE, coef_type = loadh5(filename)

            with tables.open_file(filename) as h:
                coeff = h.root.coeff[:]

            probe = np.matmul((zs_radial * zs_angulars).T, coeff[:121])
            data = np.exp(probe).reshape(rr.shape)
            
            sel = np.where(data[:,-1] < data[-1,-1]/10)
            if len(sel) > 0:
                shift = sel[0][0]
            else:
                shift = 0
            im.append((N/shift, ra, idx))
            
    # 3-d PMT number: 2/np.sqrt(3) * / np.pi * N**2
    cl = np.arange(7,21,2)             
    im = np.array(im)
    N1 = (im[:,0].reshape(len(rs), len(ts)).T)
    N2 = 2/ np.sqrt(3) / np.pi * N1**2
    
    plt.figure()
    plt.contourf(1000*rs, ts, N2, levels=15, cmap='jet')
    ax = plt.gca()
    ax.scatter(850, 1.33, marker='o', s=100, edgecolor='r', facecolor="None", linewidth=2, label='Jinping prototype')
    ax.scatter(900, 1.33, marker='*', s=100, edgecolor='r', facecolor="None", linewidth=2, label='Ideal detector')
    ax.tick_params(labelsize=20)
    ax.set_xticks((850, 1000, 1200, 1400), (832, 1000, 1200, 1400))
    cb = plt.colorbar(format='%d')
    cb.ax.set_title('$N_\mathrm{PMT}$', fontsize = 25, pad=18)
    ax.axvline(900, lw=1, ls='dashed', c='k')
    ax.axvline(850, lw=1, ls='dashed', c='k')
    ax.axhline(1.33, lw=1, ls='dashed', c='k')
    ax.legend(frameon=True, fontsize=22, loc=1)
    plt.xlabel(r'$r_\mathrm{PMT}$/mm')
    plt.ylabel(r'Refractive index')
    plt.tight_layout()
    pdf.savefig()

    plt.figure()
    plt.contourf(1000*rs, ts, N2 / 4 * 0.1**2 / (rs**2), levels=np.linspace(0.00, 0.64, 15), cmap='jet')
    plt.axhline(1.33, color='k', ls='--', lw=1)
    plt.gca().tick_params(labelsize=22)
    cb = plt.colorbar(format='%.2f')
    cb.ax.set_title('coverage', fontsize = 25, pad=18)
    plt.xlabel('$r_\mathrm{PMT}$/mm')
    plt.ylabel('Index of the outer material')
    plt.tight_layout()
    pdf.savefig()
    plt.close()
