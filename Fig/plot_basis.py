import matplotlib.pyplot as plt
import fit
from DetectorConfig import shell
from matplotlib.backends.backend_pdf import PdfPages

def plot_fit(data, ax, xtitle, ytitle, start, end, unit):
    ax.hist(data, bins = 100, histtype='step', label=xtitle)
    x, popt, pcov = fit.fitdata(data, start, end, 1)
    ax.plot(x, fit.gauss(x, popt[0], popt[1], popt[2], popt[3]), label=f'mu-{popt[0]:.3f} sigma-{popt[1]:.3f}')
    ax.legend()
    ax.set_title(f'{xtitle} Distribution sigma/mu-{(popt[1]/popt[0]):.3f}')
    ax.set_xlabel(f'{xtitle} / {unit}')
    ax.set_ylabel(ytitle)
    return popt

def plot_fit_fig(pp, data, xtitle, ytitle, start, end, unit):
    fig, ax = plt.subplots()
    data = data[(data > start) & (data < end)]
    plot_fit(data, ax, xtitle, ytitle, start, end, unit)
    pp.savefig(fig)
    plt.close(fig)

def plot_hist(pp, data, xtitle, ytitle, unit):
    fig, ax = plt.subplots()
    ax.hist(data, bins = 100, histtype='step')
    ax.set_title(f'{xtitle} Distribution')
    ax.set_xlabel(f'{xtitle} / {unit}')
    ax.set_ylabel(ytitle)
    pp.savefig(fig)
    plt.close(fig)

def plot_zxy(pp, data1, data2, title):
    fig, ax = plt.subplots()
    h = ax.hist2d(data1, data2, bins = 100, range=[[0,shell**2],[-shell,shell]], cmap='Blues')
    fig.colorbar(h[3], ax=ax)
    ax.set_title(f'{title}:z-x^2+y^2')
    ax.set_xlabel('x^2 + y^2 / m^2')
    ax.set_ylabel('z / m')
    pp.savefig(fig)
    plt.close(fig)

def plot_hist2d(pp, data1, data2, title1, title2, start1, end1, start2, end2, unit1, unit2, binnum):
    fig, ax = plt.subplots()
    h = ax.hist2d(data1, data2, bins = binnum, range=[[min(start1, end1), max(start1, end1)], [min(start2, end2), max(start2, end2)]], cmap='Greys')
    fig.colorbar(h[3], ax=ax)
    ax.set_title(f'{title1}-{title2} Distribution')
    ax.set_xlabel(f'{title1} / {unit1}')
    ax.set_ylabel(f'{title2} / {unit2}')
    pp.savefig(fig)
    plt.close(fig)

def plot_scatter(pp, data1, data2, xtitle, ytitle, unit1, unit2):
    fig, ax = plt.subplots()
    ax.scatter(data1, data2, alpha=0.2, s=5)
    ax.set_title(f'{ytitle}-{xtitle} scatter distribution')
    ax.set_xlabel(f'{xtitle} / {unit1}')
    ax.set_ylabel(f'{ytitle} / {unit2}')
    pp.savefig(fig)
    plt.close(fig)