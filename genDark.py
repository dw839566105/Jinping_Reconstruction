import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import fit
import numpy as np
import argparse

psr = argparse.ArgumentParser()
psr.add_argument("-d", dest="dark", help="figs")
psr.add_argument("-o", dest='opt', help="darkrate txt")
psr.add_argument('-i', dest='ipt', help="input")
args = psr.parse_args()

# 打开ROOT文件
file = uproot.open(args.ipt)
trigger = file["TriggerFrequencyforDN"].to_numpy()
darkrate = []
with PdfPages(args.dark) as pp:
    for i in range(30):
        darknoise = file[f"Channel_{i}/DarknoiseFrequency"].to_numpy()
        data = darknoise[0] * 1E9 / trigger[0] / 150
        data = np.nan_to_num(data, nan=0.0)
        fig, ax = plt.subplots()
        ax.hist(data, bins = 100, histtype='step', label="darkrate")
        x, popt, pcov = fit.fitdata(data, 10, 100000, 1)
        ax.plot(x, fit.gauss(x, popt[0], popt[1], popt[2], popt[3]), label=f'mu-{popt[0]:.3f} sigma-{popt[1]:.3f}')
        ax.legend()
        ax.set_title(f'darkrate Distribution Channel_{i} sigma/mu-{(popt[1]/popt[0]):.3f}')
        ax.set_xlabel('darkrate / Hz')
        ax.set_ylabel('bins')
        pp.savefig(fig)
        plt.close(fig)
        darkrate.append(popt[0])

np.savetxt(args.opt, np.array(darkrate), fmt='%d')