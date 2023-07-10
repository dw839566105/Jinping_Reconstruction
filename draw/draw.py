from logging import warning
import numpy as np
import math
from zernike import RZern
import h5py as h5
import pandas as pd
import argparse
from polynomial import *
import numpy.polynomial.legendre as LG
from matplotlib.colors import LogNorm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.animation import FuncAnimation, PillowWriter
import multiprocessing.dummy as mp
from calculate import *
from coefficient import CoefLoader, ConcatInfo, load_coef

matplotlib.use("agg")

plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "font.serif": ["Times"],
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "font.size": 13,
    }
)

psr = argparse.ArgumentParser()

psr.add_argument("command", type=str, help="command")
psr.add_argument("-c", "--coef", dest="coef", type=str, help="input file")
psr.add_argument("--table", dest="table", nargs="+", help="coef table name")
psr.add_argument("--concat", dest="concat", type=str, help="concat file")
psr.add_argument("--r_max", dest="r_max", type=float, help="maximum radius")
psr.add_argument("-o", "--output", dest="opt", type=str, help="output file")
args = psr.parse_args()

hist_rths = np.array(
    [
        (0, 0, "0"),
        (0.99, math.pi / 4, "$\\frac{\\pi}{4}$"),
        (0.99, 0, "0"),
        (0.99, math.pi, "$\\pi$"),
        (0.5, math.pi, "$\\pi$"),
        (0.7, 5 * math.pi / 6, "$\\frac 56 \\pi$"),
        (0.9, math.pi / 2, "$\\frac{\\pi}{2}$"),
    ],
    dtype=[
        ("r", np.float64),
        ("theta", np.float64),
        ("theta_text", object),
    ],
)

neighborhood_r = 0.05

def draw_pie(coef: CoefLoader, fig, ax):
    thetas2 = np.linspace(0, 2 * math.pi, 201)
    rs2 = np.linspace(0, 1, 101)
    s = CalculatePE(c.v_rs, c.v_thetas)
    ss = s.T
    m = ax.pcolormesh(thetas2, rs2, ss, norm=LogNorm(), cmap="jet")
    fig.colorbar(m)


def prepare_pie_frame(coef: CoefLoader):
    thetas2 = np.linspace(0, 2 * math.pi, 201)
    rs2 = np.linspace(0, 1, 101)
    zs2 = coef.get_zernike_grid(rs2, thetas2)
    return thetas2, rs2, np.tensordot(coef.coef, zs2, axes=1)


def draw_pie_frame(coef: CoefLoader, fig, ax, thetas2, rs2, timed_zs2, t):
    lt2 = coef.get_legendre(np.array([coef.time_to_legendre(t)]))[:, 0]
    if coef.type == "exp":
        s = np.exp(np.tensordot(lt2, timed_zs2, axes=1))
    elif coef.type == "square":
        s = np.tensordot(lt2, timed_zs2, axes=1) ** 2
    elif coef.type == "traditional":
        s = CalculatePE(rs2, thetas2.reshape(-1, 1))
        s = s.T

    m = ax.pcolormesh(thetas2, rs2, s, norm=LogNorm(), cmap="jet")


def draw_neighborhood(fig, ax):
    mr = ax.bbox.width / 2 * neighborhood_r
    ax.scatter(
        hist_rths["theta"],
        hist_rths["r"],
        facecolors="none",
        edgecolors="k",
        s=mr ** 2,
        marker="o",
    )

def draw_time_hist(coef: CoefLoader, c: ConcatInfo, r, theta, fig, ax):
    sts = c.pe_ts[
        c.pe_rs ** 2 + r ** 2 - 2 * c.pe_rs * r * np.cos(c.pe_thetas - theta)
        <= neighborhood_r ** 2
    ]
    svf = (
        c.v_rs ** 2 + r ** 2 - 2 * c.v_rs * r * np.cos(c.v_thetas - theta)
        <= neighborhood_r ** 2
    )
    svrs = c.v_rs[svf]
    svths = c.v_thetas[svf]
    ts2 = np.linspace(-1, 1, num=10001)
    lt2 = coef.get_legendre(ts2)
    zs2 = coef.get_zernike(svrs, svths)
    coef.type = 'traditional'
    if coef.type == "exp":
        s = np.exp(np.tensordot(np.dot(lt2.T, coef.coef), zs2, axes=1))
    elif coef.type == "square":
        s = np.tensordot(np.dot(lt2.T, coef.coef), zs2, axes=1) ** 2
    elif coef.type == "traditional":
        ts2 = np.linspace(coef.tmin, coef.tmax, 10001)
        expect_pe = CalculatePE(svrs, svths)
        expect_time = CalculateTime(svrs, svths, ts2, valid=False)
        s = (expect_pe[:, np.newaxis] * expect_time).T
        ts2 = np.linspace(-1, 1, num=10001)
    ss = np.average(s, axis=1)
    ax.plot(
        coef.legendre_to_time(ts2),
        ss,
        label="R(t)".format(r, theta),
    )
    n = len(svrs)
    if n != 0:
        time_range = (coef.tmin, coef.tmax)
        bins = math.ceil(coef.tbins)
        ax.hist(
            sts,
            range=time_range,
            bins=bins,
            weights=np.repeat(1.0 / n, len(sts)),
            label="histogram",
        )


def verf(coef: CoefLoader, c: ConcatInfo, fig, ax):
    r_bins = np.linspace(0, 1, 51)
    theta_bins = np.linspace(0, 2 * np.pi, 201)
    r_mid = (r_bins[1:] + r_bins[:-1]) / 2
    theta_mid = (theta_bins[1:] + theta_bins[:-1]) / 2
    Binning = [r_bins, theta_bins]
    zernike_on_points = coef.get_zernike_grid(r_mid, theta_mid)
    if coef.type == "exp":
        ts2 = np.arange(coef.tmin, coef.tmax)
        lt2 = coef.get_legendre(coef.time_to_legendre(ts2))
        Amplitude = np.sum(
            np.exp(np.tensordot(np.dot(lt2.T, coef.coef), zernike_on_points, axes=1)),
            axis=0,
        )
    elif coef.type == "square":
        Amplitude = (
            np.sum(np.tensordot(coef.coef, zernike_on_points, axes=1) ** 2, axis=0)
            * coef.tbins
            / 2
        )
    elif coef.type == "traditional":
        expect_pe = CalculatePE(r_mid, theta_mid.reshape(-1, 1))
        Amplitude = expect_pe.T

    hist_PE, binr, binθ = np.histogram2d(c.f_pe_rs, c.f_pe_thetas, bins=Binning)
    hist_predict, _, _ = np.histogram2d(c.f_v_rs, c.f_v_thetas, bins=Binning)

    X, Y = np.meshgrid(binθ, binr)
    cm = ax.pcolormesh(
        X, Y, hist_PE / hist_predict / Amplitude, norm=LogNorm(), cmap="jet"
    )
    fig.colorbar(cm)


def real_pie(c: ConcatInfo, fig, ax):
    r_bins = np.linspace(0, 1, 51)
    theta_bins = np.linspace(0, 2 * np.pi, 201)
    Binning = [r_bins, theta_bins]
    hist_PE, binr, binθ = np.histogram2d(c.f_pe_rs, c.f_pe_thetas, bins=Binning)
    hist_predict, _, _ = np.histogram2d(c.f_v_rs, c.f_v_thetas, bins=Binning)

    X, Y = np.meshgrid(binθ, binr)
    cm = ax.pcolormesh(X, Y, hist_PE / hist_predict, norm=LogNorm(), cmap="jet")
    fig.colorbar(cm)


def LoadPMT(File="./PMT.txt"):
    PMT_pos = np.loadtxt(File)
    return PMT_pos


def Validate(coef: CoefLoader, c: ConcatInfo):
    nonhitz = coef.get_zernike(c.v_rs, c.v_thetas)
    hitz = coef.get_zernike(c.pe_rs, c.pe_thetas)

    index = np.logical_and(c.pe_ts > coef.tmin, c.pe_ts < coef.tmax)
    timing = c.pe_ts[index]
    hitt = coef.get_legendre(coef.time_to_legendre(timing))
    coef.type = 'traditional'
    coef.type = 'zernike'
    # coef.type = 'double'
    if coef.type == "exp":
        step = 0.1
        tw = np.arange(coef.tmin, coef.tmax, step) + 0.5
        nonhitt = coef.get_legendre(coef.time_to_legendre(tw))
        htz = hitt @ hitz[:, index].T
        nonhit = logsumexp(nonhitt.T @ coef.coef @ nonhitz)
        hit = np.sum(htz * coef.coef)
        return hit - np.exp(nonhit) * step

    elif coef.type == "square":
        grad_hit_dnum = np.einsum("mi,mi->i", coef.coef @ hitz[:, index], hitt)
        hit = 2 * np.sum(np.log(np.abs(grad_hit_dnum)))
        nonhit = coef.tbins / 2 * np.sum((coef.coef @ nonhitz) ** 2)
        return hit - nonhit

    elif coef.type == "traditional":
        v_pe = CalculatePE(c.v_rs, c.v_thetas)
        pe_pe = CalculatePE(c.pe_rs, c.pe_thetas)
        hit = CalculateTime(c.pe_rs, c.pe_thetas, c.pe_ts)
        score = np.sum(-v_pe) + np.sum(np.log(pe_pe)) # + np.sum(np.log(hit))
        return score
    
    elif coef.type == "zernike":
        v_pe = Marginal.CalculatePE(c.v_rs, c.v_thetas)
        pe_pe = Marginal.CalculatePE(c.pe_rs, c.pe_thetas)
        hit = Marginal.CalculateTime(c.pe_rs, c.pe_thetas, c.pe_ts)
        score = np.sum(-v_pe) + np.sum(np.log(pe_pe)) # + np.sum(np.log(hit))
        return score

    elif coef.type == "double":
        v_pe = Marginal1.CalculatePE(c.v_rs, c.v_thetas)
        pe_pe = Marginal1.CalculatePE(c.pe_rs, c.pe_thetas)
        # hit = CalculateTime(c.pe_rs, c.pe_thetas, c.pe_ts)
        score = np.sum(-v_pe) + np.sum(np.log(pe_pe)) # + np.sum(np.log(hit))
        return score

if args.command == "draw":
    concat = ConcatInfo(args.concat, args.r_max)

    def draw_log_pie_fig(coef: CoefLoader):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="polar", theta_offset=math.pi / 2)
        draw_pie(coef, fig, ax)
        ax.set_title(f"{key} Pie L={coef.nt} Z={len(coef.zo)}")
        print("Pie done")
        return fig

    def draw_real_pie_fig(concat: ConcatInfo):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="polar", theta_offset=math.pi / 2)
        real_pie(concat, fig, ax)
        draw_neighborhood(fig, ax)
        ax.set_title(f"{key} RealPie L={coef.nt} Z={len(coef.zo)}")
        print("RealPie done")
        return fig

    def draw_quotient_fig(coef: CoefLoader, concat: ConcatInfo):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="polar", theta_offset=math.pi / 2)
        verf(coef, concat, fig, ax)
        ax.set_title(f"{key} Quotient L={coef.nt} Z={len(coef.zo)}")
        print("Quotient done")
        return fig

    def draw_time_fig(coef: CoefLoader, concat: ConcatInfo, i, r, theta, theta_text):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        draw_time_hist(coef, concat, r, theta, fig, ax)
        ax.legend()
        ax.set_xlabel("t/ns")
        ax.set_title(
            f"{key} Time L={coef.nt} Z={len(coef.zo)} r={r} $\\theta$={theta_text}"
        )
        print("Time", i + 1, "done")
        return fig

    with PdfPages(args.opt) as pp:
        for key, set in load_coef(args.coef, args.table):
            # ugly hack, key cannot have "_", otherwise LaTeX fails
            key = key.replace("_", "-")

            coef = CoefLoader(set)

            pool = mp.Pool(3 + len(hist_rths))
            figures = []

            figures.append(pool.apply_async(draw_log_pie_fig, (coef,)))
            '''
            figures.append(pool.apply_async(draw_real_pie_fig, (concat,)))
            figures.append(pool.apply_async(draw_quotient_fig, (coef, concat)))
            for i, (r, theta, theta_text) in enumerate(hist_rths):
                figures.append(
                    pool.apply_async(
                        draw_time_fig, (coef, concat, i, r, theta, theta_text)
                    )
                )
            '''
            pool.close()
            pool.join()

            for fig in figures:
                pp.savefig(figure=fig.get())

elif args.command == "fly":
    if args.table == None:
        warning("No table is specified, try to use the first.")

    for key, set in load_coef(args.coef, args.table):
        coef = CoefLoader(set)

        fig = plt.figure()
        ax = plt.subplot(1, 1, 1, projection="polar", theta_offset=math.pi / 2)

        thetas2, rs2, timed_zs2 = prepare_pie_frame(coef)

        def draw_frame(t):
            draw_pie_frame(coef, fig, ax, thetas2, rs2, timed_zs2, t)
            ax.set_title(f"{key} LogPie L={coef.nt} Z={len(coef.zo)} t={t}")

        ani = FuncAnimation(
            fig,
            draw_frame,
            np.linspace(coef.tmin, coef.tmax, int(math.ceil(coef.tbins)) + 1),
        )
        writer = PillowWriter(fps=30)
        ani.save(args.opt, writer=writer)

        break
elif args.command == "qfly":
    pass
elif args.command == "validate":
    keys = []
    score = []
    concat = ConcatInfo(args.concat, args.r_max)
    for key, set in load_coef(args.coef, args.table):
        print(f"Processing {key}...")
        coef = CoefLoader(set)
        s = Validate(coef, concat)
        print(key, ":", s)
        keys.append(key)
        score.append(s)
    df = pd.DataFrame(np.array(score)[:, np.newaxis].T, columns=keys)
    df.T.to_csv(args.opt, header=False)
else:
    raise argparse.ArgumentError(args.command, "Invalid command")
