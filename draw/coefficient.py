import numpy as np
import h5py as h5
import math
from zernike import RZern
from polynomial import *

class CoefLoader:
    def __init__(self, set: h5.Dataset):
        self.coef = np.atleast_2d(set[()])
        shape = self.coef.shape
        self.nt = shape[0]
        nr = shape[1]
        zlorder = math.ceil(math.sqrt(nr * 4))
        self.cart = RZern(zlorder + 1)
        self.zo = np.where(self.cart.mtab >= 0)[0][:nr]
        self.tmin = set.attrs["t_min"]
        self.tmax = set.attrs["t_max"]
        self.tbins = self.tmax - self.tmin
        self.type = set.attrs["type"]

    def time_to_legendre(self, t):
        return (t - self.tmin) / self.tbins * 2 - 1

    def legendre_to_time(self, t):
        return (t + 1) * self.tbins / 2 + self.tmin

    def get_legendre(self, ts):
        lgorder = np.sqrt((2 * np.arange(self.nt) + 1) / 2)
        return legval(ts, self.nt) * lgorder[:, np.newaxis]

    def get_zernike(self, rs, thetas):
        return radial(self.cart.coefnorm, self.cart.rhotab, self.zo, rs) * angular(
            self.cart.mtab[self.zo], thetas
        )

    def get_zernike_grid(self, rs, thetas):
        return (
            radial(self.cart.coefnorm, self.cart.rhotab, self.zo, rs)[:, :, np.newaxis]
            * angular(self.cart.mtab[self.zo], thetas)[:, np.newaxis, :]
        )


def load_coef(filename: str, table):
    with h5.File(filename, "r", swmr=True) as file:
        if table != None:
            for key in table:
                yield key, file[key]
        else:
            for key in file.keys():
                yield key, file[key]


class ConcatInfo:
    def __init__(self, filename: str, r_max: float):
        with h5.File(filename, "r", swmr=True) as file:
            concat = file["Concat"][()]
            self.pe_rs = concat["r"] / r_max
            self.pe_thetas = concat["theta"]
            self.pe_ts = concat["t"]

            self.f_pe_rs = np.hstack([self.pe_rs, self.pe_rs])
            self.f_pe_thetas = np.hstack([self.pe_thetas, 2 * np.pi - self.pe_thetas])
            self.f_pe_ts = np.hstack([self.pe_ts, self.pe_ts])

            vertices = file["Vertices"][()]
            self.v_rs = vertices["r"] / r_max
            self.v_thetas = vertices["theta"]
            self.f_v_rs = np.hstack([self.v_rs, self.v_rs])
            self.f_v_thetas = np.hstack([self.v_thetas, 2 * np.pi - self.v_thetas])
