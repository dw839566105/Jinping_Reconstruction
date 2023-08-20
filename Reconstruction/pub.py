import numpy as np
import h5py, tables
from polynomial import *

class Recon(tables.IsDescription):
    EventID = tables.Int64Col(pos=0)    # EventNo
    x = tables.Float16Col(pos=0)        # x position
    y = tables.Float16Col(pos=1)        # y position
    z = tables.Float16Col(pos=2)        # z position
    E = tables.Float16Col(pos=3)        # Energy
    t = tables.Float16Col(pos=4)        # time
    success = tables.Int64Col(pos=6)    # recon status
    Likelihood = tables.Float16Col(pos=7)

class load_coeff:
    def load_coeff_Single(PEFile, TimeFile):
        # spherical harmonics coefficients for time and PE make
        with tables.open_file(PEFile, 'r') as h:
            coeff_pe = h.root.coeff[:]
            pe_type = h.root.coeff.attrs['type']

        with tables.open_file(TimeFile,'r') as h:
            coeff_time = h.root.coeff[:]
            time_type = h.root.coeff.attrs['type']
        return coeff_pe, coeff_time, pe_type, time_type
    
    def load_coeff_Probe(File = '/mnt/stage/probe/unbinned/ode/shrink/0-240.h5'):
        with h5py.File(File, "r") as ipt:
            for key in ipt.keys(): # enum all keys
                coeff = ipt[key]
            t_min = coeff.attrs['t_min']
            t_max = coeff.attrs['t_max']
            coef = coeff[()]
        return coef, t_min, t_max

def r2c(c):
    # coordinate transformation
    v = np.zeros(3)
    v[2] = c[0] * np.cos(c[1]) #z
    rho = c[0] * np.sin(c[1])
    v[0] = rho * np.cos(c[2]) #x
    v[1] = rho * np.sin(c[2]) #y
    return v

def c2r(c):
    # coordinate transformation
    v = np.zeros(3)
    v[0] = np.linalg.norm(c)
    v[1] = np.arccos(c[2]/(v[0]+1e-6))
    v[2] = np.arctan2(c[1],c[0])
    return v

class LH_Zer:
    def Likelihood(vertex, *args, expect=False):
        '''
        vertex[1]: r
        vertex[2]: theta
        vertex[3]: phi
        '''
        PMT_pos, fired_PMT, time_array, pe_array, coeff_pe, coeff_time, cart = args
        basis_pe, basis_time = LH.Calc_basis(vertex, PMT_pos, cart)
        L1, energy = LH.Likelihood_PE(basis_pe, pe_array, coeff_pe)
        if expect:
            return energy
        else:
            L2 = LH.Likelihood_Time(basis_time, vertex[-1], fired_PMT, time_array, coeff_time)
            return L1 + L2

    def Calc_basis(vertex, PMT_pos, cart): 
        # boundary
        v = vertex[:3]
        rho = np.linalg.norm(v)
        if rho > 1-1e-3:
            rho = 1-1e-3
        # calculate cos theta
        cos_theta = np.dot(v, PMT_pos.T) / (np.linalg.norm(v)*np.linalg.norm(PMT_pos,axis=1))
        cos_theta = np.nan_to_num(cos_theta)
        theta = np.arccos(cos_theta)
        # Generate Zernike basis
        rho = rho + np.zeros(len(PMT_pos))
        zo = cart.mtab>=0
        zs_radial = cart.coefnorm[zo, np.newaxis] * polyval(cart.rhotab.T[:, zo, np.newaxis], rho)
        zs_angulars = angular(cart.mtab[zo].reshape(-1, 1), theta)
        return (zs_radial * zs_angulars).T, ((zs_radial * zs_angulars).T)[:, :np.sum((cart.ntab<=15) & zo)]

    def Likelihood_PE(basis, pe_array, coeff_pe):
        expect = np.exp(np.matmul(basis, coeff_pe))

        # Energy fit
        nml = np.sum(pe_array)/np.sum(expect)
        expect *= nml

        # Poisson likelihood of Bayesian Network
        # p(q|lambda) = sum_n p(q|n)p(n|lambda)
        #         = sum_n Gaussian(q, n, sigma_n) * exp(-expect) * expect^n / n!
        # int p(q|lambda) dq = sum_n exp(-expect) * expect^n / n! = 1

        lnL = - pe_array * np.log(expect) + expect
        return lnL.sum(), nml


    def Likelihood_Time(basis, T0, fired_PMT, time_array, coeff_time):
        basis_time = basis[fired_PMT]
        # Recover coefficient
        T_i = np.matmul(basis_time, coeff_time)
        T_i = T_i + T0
        lnL = np.nansum(LH.Likelihood_quantile(time_array, T_i, 0.1, 3))
        return lnL.sum()


    def Likelihood_quantile(y, T_i, tau, ts):
        # less = T_i[y<T_i] - y[y<T_i]
        # more = y[y>=T_i] - T_i[y>=T_i]
        # R = (1-tau)*np.sum(less) + tau*np.sum(more)
        # since lucy ddm is not sparse, use PE as weight
        L = (T_i-y) * (y<T_i) * (1-tau) + (y-T_i) * (y>=T_i) * tau
        return L/ts
    
class LH_Leg:
    def Likelihood(vertex, *args, expect=False):
        '''
        vertex[1]: r
        vertex[2]: theta
        vertex[3]: phi
        '''
        PMT_pos, fired_PMT, time_array, pe_array, coeff_pe, coeff_time, cart = args
        rho, basis = LH_Leg.Calc_basis(vertex, PMT_pos, coeff_pe)
        L1, energy = LH_Leg.Likelihood_PE(rho, basis, pe_array, coeff_pe)
        if expect:
            return energy
        else:
            L2 = LH_Leg.Likelihood_Time(rho, basis, vertex[-1], fired_PMT, time_array, coeff_time)
            return L1

    def Calc_basis(vertex, PMT_pos, coef): 
        # boundary
        v = vertex[:3]
        rho = np.linalg.norm(v)
        rho = np.clip(rho, 0, 1)
        # calculate cos theta
        cos_theta = np.dot(v, PMT_pos.T) / (np.linalg.norm(v)*np.linalg.norm(PMT_pos,axis=1))
        cos_theta = np.nan_to_num(cos_theta)
        cut = len(coef)
        t_basis = legval(cos_theta, len(coef)).T
        return rho, t_basis

    def Likelihood_PE(rho, t_basis, pe_array, coef):
        rhof = rho + np.zeros_like(pe_array)
        r_basis = legval_raw(rhof, coef.T.reshape(coef.shape[1], coef.shape[0],1)).T
        expect = np.exp((t_basis*r_basis).sum(-1))

        # Energy fit
        nml = np.sum(pe_array)/np.sum(expect)
        expect *= nml

        # Poisson likelihood
        # p(q|lambda) = sum_n p(q|n)p(n|lambda)
        #         = sum_n Gaussian(q, n, sigma_n) * exp(-expect) * expect^n / n!
        # int p(q|lambda) dq = sum_n exp(-expect) * expect^n / n! = 1

        lnL = - pe_array * np.log(expect) + expect
        return lnL.sum(), nml


    def Likelihood_Time(rho, t_basis, T0, fired_PMT, time_array, coef):
        rhof = rho + np.zeros_like(fired_PMT)
        basis_time = t_basis[fired_PMT]
        r_basis = legval_raw(rhof, coef.T.reshape(coef.shape[1], coef.shape[0],1)).T
        T_i = (t_basis[fired_PMT, :coef.shape[0]]*r_basis).sum(-1)
        T_i = T_i + T0
        lnL = np.nansum(LH_Leg.Likelihood_quantile(time_array, T_i, 0.1, 3))
        return lnL.sum()


    def Likelihood_quantile(y, T_i, tau, ts):
        # less = T_i[y<T_i] - y[y<T_i]
        # more = y[y>=T_i] - T_i[y>=T_i]
        # R = (1-tau)*np.sum(less) + tau*np.sum(more)
        # since lucy ddm is not sparse, use PE as weight
        L = (T_i-y) * (y<T_i) * (1-tau) + (y-T_i) * (y>=T_i) * tau
        return L/ts
    
class construct_Zer:
    def __init__(self, coeff_pe, PMT, r, cart):
        theta = np.arccos(np.linspace(-1, 1, 50))
        phi = np.linspace(0, 2*np.pi, 50)
        self.mesh = construct_Zer.meshgrid(r, theta, phi)
        self.tpl = construct_Zer.template(self.mesh, cart, coeff_pe, PMT)

    def meshgrid(r, theta, phi):
        xx, yy, zz = np.meshgrid(r, theta, phi, sparse=False)
        mesh = np.vstack((xx.flatten()*np.sin(yy.flatten())*np.cos(zz.flatten()), \
                    xx.flatten()*np.sin(yy.flatten())*np.sin(zz.flatten()), \
                    xx.flatten()*np.cos(yy.flatten()))).T
        return mesh

    def template(mesh, cart, coeff_pe, PMT_pos):
        zo = cart.mtab >= 0
        tpl = np.zeros((len(mesh), len(PMT_pos)))
        print(len(mesh))
        for i in np.arange(len(mesh)):
            if not i % 10000:
                print('processing:', i)
            vertex = mesh[i]
            cos_theta = np.sum(vertex*PMT_pos, axis=1)/np.linalg.norm(vertex)/np.linalg.norm(PMT_pos, axis=1)
            r = np.linalg.norm(vertex)
            zs_radial = cart.coefnorm[zo, np.newaxis] * polyval(cart.rhotab.T[:, zo, np.newaxis], r*np.ones(len(PMT_pos)))
            zs_angulars = angular(cart.mtab[zo].reshape(-1, 1), np.arccos(cos_theta))
            basis = zs_radial * zs_angulars
            expect = np.exp(np.matmul(basis.T, coeff_pe))
            tpl[i] = expect
        return tpl

class construct_Leg:
    def __init__(self, coeff_pe, PMT, r):
        theta = np.arccos(np.linspace(-1, 1, 50))
        phi = np.linspace(0, 2*np.pi, 50)
        self.mesh = construct_Leg.meshgrid(r, theta, phi)
        self.tpl = construct_Leg.template(self.mesh, coeff_pe, PMT)

    def meshgrid(r, theta, phi):
        xx, yy, zz = np.meshgrid(r, theta, phi, sparse=False)
        mesh = np.vstack((xx.flatten()*np.sin(yy.flatten())*np.cos(zz.flatten()), \
                    xx.flatten()*np.sin(yy.flatten())*np.sin(zz.flatten()), \
                    xx.flatten()*np.cos(yy.flatten()))).T
        return mesh

    def template(mesh, coef, PMT_pos):
        tpl = np.zeros((len(mesh), len(PMT_pos)))
        print(len(mesh))
        cut = len(coef)
        
        for i in np.arange(len(mesh)):
            if not i % 10000:
                print('processing:', i)
            vertex = mesh[i]
            cos_theta = np.sum(vertex*PMT_pos, axis=1)/np.linalg.norm(vertex)/np.linalg.norm(PMT_pos, axis=1)
            rhof = np.linalg.norm(vertex) + np.zeros(len(PMT_pos))
            r_basis = legval_raw(rhof, coef.T.reshape(coef.shape[1], coef.shape[0],1)).T
            t_basis = legval(cos_theta, len(coef)).T
            expect = np.exp((t_basis*r_basis).sum(-1))
            tpl[i] = expect
        return tpl
                 
class Initial:
    def ChargeWeighted(pe_array, PMT_pos, time_array):
        vertex = np.zeros(5)
        x_ini = 1.3 * np.sum(np.atleast_2d(pe_array).T*PMT_pos, axis=0) / np.sum(pe_array)
        E_ini = np.sum(pe_array)/60 # 60 PE/MeV
        t_ini = np.quantile(time_array, 0.1) # quantile 0.1
        vertex[0] = E_ini
        vertex[1:4] = x_ini / 0.65
        vertex[-1] = t_ini - 27
        return vertex
    
    def MCGrid(pe_array, mesh, tpl, time_array):
        vertex = np.zeros(5)
        scale = np.sum(tpl, axis=1)/np.sum(pe_array)
        tpl /= np.atleast_2d(scale).T
        L = -np.nansum(-tpl + np.log(tpl)*pe_array, axis=1)
        index = np.where(L == np.min(L))[0][0]
        
        x_ini = mesh[index]
        E_ini = np.sum(pe_array)/60
        t_ini = np.quantile(time_array, 0.1)
        vertex[0] = E_ini
        vertex[1:4] = c2r(x_ini/1000)
        vertex[1] /= 0.65
        vertex[-1] = t_ini - 27
        return vertex
    
    def FitGrid(pe_array, mesh, tpl, time_array):
        vertex = np.zeros(5)
        scale = np.sum(tpl, axis=1)/np.sum(pe_array) # fit energy
        tpl /= np.atleast_2d(scale).T
        L = -np.nansum(-tpl + np.log(tpl)*pe_array, axis=1) # pe likelihood
        index = np.argmin(L) # min position

        x_ini = mesh[index]
        E_ini = np.sum(pe_array)/60
        t_ini = np.quantile(time_array, 0.1)
        vertex[0] = E_ini
        vertex[1:4] = x_ini
        vertex[-1] = t_ini - 27
        return vertex
