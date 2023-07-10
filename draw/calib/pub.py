import tables
import numpy as np
from numba import jit

def LoadBase(file=r'./base.h5'):
    '''
    # to vanish the PMT difference, just a easy script
    # output: relative different bias
    '''
    
    h1 = tables.open_file(file)
    base = h1.root.correct[:]
    h1.close()
    return base

def ReadJPPMT(file=r"./PMT.txt"):
    '''
    # Read PMT position
    # output: 2d PMT position 30*3 (x, y, z)
    '''
    f = open(file)
    line = f.readline()
    data_list = [] 
    while line:
        num = list(map(float,line.split()))
        data_list.append(num)
        line = f.readline()
    f.close()
    PMT_pos = np.array(data_list)
    return PMT_pos


@jit(nopython=True)
def legval(x, c):
    """
    stole from the numerical part of numpy.polynomial.legendre

    """
    if len(c) == 1:
        return c[0]
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1*(nd - 1))/nd
            c1 = tmp + (c1*x*(2*nd - 1))/nd
    return c0 + c1*x

def LegendreCoeff(PMT_pos, vertex, cut, Legendre=True):
    '''
    # calulate the Legendre value of transformed X
    # input: PMT_pos: PMT No * 3
          vertex: 'v' 
          cut: cut off of Legendre polynomial
    # output: x: as 'X' at the beginnig    
    
    '''
    size = np.size(PMT_pos[:,0])
    # oh, it will use norm in future version
    
    if(np.sum(vertex**2) > 1e-6):
        cos_theta = np.sum(vertex*PMT_pos,axis=1)\
            /np.sqrt(np.sum(vertex**2, axis=1)*np.sum(PMT_pos**2,axis=1))
    else:
        # make r=0 as a boundry, it should be improved
        cos_theta = np.zeros(size)
    
    if(np.sum(np.isnan(cos_theta))):
        print('NaN value in cos theta!')
    cos_theta = np.nan_to_num(cos_theta)
    if(Legendre):
        x = legval(cos_theta, np.eye(cut).reshape((cut,cut,1))).T
        print(PMT_pos.shape, x.shape, cos_theta.shape)
        return x, cos_theta
    else:
        return cos_theta

def repeat(a0):
    # not suit for end with 1
    end = a0[np.roll(a0==1,-1)*(a0!=1)]
    end0 = a0.copy()
    end_index = np.where(np.roll(a0==1,-1)*(a0!=1))
    a = a0.copy()
    for i in np.arange(len(end_index[0])-1):
        a[end_index[0][i]+1:end_index[0][i+1]+1] += end[i]
        end[i+1:] += a0[end_index[0][i]]
    return a

def ReadFile(filename, cut):
    '''
    # Read single file
    # input: filename [.h5]
    # output: EventID, ChannelID, x, y, z
    '''
    h1 = tables.open_file(filename,'r')
    print(filename, flush=True)
    truthtable = h1.root.GroundTruth
    EventID0 = truthtable[:]['EventID']
    index1 = EventID0 < cut
    EventID = repeat(EventID0[index1])
    ChannelID = truthtable[:]['ChannelID'][index1]
    PETime = truthtable[:]['PETime'][index1]
    photonTime = truthtable[:]['photonTime'][index1]
    PulseTime = truthtable[:]['PulseTime'][index1]
    dETime = truthtable[:]['dETime'][index1]
    
    ID = h1.root.TruthData[:]['ID']
    index2 = ID < cut
    Q = h1.root.PETruthData[:]['Q']
    Q = Q.reshape(-1,30)[index2].flatten()
    x = h1.root.TruthData[:]['x'][index2]
    y = h1.root.TruthData[:]['y'][index2]
    z = h1.root.TruthData[:]['z'][index2]
    h1.close()
    print('read complete!')
    breakpoint()
    if(np.float(np.size(np.unique(EventID)))!= Q.shape[0]/30):
        print('Event do not match!')
        exit()
    
    return (EventID, ChannelID, Q, PETime, photonTime, PulseTime, dETime, x, y, z)

def ReadChain(path, axis):
    '''
    # This program is to read series files
    # Since root file will recorded as 'filename.root', if too large, it will use '_n' as suffix
    # input: radius: %+.3f, 'str'
    #        path: file storage path, 'str'
    #        axis: 'x' or 'y' or 'z', 'str'
    # output: the gathered result EventID, ChannelID, x, y, z
    '''
    EventID = np.zeros(0)
    Q = np.zeros(0)
    x = np.zeros(0)
    y = np.zeros(0)
    z = np.zeros(0)
    for radius in np.arange(0, 0.55, 0.01):
        # filename = path + '1t_' + radius + '.h5'
        # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030.h5
        filename = '%s1t_+%.3f_randQ.h5' % (path, radius)
        #EventID, Q, x, y, z = readfile(filename)
        if(len(EventID)>0):
            EventID1, Q1, x1, y1, z1 = readfile(filename) + np.max(EventID)
        else:
            EventID1, Q1, x1, y1, z1 = readfile(filename)
        EventID = np.hstack((EventID, EventID1))
        Q = np.hstack((Q, Q1))
        x = np.hstack((x, x1))
        y = np.hstack((y, y1))
        z = np.hstack((z, z1))
    return EventID, Q, x, y, z

def CalibPE(theta, *args):
    '''
    # core of this program
    # input: theta: parameter to optimize
    #      *args: include 
          total_pe: [event No * PMT size] * 1 vector ( used to be a 2-d matrix)
          PMT_pos: PMT No * 3
          cut: cut off of Legendre polynomial
          LegendreCoeff: Legendre value of the transformed PMT position (Note, it is repeated to match the total_pe)
    # output: L : likelihood value
    '''
    Q, PMTPos, X = args
    y = Q
    # if offset:
    # corr = np.dot(X, theta) + np.tile(base, (1, np.int(np.size(X)/np.size(base)/np.size(theta))))[0,:]
    LogExpect = np.dot(X, theta)
    # Poisson regression as a log likelihood
    # https://en.wikipedia.org/wiki/Poisson_regression
    L0 = - np.sum(np.sum(np.transpose(y)*np.transpose(LogExpect) \
        - np.transpose(np.exp(LogExpect))))
    # how to add the penalty? see
    # http://jmlr.csail.mit.edu/papers/volume17/15-021/15-021.pdf
    # the following 2 number is just a good attempt
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet
    rho = 1
    alpha = 0
    L = L0/(2*np.size(y)) + alpha * rho * np.linalg.norm(theta,1) + 1/2* alpha * (1-rho) * np.linalg.norm(theta,2) # elastic net
    return L

def CalibTime(theta, *args):
    EventID, y, X, qt, ts = args
    T_i = np.dot(X, theta)
    # quantile regression
    # quantile = 0.01
    L0 = Quantile(y, T_i, qt, ts, EventID)
    L = L0 + np.sum(np.abs(theta))
    return L0

def Quantile(y, T_i, qt, ts, EventID):
    #less = T_i[y<T_i] - y[y<T_i]
    #more = y[y>=T_i] - T_i[y>=T_i]
    #R = (1-tau)*np.sum(less) + tau*np.sum(more)
    R = (1-qt)*(T_i-y)*(y<T_i) + (qt)*(y-T_i)*(y>=T_i)
    H,edges = np.histogram(EventID, weights = R, bins = np.hstack((np.unique(EventID), np.max(EventID)+1)))
    Q = np.bincount(EventID)
    #L0 = 0
    L = Q[1:]*np.log(qt*(1-qt)/ts) - H/ts
    L0 = np.nansum(L)
    return -L0

def MyHessian(x, fun, *args):
    # hession matrix calulation written by dw, for later uncertainty analysis
    # it not be examed
    # what if it is useful one day
    total_pe, PMT_pos, LegendreCoeff = args
    H = np.zeros((len(x),len(x)))
    h = 1e-6
    k = 1e-6
    for i in np.arange(len(x)):
        for j in np.arange(len(x)):
            if (i != j):
                delta1 = np.zeros(len(x))
                delta1[i] = h
                delta1[j] = k
                delta2 = np.zeros(len(x))
                delta2[i] = -h
                delta2[j] = k

                L1 = - fun(x + delta1, *args)
                L2 = - fun(x - delta1, *args)
                L3 = - fun(x + delta2, *args)
                L4 = - fun(x - delta2, *args)
                H[i,j] = (L1+L2-L3-L4)/(4*h*k)
            else:
                delta = np.zeros(len(x))
                delta[i] = h
                L1 = - fun(x + delta, *args)
                L2 = - fun(x - delta, *args)
                L3 = - fun(x, *args)
                H[i,j] = (L1+L2-2*L3)/h**2                
    return H

