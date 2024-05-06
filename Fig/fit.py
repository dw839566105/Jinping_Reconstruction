import numpy as np
from scipy.optimize import curve_fit

def gauss(x,mu,sigma,A,B):
    return A*(np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))) + B

def fitdata(data, start, end, base):
    #count = np.histogram(data,bins = int(np.max(data)-np.min(data)),range = [np.min(data),np.max(data)])
    data = np.array(data,dtype=np.float64)
    count = np.histogram(data, bins=100, range=[np.min(data),np.max(data)])
    x = 0.5*(count[1][1:]+count[1][:-1])
    cut = (x > start) & (x < end)
    count[0][~cut] = 0
    mu = x[np.argmax(count[0])]
    std = np.std(data)

    p0 = [mu,std,np.max(count[0]),0]
    if mu > 0:
        bound = ([0.1*mu,0.1*std,0.1*np.max(count[0]),-base],[10*mu,10*std,10*np.max(count[0]),base])
    else:
        bound = ([10*mu,0.1*std,0.1*np.max(count[0]),-base],[0.1*mu,10*std,10*np.max(count[0]),base])
    try:
        popt,pcov = curve_fit(gauss,x[(x<mu+3*std)&(x>mu-3*std)],
                            count[0][(x<mu+3*std)&(x>mu-3*std)],
                            p0=p0,bounds=bound)
    except:
        breakpoint()

    mu,std,A,B = popt[0],popt[1],popt[2],popt[3]
    p0 = [mu,std,A,B]
    if mu > 0:
        bound = ([0.5*mu,0.5*std,0.5*A,-base],[2*mu,2*std,2*A,base])
    else:
        bound = ([2*mu,0.5*std,0.5*A,-base],[0.5*mu,2*std,2*A,base])
    try:
        popt,pcov = curve_fit(gauss,x[(x<mu+3*std)&(x>mu-3*std)],
                            count[0][(x<mu+3*std)&(x>mu-3*std)],
                            p0=p0,bounds=bound)
    except:
        breakpoint()

    return x,popt,pcov
