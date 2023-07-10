import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

def calc_probe(ddx, ddy, tir=False):
    v = np.vstack((ddx, ddy)).T
    dist = np.linalg.norm(PMT - v, axis=1)
    if tir:
        a0 = (v[:, 1] - PMT[:,1])/(v[:,0] - PMT[:, 0])
        a1 = - v[:, 0] * a0 + v[:, 1]

        a = a0**2 + 1
        b = 2*a0*a1
        c = a1**2 - 1**2

        delta = np.sqrt(b**2 - 4*a*c)
        x1 = (-b - delta)/2/a
        x2 = (-b + delta)/2/a
        intercept = np.vstack((x2, a0*x2 + a1)).T

        cth = np.sum((intercept-v)*intercept, axis=1)/np.linalg.norm((intercept-v), axis=1)
        cth = np.nan_to_num(cth, nan=1)
        th1 = np.arccos(np.clip(cth, -1, 1))
        th2 = np.arcsin(np.sin(th1)*1.5/1.33)
        t_ratio = 2*1.47*cth/(1.47*cth + 1.33*np.cos(th2))
        tr = 1 - (t_ratio-1)**2
        return cth/dist**2*np.nan_to_num(tr)
    else:
        return dist**2

N = 20
PMT = 1.3 * np.array([(0,1),(np.sin(2*np.pi/N), np.cos(2*np.pi/N))])

x, y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))

xx = np.linspace(-1,1,500)
yy = np.linspace(-1,1,500)
theta = np.linspace(-np.pi, np.pi, 200)

with PdfPages('App.pdf') as pdf:
    for i in [True, False]:
        plate = np.empty((len(xx), len(yy), len(PMT)))
        for x_index, x in enumerate(tqdm(xx)):
            for y_index, y in enumerate(yy):
                
                probe = calc_probe(y, x, tir=i).T
                if(x**2 + y**2)>1:
                    plate[x_index, y_index] = np.nan
                else:
                    plate[x_index, y_index] = np.nan_to_num(probe, nan=1e-6)
        fig = plt.figure(dpi=200)      
        cs = plt.contour(xx, yy, np.log(plate[:,:,1]/plate[:,:,0]), 
            levels = np.arange(-1,1,0.2), 
            cmap='jet')
        plt.gca().clabel(cs, inline=1, fontsize=10)
        plt.axis('equal')
        
        plt.plot(np.cos(theta), np.sin(theta), 'k', label='Acrylic')
        plt.plot(0.7*np.cos(theta), 0.7*np.sin(theta), color='grey', linestyle='--')
        plt.scatter(PMT[:,0], PMT[:,1], marker='*', c='r', label='PMT')
        plt.axvline(0, color='k', linestyle='--', label = 'x=0')
        plt.legend()
        plt.xlabel('x/R')
        plt.ylabel('y/R')
        pdf.savefig(fig)

    ## tr
    ts = np.linspace(25*np.pi/90, 32*np.pi/90, 7)
    fig = plt.figure(dpi=300)
    plt.axis('equal')
    for i in ts:
        px = np.linspace(0,1.2,100)
        nx = np.linspace(-0.5,0,100)
    
        inc = i
        ref = np.arcsin(np.sin(i)*1.5/1.33)
        t = (2*1.33*np.cos(inc))/(1.33*np.cos(inc) + 1.5*np.cos(ref))
        t_ratio = np.nan_to_num(1-(t-1)**2)
        p1 = plt.plot(nx, 1/np.tan(inc)*nx + 1,)
        tt = 1/np.tan(ref)*px + 1
        p = plt.plot(px, tt, color=p1[-1].get_color(), alpha=t_ratio**2)
        plt.text(px[-1], np.nan_to_num(tt[-1], nan=1), '%.2f' % t_ratio, color=p[-1].get_color())
    
    plt.plot(np.cos(theta), np.sin(theta), 'k', label='Acrylic')

    xx = np.linspace(-0.8,0,10)
    plt.fill_between(xx, np.sqrt(1.5**2 - 1.33**2)/1.33*xx+1, np.sqrt(1-xx**2), alpha=0.2, label='TIR region')

    plt.axvline(0, color='grey', linestyle='--')
    plt.axhline(1, color='grey', linestyle='--')

    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.8)
    plt.xlabel('x/R')
    plt.ylabel('y/R')
    plt.legend(loc=2)
    pdf.savefig(fig)


    # tr
    n1 = 1
    n2 = np.linspace(1.00, 2, 100)
    fig = plt.figure(dpi=300)
    for index, ref in enumerate(np.linspace(0,np.pi/2 - 0.01,10)):
        inc = np.sin(ref)*n1/n2
        t = 2*n2*np.cos(inc)/(n2*np.cos(inc) + n1*np.cos(ref))
        T = np.nan_to_num(1-(t-1)**2)
        p1 = plt.plot(n2, T, alpha=0.3)
        plt.text(n2[-index*10-5], T[-index*10-5], f'{ref:.2f}', color=p1[0].get_color())
        
    ref = np.arcsin(n1/n2)
    inc = np.sin(ref)*n1/n2
    t = 2*n2*np.cos(inc)/(n2*np.cos(inc) + n1*np.cos(ref))
    T = np.nan_to_num(1-(t-1)**2)
    plt.plot(n2, T,  'k--', label='arcsin($n_1$/$n_2$)')
    plt.axhline(0.9, color='r', linestyle='--', label='$y$=0.9')
    plt.legend()
    plt.xlabel('$n_2/n_1$')
    plt.ylabel('Transmittance')
    pdf.savefig(fig)

    # optimize    
    fig = plt.figure(dpi=300)
    N = 5
    PMT = 1.3 * np.array([(0,1),(np.sin(np.pi/N), np.cos(np.pi/N))])

    x, y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))

    xx = np.linspace(-2,2,100)
    yy = np.linspace(-2,2,100)
    plate = np.empty((len(xx), len(yy), len(PMT)))

    a = np.linalg.norm(PMT[0])
    b = np.linalg.norm(PMT[1])
    k = 10/11
    center = (PMT[0] - k**2*PMT[1])/(1-k**2)
    ar = k/(1-k**2)*np.linalg.norm(PMT[0] - PMT[1])
    plt.axis('equal')
    theta = np.linspace(-np.pi, np.pi, 200)
    plt.plot(np.cos(theta), np.sin(theta), 'k', label='Acrylic')
    plt.plot(0.9*np.cos(theta), 0.9*np.sin(theta), color='grey', linestyle='--', linewidth=0.5)
    plt.plot(center[0] + ar*np.cos(theta), center[1] + ar*np.sin(theta), 'b', label='contour', 
            linestyle='--', linewidth =0.5, alpha=0.5)

    c1 = np.linspace(0,np.pi/10,3)
    for i,j in zip(c1[:-1], c1[1:]):
        plt.gca().annotate("",
            xytext=(0.9*np.sin(i), 0.9*np.cos(i)),
            xy=(0.9*np.sin(j), 0.9*np.cos(j)),
            arrowprops=dict(arrowstyle="-|>",),
            size=1,
        )

    c2 = np.linspace(62.5*np.pi/100,66.8*np.pi/100,5)
    for i,j in zip(c2[:-1], c2[1:]):
        plt.gca().annotate("",
            xytext=(center[0] + ar*np.sin(i), center[1] + ar*np.cos(i)),
            xy=(center[0] + ar*np.sin(j), center[1] + ar*np.cos(j)),
            arrowprops=dict(arrowstyle="-|>",),
            size=1,
        )
    x = np.linspace(-0.8,0.36,100)
    y1 = (x-PMT[1,0])*0.3+PMT[1,1]
    y2 = np.sqrt(1-x**2)
    plt.plot(x, y1)
    plt.fill_between(x, y1, y2, alpha=0.1, label='TIR')
    plt.scatter(0, 0.9, color='r', marker='s', label='initial')
    plt.scatter(0, 0.4, color='r', marker='*', label='Truth')
    plt.scatter(PMT[:,0], PMT[:,1], marker='^', label='PMT')
    plt.axvline(0, color='c', linestyle='--', linewidth=0.5, c='grey')
    plt.xlim([-1.2,1.2])
    plt.ylim([-1.2,1.55])

    plt.legend()
    plt.xlabel('x/R')
    plt.ylabel('y/R')
    pdf.savefig(fig)