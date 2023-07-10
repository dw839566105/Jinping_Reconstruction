import tables
import numpy as np
import matplotlib.pyplot as plt

plt.figure(dpi=300)
x = np.linspace(0, np.pi/2,300)
[p3,] = plt.plot(np.sin(x), np.cos(x), c='k', ls='--', label='Acrylic')
s1 = plt.scatter(0,0.62/0.65, marker='*', s=100, c='r', label='vertex')
s2 = plt.scatter(np.sqrt(0.832**2-0.700**2)/0.65, 0.700/0.65, marker='^', s=100, c='k', label='PMT')
plt.gca().set_aspect(1)
[p4,] = plt.plot([0.06/0.65, 0.06/0.65], [np.sqrt(1-(0.06/0.65)**2), 1.4], c='k', ls='-.', label='outlet')
for i in np.arange(11, 17):
    plt.plot([0,np.sin(x[i])], [0.62/0.65, np.cos(x[i])], c='k', alpha=0.1)
    [p1,] = plt.plot([np.sqrt(0.832**2-0.700**2)/0.65,np.sin(x[i])], [0.700/0.65, np.cos(x[i])], c='k', alpha=0.3, label='Shielded tracks')
#for i in np.arange(20, 25):
#    plt.plot([0,np.sin(x[i])], [0.62/0.65, np.cos(x[i])], c='r', alpha=0.1)
#    [p2,] = plt.plot([np.sqrt(0.832**2-0.700**2)/0.65,np.sin(x[i])], [0.700/0.65, np.cos(x[i])], c='r', alpha=0.3, label='Tracks(Outlet)')
    
plt.plot([0,0],[0,0.62/0.65], lw=1, c='k', ls='dotted')
plt.plot([0, 0.06/0.65],[0, np.sqrt(1-(0.06/0.65)**2)], lw=1, c='k', ls='dotted')
ls = np.linspace(0, np.arcsin(0.06/0.65), 100)

plt.plot(0.1*np.sin(ls), 0.1*np.cos(ls), lw=1, c='k')
plt.text(0.02,0.1, r'$\theta_\mathrm{acr}$', fontsize=20)
# leg = plt.legend(handles=[p1, p2, p3, p4, s1, s2], loc=1, fontsize=18)

plt.xlabel('$x/r_{\mathrm{LS}}$')
plt.ylabel('$y/r_{\mathrm{LS}}$')
plt.xlim(-0.05,2)
plt.ylim(-0.05,1.4)
ax = plt.gca()
axins = ax.inset_axes([0.2, 0.02, 0.8, 0.4])

x = np.linspace(0, np.pi/2,300)
[p3,] = axins.plot(np.sin(x), np.cos(x), c='k', ls='--', label='acrylic')
s1 = axins.scatter(0,0.62/0.65, marker='*', s=100, c='r', label='vertex')
s2 = axins.scatter(np.sqrt(0.832**2-0.700**2)/0.65, 0.700/0.65, marker='^', s=100, c='k', label='PMT')

[p2,] = axins.plot([0.06/0.65, 0.06/0.65], [np.sqrt(1-(0.06/0.65)**2), 1.1], c='k', ls='-.', label='General')
for i in np.arange(11, 17):
    axins.plot([0,np.sin(x[i])], [0.62/0.65, np.cos(x[i])], c='k', alpha=0.1)
    axins.plot([np.sqrt(0.832**2-0.700**2)/0.65,np.sin(x[i])], [0.700/0.65, np.cos(x[i])], c='k', alpha=0.3, label='Tracks(General)')
# for i in np.arange(20, 25):
#    axins.plot([0,np.sin(x[i])], [0.62/0.65, np.cos(x[i])], c='r', alpha=0.1)
#    [p2,] = axins.plot([np.sqrt(0.832**2-0.700**2)/0.65,np.sin(x[i])], [0.700/0.65, np.cos(x[i])], c='r', alpha=0.3, label='Tracks(Outlet)')
cth = np.arange(0.98, 1.01, 0.01)
#axins.scatter(np.sqrt(1-cth**2), cth, marker='+', c='k', s=100)
#axins.scatter(np.sqrt(1-np.cos(0.05)**2), np.cos(0.05), marker='+', c='k', s=100)
#axins.text(np.sqrt(1-np.cos(0.05)**2)-0.03, np.cos(0.05) - 0.06, '0.05', fontsize=15)
axins.scatter(np.sqrt(1-np.cos(0.10)**2), np.cos(0.10), marker='+', c='k', s=100)
axins.text(np.sqrt(1-np.cos(0.10)**2)-0.03, np.cos(0.10) - 0.03, '0.10', fontsize=15)
#axins.scatter(np.sqrt(1-np.cos(0.15)**2), np.cos(0.15), marker='+', c='k', s=100)
#axins.text(np.sqrt(1-np.cos(0.15)**2)-0.03, np.cos(0.15) - 0.06, '0.15', fontsize=15)
#axins.text(np.sqrt(1-0.98**2), 0.98 - 0.06, '0.98', fontsize=15)
axins.text(0-0.01, 1 -0.03, '0', fontsize=15)
axins.scatter(np.sqrt(1-np.cos(0)**2), np.cos(0), marker='+', c='k', s=100)
# axins.hist(data, bins=np.linspace(-1,1,201), weights=np.ones(data.shape)/Evt.max()*100/30, histtype='step', color='k', label='Truth')
# axins.plot(np.linspace(-1,1,201), np.exp(LG.legval(np.linspace(-1,1,201), coeff)), c='r', linewidth=1, label='Fit', alpha=0.7)
# sub region of the original image
x1, x2, y1, y2 = -0.02, 0.75, 0.95, 1.1
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])
ax.indicate_inset_zoom(axins, edgecolor="black")
leg = plt.legend(handles=[p1, p3, p4, s1, s2], loc=1, fontsize=18)
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.savefig('outlet.pdf')
