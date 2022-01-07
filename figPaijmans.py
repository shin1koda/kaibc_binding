import numpy as np
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def getFigPaijmans(figname):
    print("creating "+figname)

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.size'] = 8
    matplotlib.rcParams['font.family'] = 'Arial'
    fig = plt.figure(tight_layout=True,figsize=(5.2,2.0))
    gs = fig.add_gridspec(1,2,width_ratios=[2.2,1])
    subgs = gs[0].subgridspec(1,2,wspace=0)
    ax0 = fig.add_subplot(subgs[0])
    ax1 = fig.add_subplot(subgs[1])
    ax0.set_xlim(-1,17)
    ax1.set_xlim(-1,17)
    ax0.set_xticks(np.arange(0,18,4))
    ax1.set_xticks(np.arange(0,18,4))
    ax1.set_yticks([])
    ax0.set_xlabel('Time (h)')
    ax1.set_xlabel('Time (h)')
    ax0.set_ylabel('Bound KaiB/KaiC')
    ax2 = fig.add_subplot(gs[1])
    ax2.set_ylim(-0.05,1.55)
    ax2.set_yticks(np.arange(0,1.6,0.5))
    ax2.set_xlabel('Bound KaiB/KaiC')
    ax2.set_ylabel('Binding rate (h$^{-1}$)')

    a1s=np.zeros(7)
    g1s=np.zeros(7)
    a1errs=np.zeros(7)
    g1errs=np.zeros(7)
    for mode in ['default','modified']:
        c=iter(matplotlib.colors.TABLEAU_COLORS.items())
        for p in range(7):
            cname=next(c)[0]
            filename=f'paijmans/{mode}_wKaiB_p{p}.dat'
            t=[]
            b=[]
            with open(filename) as f:
                for line in f:
                    if '#' in line: continue
                    nums=line.split()
                    t.append(float(nums[0]))
                    b.append(float(nums[4]))
            t=np.array(t)
            b=np.array(b)

            def func1(x,a1,g1):
                return a1*(1.0-np.exp(-g1*x))

            popt,pcov=curve_fit(func1,t,b)
            #bopt=func1(t,*popt)
            perr = np.sqrt(np.diag(pcov))
            a1s[p]=popt[0]
            g1s[p]=popt[1]
            a1errs[p]=perr[0]
            g1errs[p]=perr[1]

            #print(popt)
            #print(perr)
            if mode=='default':
              ax=ax0
            else:
              ax=ax1
            ax.plot(t,b,color=cname,label=str(p))

        if mode=='default':
          shp='o'
          l='A'
        else:
          shp='D'
          l='B'
        ax2.plot(a1s,g1s,shp,mew=1., mec='k',ms=5,label=l)

    ax0.text(0.1, 0.9,'A',transform=ax0.transAxes,fontweight='bold',
             horizontalalignment="left",
             verticalalignment="center" )
    ax1.text(0.1, 0.9,'B',transform=ax1.transAxes,fontweight='bold',
             horizontalalignment="left",
             verticalalignment="center" )
    ax2.text(0.2, 0.9,'C',transform=ax2.transAxes,fontweight='bold',
             horizontalalignment="left",
             verticalalignment="center" )
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,title='#D')
    ax2.legend(loc='upper right', title='Panel')
    fig.savefig(figname,dpi=300,bbox_inches='tight')


if __name__ == '__main__':
    getFigPaijmans('fig5.tif')
