import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import binom
from scipy.optimize import minimize
from scipy.linalg import eigvals
from scipy.linalg import eig
import emcee
from multiprocessing import Pool
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

samplefile='samplesLimByATPase.h5'


BBIND_EXP=pd.read_csv('./bbind_exp.csv')
SIGMA2=0.0025*np.ones_like(BBIND_EXP.Time)
SIGMA2ATPase=np.array([2.0,0.9])
uLimATPase=np.array([14.5,8.9])

def kaibbind(t,x,kh,keI,keC):
    return np.array([-(6.0*keC       )*x[0]             +1.0*kh*x[1],\
            -(5.0*keI+1.0*kh)*x[1]+6.0*keC*x[0]+2.0*kh*x[2],\
            -(4.0*keI+2.0*kh)*x[2]+5.0*keI*x[1]+3.0*kh*x[3],\
            -(3.0*keI+3.0*kh)*x[3]+4.0*keI*x[2]+4.0*kh*x[4],\
            -(2.0*keI+4.0*kh)*x[4]+3.0*keI*x[3]+5.0*kh*x[5],\
            -(1.0*keI+5.0*kh)*x[5]+2.0*keI*x[4]+6.0*kh*x[6],\
            -(        6.0*kh)*x[6]+1.0*keI*x[5]            ])

def initCI(kh,keI):
    return binom.pmf(range(7),6,keI/(keI+kh))

def kaibbind_mono(t,x,kh,keI,keC):
    return [-keC*x[0] +kh*x[1],
            +keC*x[0] -kh*x[1]]

def initCI_mono(kh,keI):
    return binom.pmf(range(2),1,keI/(keI+kh))

def getATPaseFin(kh,keI,keC):
    CI=initCI(kh,keI)
    CI[0]=CI[0]*keI/keC
    CI=CI/np.sum(CI)
    m=np.arange(7, dtype=float)/6.0
    return 24.0*kh*np.sum(m*CI)

def log_likeATPase(params):
    kh, keIWT, keCWT, keIEE, keCEE = params
    ATPaseWT=np.array([24.0*kh*keIWT/(kh+keIWT), getATPaseFin(kh,keIWT,keCWT)])
    diff=np.zeros(2)
    for i in range(2):
        diff[i]=max(ATPaseWT[i]-uLimATPase[i],0.0)
    return -0.5*np.sum(diff**2/SIGMA2ATPase)

def log_like(params):
    kh, keIWT, keCWT, keIEE, keCEE = params
    solWT = solve_ivp(kaibbind, [0,16], initCI(kh,keIWT), method='LSODA',
                      t_eval=BBIND_EXP.Time, args=(kh,keIWT,keCWT))
    solEE = solve_ivp(kaibbind, [0,16], initCI(kh,keIEE), method='LSODA',
                      t_eval=BBIND_EXP.Time, args=(kh,keIEE,keCEE))
    return -0.5*np.sum((solWT.y[0]-BBIND_EXP.WT)**2/SIGMA2)\
           -0.5*np.sum((solEE.y[0]-BBIND_EXP.EE)**2/SIGMA2)

def log_like_mono(params):
    kh, keIWT, keCWT, keIEE, keCEE = params
    solWT = solve_ivp(kaibbind_mono, [0,16], initCI_mono(kh,keIWT), method='LSODA',
                      t_eval=BBIND_EXP.Time, args=(kh,keIWT,keCWT))
    solEE = solve_ivp(kaibbind_mono, [0,16], initCI_mono(kh,keIEE), method='LSODA',
                      t_eval=BBIND_EXP.Time, args=(kh,keIEE,keCEE))
    return -0.5*np.sum((solWT.y[0]-BBIND_EXP.WT)**2/SIGMA2)\
           -0.5*np.sum((solEE.y[0]-BBIND_EXP.EE)**2/SIGMA2)

def log_prior(params):
    if min(params)>0.0 and max(params)<100.0:
        return 0.0
    return -np.inf

def log_prob(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_like(params)

def log_probATPase(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_like(params)+log_likeATPase(params)

def log_prob_mono(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_like_mono(params)

def runMCMC_ATPase():
    print('running MCMC')

    nll = lambda *args: -log_prob(*args)
    initial = [2.0,2.0,0.03,0.5,0.01]
    sol = minimize(nll, initial,method='Powell')

    np.random.seed(100)
    pos = sol.x * np.exp(0.0001 * np.random.randn(32, 5))
    nwalkers, ndim = pos.shape
    #for i in range(nwalkers):
    #    print(log_probATPase(pos[i]))

    filename = samplefile
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probATPase,
                                        pool=pool,backend=backend)
        sampler.run_mcmc(pos,50000, progress=True)


def getSamples():
    print('getting Samples')
    filename = samplefile
    reader = emcee.backends.HDFBackend(filename)

    samples = reader.get_chain(flat=True)
    logp = reader.get_log_prob(flat=True)
    a=np.argmax(logp)
    truth=samples[a]

    tau = reader.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
    logp = reader.get_log_prob(discard=burnin, flat=True, thin=thin)

    print("burnin:",burnin)
    print("thin:",thin)
    print("log_prob:",logp)

    return samples, truth, logp


def getMinEigen():
    print("getting minimum eigenvalues")

    r=np.linspace(0.,2.,num=201)
    lr=len(r)
    Z=np.reshape(np.zeros(lr**2),[lr,lr])
    for nx,ny in [(nx,ny) for nx in range(lr) for ny in range(lr)]:
        Z[ny,nx]=sorted(eigvals(-kaibbind(0.0,np.identity(7),1.0,r[nx],r[ny])).real)[1]
    np.savetxt('mineigen.dat',Z)


    x1=1.0
    y1=0.05

    rx=np.linspace(0.,x1,num=201)
    ry=np.linspace(0.,y1,num=201)
    lrx=len(rx)
    lry=len(ry)
    Z=np.reshape(np.zeros(lrx*lry),[lry,lrx])
    for nx,ny in [(nx,ny) for nx in range(lrx) for ny in range(lry)]:
        Z[ny,nx]=sorted(eigvals(-kaibbind(0.0,np.identity(7),1.0,rx[nx],ry[ny])).real)[1]
    np.savetxt('mineigenZoom.dat',Z)


def getPtbData():
    print("getting data by perturbation theory")

    r=np.linspace(0.,2.,num=201)
    lr=len(r)
    C=np.zeros(lr)
    E0=np.zeros(lr)
    for nx in range(lr):
        w,vl,vr=eig(-kaibbind(0.0,np.identity(7),1.0,r[nx],0.0),left=True,right=True)
        j=w.real.argsort()[1]
        #print(w.real[j],vl.real[1,j]*vr.real[1,j]/np.sum(vl.real[:,j]*vr.real[:,j])/w.real[j])
        E0[nx]=w.real[j]
        C[nx]=vl.real[1,j]*vr.real[1,j]/np.sum(vl.real[:,j]*vr.real[:,j])/w.real[j]
    np.savetxt('ptbcoeff.dat',C)
    np.savetxt('mineigenZero.dat',E0)

    x1=1.0
    y1=0.05

    rx=np.linspace(0.,x1,num=201)
    ry=np.linspace(0.,y1,num=201)
    lrx=len(rx)
    lry=len(ry)

    C=np.zeros(lrx)
    E0=np.zeros(lrx)
    for nx in range(lrx):
        w,vl,vr=eig(-kaibbind(0.0,np.identity(7),1.0,rx[nx],0.0),left=True,right=True)
        j=w.real.argsort()[1]
        #print(w.real[j],vl.real[1,j]*vr.real[1,j]/np.sum(vl.real[:,j]*vr.real[:,j])/w.real[j])
        E0[nx]=w.real[j]
        C[nx]=vl.real[1,j]*vr.real[1,j]/np.sum(vl.real[:,j]*vr.real[:,j])/w.real[j]
    np.savetxt('ptbcoeffZoom.dat',C)
    np.savetxt('mineigenZeroZoom.dat',E0)

    ZA=np.reshape(np.zeros(lrx*lry),[lry,lrx])
    for nx,ny in [(nx,ny) for nx in range(lrx) for ny in range(lry)]:
        ZA[ny,nx]=E0[nx]+C[nx]*6.0*ry[ny]
    np.savetxt('appmineigenZoom.dat',ZA)

def getATPaseWT(samples):
    atpaseWT=np.zeros([samples.shape[0],2])
    atpaseWT[:,0]=24.0*samples[:,0]*samples[:,1]/(samples[:,0]+samples[:,1])

    for i in range(len(samples)):
        atpaseWT[i,1]=getATPaseFin(samples[i,0],samples[i,1],samples[i,2])
    return atpaseWT


def printFigFit(ax0,samples,truth,logp):

    np.random.seed(42)
    inds = np.random.randint(len(samples), size=100)
    for ind in inds:
        sample = samples[ind]
        kh, keIWT, keCWT, keIEE, keCEE = sample
        solWT = solve_ivp(kaibbind, [0,16], initCI(kh,keIWT), method='LSODA',
                          args=(kh,keIWT,keCWT))
        solEE = solve_ivp(kaibbind, [0,16], initCI(kh,keIEE), method='LSODA',
                          args=(kh,keIEE,keCEE))
        ax0.plot(solWT.t, solWT.y[0],color='tab:cyan',alpha=0.1)
        ax0.plot(solEE.t, solEE.y[0],color='gold',alpha=0.1)

    nll = lambda *args: -log_prob_mono(*args)
    initial = [2.0,2.0,0.03,0.5,0.01]
    soln = minimize(nll, initial,method='Powell')

    kh, keIWT, keCWT, keIEE, keCEE = soln.x
    solWT = solve_ivp(kaibbind_mono, [0,16], initCI_mono(kh,keIWT), method='LSODA',
                      args=(kh,keIWT,keCWT))
    solEE = solve_ivp(kaibbind_mono, [0,16], initCI_mono(kh,keIEE), method='LSODA',
                      args=(kh,keIEE,keCEE))
    ax0.plot(solWT.t, solWT.y[0],color='tab:blue'  ,lw=1.2,ls='dashed')
    ax0.plot(solEE.t, solEE.y[0],color='tab:orange',lw=1.2,ls='dashed')

    kh, keIWT, keCWT, keIEE, keCEE = truth
    solWT = solve_ivp(kaibbind, [0,16], initCI(kh,keIWT), method='LSODA',
                      args=(kh,keIWT,keCWT))
    solEE = solve_ivp(kaibbind, [0,16], initCI(kh,keIEE), method='LSODA',
                      args=(kh,keIEE,keCEE))
    ax0.plot(solWT.t, solWT.y[0],color='tab:blue'  ,lw=1.2)
    ax0.plot(solEE.t, solEE.y[0],color='tab:orange',lw=1.2)

    ax0.plot(BBIND_EXP.Time, BBIND_EXP.EE,'D',markeredgewidth=1.0,
             markersize=5.0,markerfacecolor='none',label='EE',c='tab:orange')
    ax0.plot(BBIND_EXP.Time, BBIND_EXP.WT,'s',markeredgewidth=1.0,
             markersize=5.0,markerfacecolor='none',label='WT',c='tab:blue')
    ax0.plot([],[],color='k',lw=1.2,label='Hexa.')
    ax0.plot([],[],color='k',lw=1.2,ls='dashed',label='Mono.')
    ax0.legend(loc=4)

    ax0.set_xticks(np.arange(0,17,4))
    ax0.set_xlabel('Time (h)')
    ax0.set_ylabel('Bound KaiB/KaiC')
    ax0.text(0.1, 0.9,'A',transform=ax0.transAxes,fontweight='bold',
             horizontalalignment="center",
             verticalalignment="center" )
    #plt.show()

def printFigDist1D(ax1,ax2,ax3,ax4,samples,truth,logp):
    ax1.set_xticks([-15,-10,-5,0])
    ax1.set_yticks([])
    ax1.set_xlabel('$\log$(Likelihood)')
    ax1.text(0.2, 0.85,'B',transform=ax1.transAxes,fontweight='bold',
             horizontalalignment="center",
             verticalalignment="center" )
    ax2.set_yticks([])
    ax2.set_xlabel('$k_h$ (h$^{-1}$)')
    ax2.set_xticks([0,1,2])
    ax2.text(0.1, 0.85,'C',transform=ax2.transAxes,fontweight='bold',
             horizontalalignment="center",
             verticalalignment="center" )
    ax3.set_yticks([])
    ax3.set_xlabel('$k_e$ (h$^{-1}$)')
    ax3.set_xticks([0,0.5,1,1.5])
    ax3.text(0.1, 0.85,'D',transform=ax3.transAxes,fontweight='bold',
             horizontalalignment="center",
             verticalalignment="center" )
    ax4.set_yticks([])
    ax4.set_xlabel('$k_e^*$ (h$^{-1}$)')
    ax4.set_xticks([0,0.03,0.06])
    ax4.text(0.2, 0.85,'E',transform=ax4.transAxes,fontweight='bold',
             horizontalalignment="center",
             verticalalignment="center" )
    logpbind=np.zeros_like(logp)
    for i in range(len(logp)):
        logpbind[i]=logp[i]-log_likeATPase(samples[i])
    ax1.hist(logpbind,bins=50,range=(-17.0,0.0),histtype='step',color='k')
    ax1.axvline(-16.12,color='k',linestyle='dashed',lw=1.0)
    ax2.hist(samples[:,0],bins=50,range=(0.0,2.0),histtype='step',color='k')
    ax2.axvline(truth[0],color='k',lw=1.0)
    ax3.hist(samples[:,1],bins=50,range=(0.0,1.5),histtype='step')
    ax3.hist(samples[:,3],bins=50,range=(0.0,1.5),histtype='step')
    ax3.axvline(truth[1],color='tab:blue',lw=1.0)
    ax3.axvline(truth[3],color='tab:orange',lw=1.0)
    ax4.hist(samples[:,2],bins=50,range=(0.0,0.07),histtype='step')
    ax4.hist(samples[:,4],bins=50,range=(0.0,0.07),histtype='step')
    ax4.axvline(truth[2],color='tab:blue',lw=1.0)
    ax4.axvline(truth[4],color='tab:orange',lw=1.0)

def getFigModel6(figname):
    print("creating "+figname)

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.size'] = 8
    matplotlib.rcParams['font.family'] = 'Arial'
    fig = plt.figure(tight_layout=True,figsize=(5.2,2.6))
    gs = fig.add_gridspec(2,3,width_ratios=[2.5,1,1])
    ax0 = fig.add_subplot(gs[:,0])
    ax1 = fig.add_subplot(gs[0,1],box_aspect=1)
    ax2 = fig.add_subplot(gs[0,2],box_aspect=1)
    ax3 = fig.add_subplot(gs[1,1],box_aspect=1)
    ax4 = fig.add_subplot(gs[1,2],box_aspect=1)
    #samples, truth, logp = getSamples()
    printFigFit(ax0,samples,truth,logp)
    printFigDist1D(ax1,ax2,ax3,ax4,samples,truth,logp)
    fig.savefig(figname,dpi=300,bbox_inches='tight')
    #plt.show()




def getFigPtb(figname):
    print("creating "+figname)
    from mpl_toolkits.axes_grid1 import ImageGrid
    import matplotlib.patches as patches

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.size'] = 8
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams["contour.linewidth"] = 0.8

    fig = plt.figure(tight_layout=True,figsize=(5.2,4.2))
    grid = ImageGrid(fig, 221,
                 nrows_ncols=(1, 1),
                 cbar_mode="single",
                 cbar_pad="5%",
                 )
    Z=np.loadtxt('mineigen.dat')
    r=np.linspace(0.,2.,num=201)
    contours = grid[0].contour(r, r, Z,np.arange(0.,3.,0.25), colors='black')
    grid[0].clabel(contours,manual=[(1.5,0.1),(1.0,0.2),(0.41,0.5),(1.0,1.86),(1.5,1.86)],inline=True,fmt='%1.1f')
    im0=grid[0].imshow(Z,extent=[0,2,0,2],origin='lower',cmap='coolwarm')
    cax = grid.cbar_axes[0]

    r = patches.Rectangle(xy=(0, 0), width=1., height=0.05, ec='k', fc='y')
    grid[0].add_patch(r)

    cb = fig.colorbar(im0, cax=cax,label='Binding rate/$k_{h}$')
    cb.set_ticks(np.arange(0,3.1,0.5))
    grid[0].set_xticks(np.arange(0,2.1,0.5))
    grid[0].set_xlabel('$k_{e}/k_{h}$')
    grid[0].set_yticks(np.arange(0,2.1,0.5))
    grid[0].set_ylabel('$k_{e}^{*}/k_{h}$')
    grid[0].text(-0.25, 1.0,'A',transform=grid[0].transAxes,fontweight='bold',
             horizontalalignment="center",
             verticalalignment="center" )

    grid = ImageGrid(fig, 222,
                 nrows_ncols=(1, 1),
                 cbar_mode="single",
                 cbar_pad="5%",
                 )
    x1=1.0
    y1=0.05
    Z=np.loadtxt('mineigenZoom.dat')
    rx=np.linspace(0.,x1,num=201)
    ry=np.linspace(0.,y1,num=201)
    contours = grid[0].contour(rx, ry, Z,np.arange(0.0,1.2,0.1), colors='black')
    grid[0].clabel(contours,manual=[(0.9,0.02),(0.4,0.03),(0.2,0.03),(0.1,0.03)],inline=True,fmt='%1.1f')
    im1=grid[0].imshow(Z,extent=[0,x1,0,y1],origin='lower',cmap='coolwarm',aspect=20.0)

    #samples, truth, logp = getSamples()
    grid[0].plot(truth[3]/truth[0], truth[4]/truth[0],'D',
                 mew=1., mec='k',ms=4.5,label='EE',c='tab:orange')
    grid[0].plot(truth[1]/truth[0], truth[2]/truth[0],'s',
                 mew=1., mec='k',ms=4.5,label='WT',c='b')
    grid[0].legend(loc=1)

    cax = grid.cbar_axes[0]
    cb = fig.colorbar(im1, cax=cax,label='Binding rate/$k_{h}$')
    cb.set_ticks(np.arange(0,1.3,0.2))
    grid[0].set_xticks(np.arange(0,1.1,0.2))
    grid[0].set_xlabel('$k_{e}/k_{h}$')
    grid[0].set_yticks(np.arange(0,0.055,0.01))
    grid[0].set_ylabel('$k_{e}^{*}/k_{h}$')
    grid[0].text(-0.25, 1.0,'B',transform=grid[0].transAxes,fontweight='bold',
             horizontalalignment="center",
             verticalalignment="center" )

    grid = ImageGrid(fig, 223,
                 nrows_ncols=(1, 1),
                 cbar_mode="single",
                 cbar_pad="5%",
                 )
    dummy=np.reshape(np.zeros(2),[2,1])
    im1=grid[0].imshow(dummy,extent=[0,2.,0,1.],origin='lower',alpha=0.,aspect=2.)
    cax = grid.cbar_axes[0]
    cb = fig.colorbar(im1, cax=cax)
    cb.remove()
    rx=np.linspace(0.,2.,num=201)
    C =np.loadtxt('ptbcoeff.dat')
    E0=np.loadtxt('mineigenZero.dat')
    grid[0].plot(rx,E0,c="k",label="Binding rate",lw=1.2)
    grid[0].plot(rx,C, c="k",label="Coefficient",ls="dashed",lw=1.2)
    grid[0].tick_params(right=True,labelright=True)
    grid[0].text(1.2, 0.5,'Coefficient',transform=grid[0].transAxes,rotation=90,
             horizontalalignment="center",
             verticalalignment="center" )
    grid[0].set_xticks(np.arange(0,2.1,0.5))
    grid[0].set_xlabel('$k_{e}/k_{h}$')
    grid[0].set_yticks(np.arange(0,1.1,0.2))
    grid[0].set_ylabel('Binding rate/$k_{h}$')
    grid[0].text(-0.25, 1.0,'C',transform=grid[0].transAxes,fontweight='bold',
             horizontalalignment="center",
             verticalalignment="center" )
    grid[0].legend(loc=5)

    grid = ImageGrid(fig, 224,
                 nrows_ncols=(1, 1),
                 cbar_mode="single",
                 cbar_pad="5%",
                 )
    x1=1.0
    y1=0.05
    Z=np.loadtxt('mineigenZoom.dat')
    ZA=np.loadtxt('appmineigenZoom.dat')
    rx=np.linspace(0.,x1,num=201)
    ry=np.linspace(0.,y1,num=201)
    contours = grid[0].contour(rx, ry, 100.*np.abs(Z-ZA)/Z,
                               [0.2,0.4,0.6,0.8,1.0,1.2,1.4],colors='black')
    grid[0].clabel(contours,manual=[(0.4,0.022),(0.4,0.035),(0.4,0.042)],inline=True,fmt='%1.1f')
    im1=grid[0].imshow(100.*np.abs(Z-ZA)/Z,extent=[0,x1,0,y1],
                       origin='lower',cmap='Greys',aspect=20.0)

    grid[0].plot(truth[3]/truth[0], truth[4]/truth[0],'D',
                 mew=1., mec='k',ms=4.5,label='EE',c='tab:orange')
    grid[0].plot(truth[1]/truth[0], truth[2]/truth[0],'s',
                 mew=1., mec='k',ms=4.5,label='WT',c='b')
    grid[0].legend(loc=1)

    cax = grid.cbar_axes[0]
    cb = fig.colorbar(im1, cax=cax,label='% error')
    cb.set_ticks(np.arange(0,2.1,0.5))
    grid[0].set_xticks(np.arange(0,1.1,0.2))
    grid[0].set_xlabel('$k_{e}/k_{h}$')
    grid[0].set_yticks(np.arange(0,0.055,0.01))
    grid[0].set_ylabel('$k_{e}^{*}/k_{h}$')
    grid[0].text(-0.25, 1.0,'D',transform=grid[0].transAxes,fontweight='bold',
             horizontalalignment="center",
             verticalalignment="center" )

    fig.savefig(figname,dpi=300,bbox_inches='tight')
    #plt.show()

def getFigATPase(figname):
    print("creating "+figname)

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.size'] = 8
    matplotlib.rcParams['font.family'] = 'Arial'
    fig = plt.figure(tight_layout=True,figsize=(5.2,2.6))
    gs = fig.add_gridspec(2,2,width_ratios=[1.5,1],hspace=0)
    ax0 = fig.add_subplot(gs[:,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[1,1])

    #samples, truth, logp = getSamples()
    atpaseWT = getATPaseWT(samples)
    upperLim=[14.5,8.9]
    truthATPase=[24.0*truth[0]*truth[1]/(truth[0]+truth[1]),
            getATPaseFin(*truth[0:3])]

    ax1.set_xticks([0,5,10,15,20])
    ax1.set_yticks([])
    ax1.set_ylim(0.,6800)
    ax1.tick_params(labelbottom=False)
    ax1.text(0.05, 0.85,'B',transform=ax1.transAxes,fontweight='bold',
             horizontalalignment="left",
             verticalalignment="center" )
    ax1.text(0.05, 0.7,'w/o KaiB',transform=ax1.transAxes,
             horizontalalignment="left",
             verticalalignment="center" )
    ax2.set_xticks([0,5,10,15,20])
    ax2.set_yticks([])
    ax2.set_ylim(0.,6800)
    ax2.set_xlabel('ATPase activity (/day)')
    ax2.text(0.05, 0.85,'C',transform=ax2.transAxes,fontweight='bold',
             horizontalalignment="left",
             verticalalignment="center" )
    ax2.text(0.05, 0.7,'w KaiB',transform=ax2.transAxes,
             horizontalalignment="left",
             verticalalignment="center" )
    ax1.hist(atpaseWT[:,0],bins=50,range=(0.0,20.0),histtype='step',color='tab:blue')
    ax1.axvline(upperLim[0],color='k',linestyle='dashed',lw=1.0)
    ax1.axvline(truthATPase[0],color='tab:blue',lw=1.0)
    ax2.hist(atpaseWT[:,1],bins=50,range=(0.0,20.0),histtype='step',color='tab:blue')
    ax2.axvline(upperLim[1],color='k',linestyle='dashed',lw=1.0)
    ax2.axvline(truthATPase[1],color='tab:blue',lw=1.0)

    np.random.seed(42)
    inds = np.random.randint(len(samples), size=200)
    for ind in inds:
        sample = samples[ind]
        kh, keIWT, keCWT, keIEE, keCEE = sample
        solWT = solve_ivp(kaibbind, [0,16], initCI(kh,keIWT), method='LSODA',
                          args=(kh,keIWT,keCWT))
        ax0.plot(solWT.t,
        4.*kh*(6.*solWT.y[6]+5.*solWT.y[5]+4.*solWT.y[4]+3.*solWT.y[3]+2.*solWT.y[2]+solWT.y[1]),
        color='tab:cyan',alpha=0.1)

    kh, keIWT, keCWT, keIEE, keCEE = truth
    solWT = solve_ivp(kaibbind, [0,16], initCI(kh,keIWT), method='LSODA',
                      args=(kh,keIWT,keCWT))
    ax0.plot(solWT.t,
    4.*kh*(6.*solWT.y[6]+5.*solWT.y[5]+4.*solWT.y[4]+3.*solWT.y[3]+2.*solWT.y[2]+solWT.y[1]),
    color='tab:blue',lw=1.2)


    ax0.set_xticks(np.arange(0,17,4))
    ax0.set_xlabel('Time (h)')
    ax0.set_ylabel('ATPase activity (/day)')
    ax0.set_yticks([0,5,10,15,20])
    ax0.set_ylim(0.,20.)
    ax0.text(0.1, 0.9,'A',transform=ax0.transAxes,fontweight='bold',
             horizontalalignment="left",
             verticalalignment="center" )
    fig.savefig(figname,dpi=300,bbox_inches='tight')
    #plt.show()


if __name__ == '__main__':

    if not os.path.exists(samplefile):
        runMCMC_ATPase()
    samples, truth, logp = getSamples()
    getMinEigen()
    getPtbData()
    getFigModel6('fig1.tif')
    getFigPtb('fig2.tif')

