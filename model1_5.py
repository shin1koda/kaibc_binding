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
import matplotlib.gridspec as gridspec
import os


#Parameters
BBIND_EXP=pd.read_csv('./bbind_exp.csv')
SIGMA2=0.0025*np.ones_like(BBIND_EXP.Time)
SIGMA2ATPase=np.array([4.0,0.81])
uLimATPase=np.array([14.5,8.9])

#Grobal variables
NT=0

#Comments
#I in x[I]: 0<=I<=6, num of bound ATP
#NT: threshold num of bound "ATP"
#    if NT<I<=6 then x[I] is binding-Incompetent
#    if 0<=I<=NT then x[I] is binding-Competent
#thus, NT corresponds to Model "6-NT"

def getSampleFilename():
    if NT==0:
        return 'samplesLimByATPase.h5'
    return 'samples_wATPase_N'+str(6-NT)+'.h5'

def kaibbind(t,x,khI,khC,keI,keC):

    jFrwd = np.array([float(I)  *khI*x[I] if I>NT else
                      float(I)  *khC*x[I] for I in range(7)])
    jBkwd = np.array([float(6-I)*keI*x[I] if I>NT else
                      float(6-I)*keC*x[I] for I in range(7)])

    jSys = -jFrwd-jBkwd
    jSys[0:6] += jFrwd[1:7]
    jSys[1:7] += jBkwd[0:6]

    return jSys

def initCI(khI,keI):
    return binom.pmf(range(7),6,keI/(keI+khI))

#def kaibbind_mono(t,x,khI,khC,keI,keC):
#    return [-keC*x[0] +khI*x[1],
#            +keC*x[0] -khI*x[1]]
#
#def initCI_mono(khI,keI):
#    return binom.pmf(range(2),1,keI/(keI+khI))

# M: the num of ATPase-disabled sites
def getFinDist(khI,khC,keI,keC,M):

    kFrwd = np.array([float(max(I-M,0))*khI if I>NT else
                      float(max(I-M,0))*khC for I in range(7)])
    kBkwd = np.array([float(6-I)*keI if I>NT else
                      float(6-I)*keC for I in range(7)])
    x=np.ones(7)
    for I in range(6):
        x[0:I+1] *= kFrwd[I+1]
        x[I+1:7] *= kBkwd[I]
    x=x/np.sum(x)

    return x

def getPopCmp(x):
    return np.sum(x[0:NT+1])

def getFinPopsCmpWDisabledSites(khI,khC,keI,keC):

    return np.array([getPopCmp(getFinDist(khI,khC,keI,keC,M))
                    for M in range(7)])


def getATPaseFin(khI,khC,keI,keC):

    kFrwd = np.array([float(I)  *khI if I>NT else
                      float(I)  *khC for I in range(7)])
    kBkwd = np.array([float(6-I)*keI if I>NT else
                      float(6-I)*keC for I in range(7)])
    x=np.ones(7)
    for I in range(6):
        x[0:I+1] *= kFrwd[I+1]
        x[I+1:7] *= kBkwd[I]
    x=x/np.sum(x)

    return 4.*np.sum(kFrwd*x)

def log_likeATPase(params):
    khI, khC, keIWT, keCWT, keIEE, keCEE = params
    ATPaseWT=np.array([24.0*khI*keIWT/(khI+keIWT),
                       getATPaseFin(khI,khC,keIWT,keCWT)])
    diff=np.zeros(2)
    for i in range(2):
        diff[i]=max(ATPaseWT[i]-uLimATPase[i],0.0)

    d=(ATPaseWT[0]-ATPaseWT[1]-uLimATPase[0]+uLimATPase[1])**2
    return -0.5*np.sum(diff**2/SIGMA2ATPase) -0.5*d**2/np.sum(SIGMA2ATPase)

def log_like(params):
    khI, khC, keIWT, keCWT, keIEE, keCEE = params
    solWT = solve_ivp(kaibbind, [0,16], initCI(khI,keIWT), method='LSODA',
                      t_eval=BBIND_EXP.Time, args=(khI,khC,keIWT,keCWT))
    solEE = solve_ivp(kaibbind, [0,16], initCI(khI,keIEE), method='LSODA',
                      t_eval=BBIND_EXP.Time, args=(khI,khC,keIEE,keCEE))
    BonCWT=solWT.y[0]
    BonCEE=solEE.y[0]
    if NT>0:
        for I in range(1,NT+1):
            BonCWT += solWT.y[I]
            BonCEE += solEE.y[I]
    return -0.5*np.sum((BonCWT-BBIND_EXP.WT)**2/SIGMA2)\
           -0.5*np.sum((BonCEE-BBIND_EXP.EE)**2/SIGMA2)


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


def runMCMC_ATPase():
    print('running MCMC')

    nll = lambda *args: -log_prob(*args)
    initial = [2.0,2.0,2.0,0.03,0.5,0.01]
    sol = minimize(nll, initial,method='Powell')

    np.random.seed(100)
    pos = sol.x * np.exp(0.0001 * np.random.randn(32, 6))
    nwalkers, ndim = pos.shape

    filename = getSampleFilename()
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probATPase,
                                        pool=pool,backend=backend)
        sampler.run_mcmc(pos,50000, progress=True)


def getSamples():
    print('getting Samples')
    filename = getSampleFilename()
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


def getATPaseWT(samples):
    atpaseWT=np.zeros([samples.shape[0],2])
    atpaseWT[:,0]=24.0*samples[:,0]*samples[:,2]/(samples[:,0]+samples[:,2])

    for i in range(len(samples)):
        atpaseWT[i,1]=getATPaseFin(samples[i,0],samples[i,1],samples[i,2],samples[i,3])

    return atpaseWT

def getDiffInitATPase(samples):
    diffATPase=np.zeros([samples.shape[0],2])
    for i in range(len(samples)):
        khI, khC, keIWT, keCWT, keIEE, keCEE = samples[i]
        cf=4.*np.arange(7, dtype=float)
        cf[0:NT+1]*=khC-khI
        cf[NT+1:]*=0.
        diffATPase[i,0]=np.sum(cf*initCI(khI,keIWT))
        diffATPase[i,1]=np.sum(cf*initCI(khI,keIEE))

    return diffATPase

def getFigATPase(figname):
    global NT
    print("creating "+figname)

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.size'] = 8
    matplotlib.rcParams['font.family'] = 'Arial'
    fig = plt.figure(tight_layout=True,figsize=(5.2,4.0))
    gs = fig.add_gridspec(2,1,height_ratios=[2,1])
    subgs0 = gs[0].subgridspec(2,2,width_ratios=[1.5,1],hspace=0)
    subgs1 = gs[1].subgridspec(1,4,wspace=0)
    ax0 = fig.add_subplot(subgs0[:,0])
    ax1 = fig.add_subplot(subgs0[0,1])
    ax2 = fig.add_subplot(subgs0[1,1])
    axs01 = subgs1.subplots(sharex=True)

    NT=0
    samples, truth, logp = getSamples()
    samples2=np.zeros([samples.shape[0],samples.shape[1]+1])
    samples2[:,0]=samples[:,0]
    samples2[:,1:]=samples[:,0:]
    atpaseWT = getATPaseWT(samples2)
    upperLim=[14.5,8.9]
    truthATPase=[24.0*truth[0]*truth[1]/(truth[0]+truth[1]),
            getATPaseFin(truth[0],*truth[0:3])]

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
                          args=(kh,kh,keIWT,keCWT))
        ax0.plot(solWT.t,
        4.*kh*(6.*solWT.y[6]+5.*solWT.y[5]+4.*solWT.y[4]+3.*solWT.y[3]+2.*solWT.y[2]+solWT.y[1]),
        color='tab:cyan',alpha=0.1)

    kh, keIWT, keCWT, keIEE, keCEE = truth
    solWT = solve_ivp(kaibbind, [0,16], initCI(kh,keIWT), method='LSODA',
                      args=(kh,kh,keIWT,keCWT))
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
    ax0.text(0.15, 0.9,'Model6',transform=ax0.transAxes,
             horizontalalignment="left",
             verticalalignment="center" )
    abc='DEFG'
    for I in range(4):
        NT=4-I
        samples, truth, logp = getSamples()
        axs01[I].set_xticks(np.arange(0,17,4))
        axs01[I].set_xlabel('Time (h)')
        if I==0:
            axs01[I].set_ylabel('ATPase activity (/day)')
            axs01[I].set_yticks([0,5,10,15,20])
        else:
            axs01[I].set_yticks([])
        axs01[I].set_ylim(0.,20.)
        axs01[I].text(0.1, 0.9,abc[I],
             transform=axs01[I].transAxes,fontweight='bold',
             horizontalalignment="left",
             verticalalignment="center" )
        axs01[I].text(0.2, 0.9,'Model '+str(I+2),
             transform=axs01[I].transAxes,
             horizontalalignment="left",
             verticalalignment="center" )
        np.random.seed(42)
        inds = np.random.randint(len(samples), size=200)
        for ind in inds:
            sample = samples[ind]
            khI,khC, keIWT, keCWT, keIEE, keCEE = sample
            solWT = solve_ivp(kaibbind, [0,16], initCI(khI,keIWT), method='LSODA',
                              args=(khI,khC,keIWT,keCWT))
            axs01[I].plot(solWT.t,
            4.*khI*(6.*solWT.y[6]+5.*solWT.y[5]+4.*solWT.y[4]+3.*solWT.y[3]+2.*solWT.y[2]+solWT.y[1]),
            color='tab:cyan',alpha=0.1)

    fig.savefig(figname,dpi=300,bbox_inches='tight')
    #plt.show()

def getFigExp(figname):
    global NT

    print("creating "+figname)

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.size'] = 8
    matplotlib.rcParams['font.family'] = 'Arial'
    fig = plt.figure(tight_layout=True,figsize=(3.9,2.6))
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal', adjustable='box')

    #ax.set_xlim(-0.1,1.1)
    #ax.set_ylim(-0.1,1.1)
    ax.set_xticks(np.arange(0,1.1,0.2))
    ax.set_yticks(np.arange(0,1.1,0.2))
    ax.set_xlabel("Proportion of C1cat$^{-}$-EE")
    ax.set_ylabel('Bound KaiB/KaiC')

    c=iter(matplotlib.colors.TABLEAU_COLORS.items())
    x=np.linspace(0.,1.,100)
    bnms=np.array([binom.pmf(I,6,x) for I in range(7)])
    for JNT in reversed(range(5)):
        cname=next(c)[0]
        NT=JNT
        samples, truth, logp = getSamples()
        np.random.seed(42)
        inds = np.random.randint(len(samples), size=100)
        ax.plot([],[],color=cname,label=str(6-NT))
        for ind in inds:
            sample = samples[ind]
            if NT==0:
                khI, keIWT, keCWT, keIEE, keCEE = sample
                khC=0.0
            else:
                khI, khC, keIWT, keCWT, keIEE, keCEE = sample
            y=np.dot(np.array([getFinPopsCmpWDisabledSites(khI,khC,keIEE,keCEE)]),
                     bnms)
            ax.plot(x,y[0],color=cname,alpha=0.1)

    ax.legend(loc='upper right',title='Model')
    fig.savefig(figname,dpi=300,bbox_inches='tight')
    #plt.show()

def getFigModel1to5(figname):
    global NT

    print("creating "+figname)

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.size'] = 8
    matplotlib.rcParams['font.family'] = 'Arial'
    fig = plt.figure(tight_layout=True,figsize=(5.2,4.5))
    gs0 = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1,2])
    gs00 = gs0[0].subgridspec(5, 1, hspace=0)
    gs01 = gs0[1].subgridspec(3, 2, hspace=0.5)
    axs00 = gs00.subplots(sharex=True, sharey=True)
    axs01 = gs01.subplots()

    axs00[4].set_xticks(np.arange(0,17,4))
    axs00[4].set_xlabel('Time (h)')
    axs00[2].set_ylabel('Bound KaiB/KaiC')
    abc=['A','B','C','D','E']
    upprcs = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for I in range(5):
        axs00[I].text(0.1, 0.85,upprcs[I],transform=axs00[I].transAxes,fontweight='bold',
                     horizontalalignment="center",
                     verticalalignment="center" )
        axs00[I].text(0.8, 0.1,'Model '+str(I+1),transform=axs00[I].transAxes,
                     horizontalalignment="center",
                     verticalalignment="center" )

    for I in range(3):
      for J in range(2):
        axs01[I,J].set_yticks([])
        axs01[I,J].text(0.92, 0.85,upprcs[2*I+J+5],
                        transform=axs01[I,J].transAxes,fontweight='bold',
                        horizontalalignment="center", verticalalignment="center" )

    axs01[0,0].set_xlabel("$k_h$ (h$^{-1}$)")
    axs01[0,0].set_xlim(-0.1,2.1)
    axs01[0,1].set_xlabel("$k_h^{*}$ (h$^{-1}$)")
    axs01[0,1].set_xlim(-0.25,5.25)
    axs01[1,0].set_xlabel("$k_{eWT}$ (h$^{-1}$)")
    axs01[1,0].set_xlim(-0.5,10.5)
    axs01[1,1].set_xlabel("$k_{eWT}^{*}$ (h$^{-1}$)")
    axs01[1,1].set_xlim(-0.025,0.525)
    axs01[1,1].set_ylim(None,8)
    axs01[2,0].set_xlabel("$k_{eEE}$ (h$^{-1}$)")
    axs01[2,0].set_xlim(-0.25,5.25)
    axs01[2,1].set_xlabel("$k_{eEE}^{*}$ (h$^{-1}$)")
    axs01[2,1].set_xlim(-0.025,0.525)
    axs01[2,1].set_ylim(None,10)


    x0,y0,wi,he = axs01[0,1].get_position().bounds
    x0=x0+0.3*wi-0.02
    wi=0.5*wi
    y0=y0+0.35*he+0.09
    he=0.6*he
    ax0101s=plt.axes((x0,y0,wi,he))
    ax0101s.set_yticks([])
    ax0101s.set_xlim(-2.5,52.5)

    x0,y0,wi,he = axs01[1,0].get_position().bounds
    x0=x0+0.3*wi-0.02
    wi=0.5*wi
    y0=y0+0.35*he+0.055
    he=0.6*he
    ax0110s=plt.axes((x0,y0,wi,he))
    ax0110s.set_yticks([])
    ax0110s.set_xlim(-4,84)

    NT=5
    cf=np.zeros(7)
    cf[0:NT+1]=1.
    #nll = lambda *args: -log_probATPase(*args)
    nll = lambda *args: -log_prob(*args)
    initial = [2.0,2.0,2.0,0.03,0.5,0.01]
    sol = minimize(nll, initial,method='Powell')
    khI, khC, keIWT, keCWT, keIEE, keCEE = sol.x
    solWT = solve_ivp(kaibbind, [0,16], initCI(khI,keIWT), method='LSODA',
                      args=(khI,khC,keIWT,keCWT))
    solEE = solve_ivp(kaibbind, [0,16], initCI(khI,keIEE), method='LSODA',
                      args=(khI,khC,keIEE,keCEE))
    valWT=np.zeros(len(solWT.y[0]))
    valEE=np.zeros(len(solEE.y[0]))
    for J in range(7):
        valWT+=solWT.y[J]*cf[J]
        valEE+=solEE.y[J]*cf[J]
    axs00[0].plot(solWT.t, valWT,color='tab:cyan')
    axs00[0].plot(solEE.t, valEE,color='gold')
    axs00[0].plot(BBIND_EXP.Time, BBIND_EXP.EE,'D',markeredgewidth=1.0,
             markersize=3.0,markerfacecolor='none',label='EE',c='tab:orange')
    axs00[0].plot(BBIND_EXP.Time, BBIND_EXP.WT,'s',markeredgewidth=1.0,
             markersize=3.0,markerfacecolor='none',label='WT',c='tab:blue')
    axs00[0].set_ylim(-0.15,1.15)

    c=iter(matplotlib.colors.TABLEAU_COLORS.items())
    for I in range(1,6):
        cname=next(c)[0]
        NT=5-I
        print('NT='+str(NT))
        samples, truth, logp = getSamples()

        if NT>0:
            cf=np.zeros(7)
            cf[0:NT+1]=1.

            np.random.seed(42)
            inds = np.random.randint(len(samples), size=50)
            for ind in inds:
                sample = samples[ind]
                khI, khC, keIWT, keCWT, keIEE, keCEE = sample
                solWT = solve_ivp(kaibbind, [0,16], initCI(khI,keIWT), method='LSODA',
                                  args=(khI,khC,keIWT,keCWT))
                solEE = solve_ivp(kaibbind, [0,16], initCI(khI,keIEE), method='LSODA',
                                  args=(khI,khC,keIEE,keCEE))
                valWT=np.zeros(len(solWT.y[0]))
                valEE=np.zeros(len(solEE.y[0]))
                for J in range(7):
                    valWT+=solWT.y[J]*cf[J]
                    valEE+=solEE.y[J]*cf[J]
                axs00[I].plot(solWT.t, valWT,color='tab:cyan',alpha=0.1)
                axs00[I].plot(solEE.t, valEE,color='gold',alpha=0.1)

            axs00[I].plot(BBIND_EXP.Time, BBIND_EXP.EE,'D',markeredgewidth=1.0,
                     markersize=3.0,markerfacecolor='none',label='EE',c='tab:orange')
            axs00[I].plot(BBIND_EXP.Time, BBIND_EXP.WT,'s',markeredgewidth=1.0,
                     markersize=3.0,markerfacecolor='none',label='WT',c='tab:blue')
            axs00[I].set_ylim(-0.15,1.15)


            axs01[0,0].hist(samples[:,0],bins=20,histtype='step',density=True,color=cname)
            axs01[0,1].hist(samples[:,1],bins=20,histtype='step',density=True,color=cname)
            axs01[1,0].hist(samples[:,2],bins=20,histtype='step',density=True,color=cname)
            axs01[1,1].hist(samples[:,3],bins=20,histtype='step',density=True,color=cname)
            axs01[2,0].hist(samples[:,4],bins=20,histtype='step',density=True,color=cname)
            axs01[2,1].hist(samples[:,5],bins=20,histtype='step',density=True,color=cname)

            if NT==1:
                ax0101s.hist(samples[:,1],bins=20,histtype='step',density=True,color=cname)
            if NT==4:
                ax0110s.hist(samples[:,2],bins=20,histtype='step',density=True,color=cname)
        else:
            axs01[0,0].hist(samples[:,0],bins=20,histtype='step',density=True,color=cname)
            axs01[1,0].hist(samples[:,1],bins=20,histtype='step',density=True,color=cname)
            axs01[1,1].hist(samples[:,2],bins=20,histtype='step',density=True,color=cname)
            axs01[2,0].hist(samples[:,3],bins=20,histtype='step',density=True,color=cname)
            axs01[2,1].hist(samples[:,4],bins=20,histtype='step',density=True,color=cname)

        axs01[0][1].plot([],[],label=str(I+1),color=cname)


    axs01[0][1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,title='Model')

    fig.savefig(figname,dpi=300)


if __name__ == '__main__':
    for JNT in range(1,5):
        NT=JNT
        if not os.path.exists(getSampleFilename()):
            runMCMC_ATPase()

    getFigModel1to5('fig3.tif')
    getFigExp('fig6.tif')
    getFigATPase('fig7.tif')

