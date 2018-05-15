# Written 11 Aug 2017 by RFT to analyze correlations b/w Lya
# and other LBG properties
# v1 - based on make_lbg_lya_plots_v3 but updated for Python 3


import numpy as np
import pylab as pl
from fill_steps import fill_steps
from plot_steps import plot_steps
import lmfit
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LogLocator
#from readcol import readcol
from mad import mad
from scipy.io.idl import readsav
import matplotlib as mpl
import astropy.cosmology as cosm
import scipy.stats as stats
import k_lambda_py3 as kl
import warnings
from stack_gauss_densities import stack_gauss_densities
from colormaps import viridis
from astropy.io import ascii

import seaborn as sns
sns.set(style='white', font='serif',font_scale=2.0,palette='deep')

sns_deep=sns.palettes.SEABORN_PALETTES['deep']

fout=open('lya-correlations.dat','w')
fout.write('Variables & $N_\mathrm{gal}$ & $r_\mathrm{Pearson}$ & $\log_{10}(p)$\\\\ \n')

mpl.rcParams['mathtext.fontset']='cm'
#mpl.rcParams['font.family']='serif'
#mpl.rcParams['font.serif']=['Times']

def expmodel(x, p):
    '''Exponential with vertical shift for EqW distribution.'''

    W0=p['W0'].value
    W1=p['W1'].value
    x0=p['x0'].value

    return W0+W1*np.exp(x/x0)
    
def expresid(p, x, data, err=None):
    """Objective function for minimization."""
    if err==None:
        return expmodel(x,p)-data
    else:
        return (expmodel(x,p)-data)/err
    
def expresiderr(p, x, data, err):
    """Objective function for minimization."""
    return (expmodel(x,p)-data)/err

def u2s(arr,encoding='UTF-8'):
    """Converts array in Unicode byte type to string type."""
    if type(arr) == bytes:
        return arr.decode(encoding='UTF-8')
    elif type(arr) == np.ndarray:
        return np.asarray([x.decode(encoding='UTF-8') for x in arr])
    elif type(arr) == list:
        return [x.decode(encoding='UTF-8') for x in arr]
    else: return arr

## Read in the IDL sav files for KBSS catalogs 

kbss_obj=np.array([]);kbss_field=np.array([])
kbss_zem=np.array([]);kbss_zabs=np.array([]);kbss_zneb=np.array([])
kbss_fha=np.array([]);kbss_eha=np.array([]);
kbss_fhb=np.array([]);kbss_ehb=np.array([]);
kbss_fo2=np.array([]);kbss_eo2=np.array([]);
kbss_ro2=np.array([]);kbss_ero2=np.array([]);
kbss_fo3_5008=np.array([]);kbss_eo3_5008=np.array([]);
kbss_fo3_4960=np.array([]);kbss_eo3_4960=np.array([]);
kbss_fn2_6585=np.array([]);kbss_en2_6585=np.array([]);
kbss_mositime=np.array([])
kbss_wj=np.array([]);kbss_wje=np.array([]);
kbss_wh=np.array([]);kbss_whe=np.array([]);
kbss_wk=np.array([]);kbss_wke=np.array([]);
kbss_scor=np.array([[],[],[]]).transpose()
kbss_scor_err=np.array([[],[],[]]).transpose()
kbss_zj=np.array([]);kbss_zh=np.array([]);kbss_zk=np.array([]);
kbss_lmstar=np.array([]);kbss_rmag=np.array([])
kbss_gr=np.array([])

#savfiles=readcol('kbss_sav.list').flatten()
#savfiles=readcol('kbss_sav_cor.list').flatten()
savfiles=ascii.read('kbss_sav_cor.list').columns.values()[0]
for file in savfiles:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = readsav(file)
    data = data['kbss_cor'] #data['kbss'] 
    field = u2s(data['field'][0])
    kbss_obj=np.append(kbss_obj,[field+'-'+x for x in u2s(data['obj'])])
    kbss_field=np.append(kbss_field,[field]*len(data['obj']))
    kbss_zem=np.append(kbss_zem,[x for x in data['zem']])
    kbss_zabs=np.append(kbss_zabs,[x for x in data['zabs']])
    kbss_zneb=np.append(kbss_zneb,[x for x in data['zneb']])
    kbss_fha=np.append(kbss_fha,[x for x in data['fha']])
    kbss_eha=np.append(kbss_eha,[x for x in data['eha']])
    kbss_fhb=np.append(kbss_fhb,[x for x in data['fhb']])
    kbss_ehb=np.append(kbss_ehb,[x for x in data['ehb']])
    kbss_fo2=np.append(kbss_fo2,[x.sum() for x in data['fo2']])
    kbss_eo2=np.append(kbss_eo2,[np.sqrt(x[0]**2+x[1]**2) for x in data['eo2']])
    kbss_ro2=np.append(kbss_ro2,[x[1]/x[0] for x in data['fo2']])
    kbss_fo3_5008=np.append(kbss_fo3_5008,[x[1] for x in data['fo3n']])
    kbss_eo3_5008=np.append(kbss_eo3_5008,[x[1] for x in data['eo3n']])
    kbss_fo3_4960=np.append(kbss_fo3_4960,[x[0] for x in data['fo3n']])
    kbss_eo3_4960=np.append(kbss_eo3_4960,[x[0] for x in data['eo3n']])
    kbss_fn2_6585=np.append(kbss_fn2_6585,[x[1] for x in data['fn2']])
    kbss_en2_6585=np.append(kbss_en2_6585,[x[1] for x in data['en2']])
    kbss_mositime=np.append(kbss_mositime,[x for x in data['mositime']])
    kbss_wj=np.append(kbss_wj,[x for x in data['wj']])
    kbss_wje=np.append(kbss_wj,[x for x in data['wje']])
    kbss_wh=np.append(kbss_wh,[x for x in data['wh']])
    kbss_whe=np.append(kbss_wh,[x for x in data['whe']])
    kbss_wk=np.append(kbss_wk,[x for x in data['wk']])
    kbss_wke=np.append(kbss_wk,[x for x in data['wke']])
    kbss_scor=np.append(kbss_scor,[x for x in data['scor']],axis=0)
    kbss_scor_err=np.append(kbss_scor_err,[x for x in data['scor_err']],axis=0)
    kbss_zj=np.append(kbss_zj,[x for x in data['zj']])
    kbss_zh=np.append(kbss_zh,[x for x in data['zh']])
    kbss_zk=np.append(kbss_zk,[x for x in data['zk']])
    kbss_lmstar=np.append(kbss_lmstar,[x for x in data['lmstar']])
    kbss_rmag=np.append(kbss_rmag,[x for x in data['rmag']])
    kbss_gr=np.append(kbss_gr,[x for x in data['grcolor']])

kbss_jscor=kbss_scor[:,0]
kbss_hscor=kbss_scor[:,1]
kbss_kscor=kbss_scor[:,2]
kbss_jscor_err=kbss_scor_err[:,0]
kbss_hscor_err=kbss_scor_err[:,1]
kbss_kscor_err=kbss_scor_err[:,2]

kbss_ehb=np.abs(kbss_ehb)
kbss_eha=np.abs(kbss_eha)

kbss_rmag[kbss_rmag==99]=26.
kbss_gmag=np.minimum(kbss_gr+kbss_rmag,26)

san16_a=0.3771
san16_b=2468
san16_c=638.4
san16_rmin=0.3839
san16_rmax=1.4558
kbss_ro2_clip=np.minimum(np.maximum(kbss_ro2,san16_rmin),san16_rmax)
kbss_o2ne=(san16_c*kbss_ro2_clip-san16_a*san16_b)/(san16_a-kbss_ro2_clip)

kbss_zhk_idx=np.abs(kbss_zh-kbss_zk)<0.01
kbss_w_idx=(kbss_wk/kbss_wh>0.5)&(kbss_wk/kbss_wh<2.)
kbss_snr_idx=(kbss_fha>5*kbss_eha)&(kbss_fhb>3*kbss_ehb)&(kbss_fo3_5008>3*kbss_eo3_5008)&(kbss_fn2_6585>=2*kbss_en2_6585)&kbss_zhk_idx&kbss_w_idx
kbss_lim_idx=(kbss_fha>5*kbss_eha)&(kbss_fhb>3*kbss_ehb)&(kbss_fo3_5008>3*kbss_eo3_5008)&(kbss_fn2_6585<2*kbss_en2_6585)&kbss_zhk_idx&kbss_w_idx
#kbss_hk_idx=
kbss_jh_idx=(np.abs(kbss_zh-kbss_zj)<0.01)&(kbss_wj/kbss_wh>0.5)&(kbss_wj/kbss_wh<2.)

## Read in KBSS Lya measurements

#lbg_sname,lbg_spec,lbg_llya,lbg_wlya,lbg_zlya,lbg_sflux,lbg_sflerr,lbg_seqw,lbg_seqwerr,lbg_npks,lbg_peak2,lbg_lopeak,lbg_hipeak,lbg_pval,lbg_flya,lbg_widthA,lbg_fflux,lbg_fp2,lbg_fw2A,lbg_alya,lbg_aw1bA,lbg_aw1rA,lbg_ap2,lbg_aw2A,lbg_cA1,lbg_cA2,lbg_cmu1,lbg_cmu2,lbg_csig1a,lbg_csig1b,lbg_csigma2,lbg_frat,lbg_aA1,lbg_aA2,lbg_bglev,lbg_bgrms,lbg_slope,lbg_wlya_cut,lbg_wlya_em,lbg_wlya_abs=readcol('lbg_lya_all.yc.v10.cut',twod=False)

#lbg_sname,lbg_spec,lbg_llya,lbg_wlya,lbg_zlya,lbg_sflux,lbg_sflerr,lbg_seqw,lbg_seqwerr,lbg_npks,lbg_peak2,lbg_lopeak,lbg_hipeak,lbg_pval,lbg_flya,lbg_widthA,lbg_fflux,lbg_fp2,lbg_fw2A,lbg_alya,lbg_aw1bA,lbg_aw1rA,lbg_ap2,lbg_aw2A,lbg_cA1,lbg_cA2,lbg_cmu1,lbg_cmu2,lbg_csig1a,lbg_csig1b,lbg_csigma2,lbg_frat,lbg_aA1,lbg_aA2,lbg_bglev,lbg_bgrms,lbg_slope,lbg_wlya_cut,lbg_wlya_em,lbg_wlya_abs,lbg_ew_lis,lbg_ew_lis_err,lbg_ew_lis_err2,lbg_ew_lis_cut,lbg_ew_lis_cut_err,lbg_ew_lis_sig,lbg_ew_lis_sig_err,lbg_nlis,lbg_nlis_cut,lbg_nlis_sig,lbg_lis_cont,lbg_lis_cont_err=readcol('lbg_lya_all_lis.yc.v1',twod=False)

t1=ascii.read('lbg_lya_all_lis.yc.v1')

lbg_field,lbg_spec,lbg_llya,lbg_wlya,lbg_zlya,lbg_sflux,lbg_sflerr,lbg_seqw,lbg_seqwerr,lbg_npks,lbg_peak2,lbg_lopeak,lbg_hipeak,lbg_pval,lbg_flya,lbg_widthA,lbg_fflux,lbg_fp2,lbg_fw2A,lbg_alya,lbg_aw1bA,lbg_aw1rA,lbg_ap2,lbg_aw2A,lbg_cA1,lbg_cA2,lbg_cmu1,lbg_cmu2,lbg_csig1a,lbg_csig1b,lbg_csigma2,lbg_frat,lbg_aA1,lbg_aA2,lbg_bglev,lbg_bgrms,lbg_slope,lbg_wlya_cut,lbg_wlya_em,lbg_wlya_abs,lbg_ew_lis,lbg_ew_lis_err,lbg_ew_lis_err2,lbg_ew_lis_cut,lbg_ew_lis_cut_err,lbg_ew_lis_sig,lbg_ew_lis_sig_err,lbg_nlis,lbg_nlis_cut,lbg_nlis_sig,lbg_lis_cont,lbg_lis_cont_err=(np.asarray(x) for x in t1.columns.values())

# reformat lbg_name because Yuguang's spectra are named differently from mine
lbg_sname=np.empty(len(lbg_field),dtype='object')
for ii in range(len(lbg_spec)):
    v=lbg_spec[ii].split('.')
    lbg_sname[ii] = 'Q'+v[0].lstrip('q')+'-'+v[1]


lbg_awkms=(lbg_aw1bA+lbg_aw1bA)/(2*lbg_flya)*3e5

lbg_vlya=(lbg_flya-1215.67)/(1215.67)*3e5
lbg_vwlya=(lbg_wlya-1215.67)/(1215.67)*3e5
lbg_vwlya_em=(lbg_wlya_em-1215.67)/(1215.67)*3e5
lbg_vwlya_abs=(lbg_wlya_abs-1215.67)/(1215.67)*3e5
lbg_vwlya_cut=(lbg_wlya_cut-1215.67)/(1215.67)*3e5
    
lbg_lmidx=[]
kbss_lmidx=[]
for ii in range(len(kbss_obj)):
    obj = kbss_obj[ii]
    idx = lbg_sname==obj
    if idx.sum()>1:
        kbss_lmidx.append(np.arange(len(kbss_obj))[ii])
        LM1idx=np.array(['LM1' in sp for sp in lbg_spec[idx]])
        if LM1idx.sum()>0:
            lbg_lmidx.append(np.arange(len(lbg_sname))[idx][LM1idx][0])
            continue
        idx600=np.array(['600' in sp for sp in lbg_spec[idx]])
        if idx600.sum()>0:
            lbg_lmidx.append(np.arange(len(lbg_sname))[idx][idx600][0])
            continue
        lbg_lmidx.append(np.arange(len(lbg_sname))[idx][0])
    elif idx.sum()>0:
        kbss_lmidx.append(np.arange(len(kbss_obj))[ii])
        lbg_lmidx.append(np.arange(len(lbg_sname))[idx][0])

lbg_log_seqw=np.zeros(len(lbg_seqw))
lbg_log_seqw[lbg_seqw>0]=np.log(1+lbg_seqw[lbg_seqw>0])
lbg_log_seqw[lbg_seqw<0]=-np.log(1-lbg_seqw[lbg_seqw<0])


## Create index of objects flagged as AGN
#agnlist=readcol('kbss_agn_and_qsos.list',twod=False)[0]
agnlist=ascii.read('kbss_agn_and_qsos.list').columns.values()[0]
agnidx=np.array([obj in agnlist for obj in kbss_obj])

idx_lm_snr=((kbss_fha>5*kbss_eha)&(kbss_fhb>3*kbss_ehb)&(kbss_fo3_5008>0)&(kbss_fn2_6585>=2*kbss_en2_6585)&(agnidx==False))[kbss_lmidx]
idx_lm_lim=((kbss_fha>5*kbss_eha)&(kbss_fhb>3*kbss_ehb)&(kbss_fo3_5008>0)&(kbss_fn2_6585<2*kbss_en2_6585)&(agnidx==False))[kbss_lmidx]
idx_lm_agn=((kbss_fha>5*kbss_eha)&(kbss_fhb>3*kbss_ehb)&(kbss_fo3_5008>0)&(kbss_fn2_6585>=2*kbss_en2_6585)&(agnidx==True))[kbss_lmidx]

# Define Lya emitters, intermediates, and absorbers
idx_obs=((kbss_fn2_6585!=0)&(kbss_fha!=0)&(kbss_fo3_5008!=0)&(kbss_fhb!=0))[kbss_lmidx]
idx_lae=(lbg_seqw[lbg_lmidx]>=20)&idx_obs
idx_lem=(lbg_seqw[lbg_lmidx]<20)&(lbg_seqw[lbg_lmidx]>0)&idx_obs
idx_lab=(lbg_seqw[lbg_lmidx]<=0)&idx_obs
idx_lab1=(lbg_seqw[lbg_lmidx]<=0)&(lbg_seqw[lbg_lmidx]>np.median(lbg_seqw[lbg_lmidx][idx_lab]))&idx_obs
idx_lab2=(lbg_seqw[lbg_lmidx]>np.median(lbg_seqw[lbg_lmidx][idx_lab]))&idx_obs

pl.figure(1)
pl.clf()

kbss_eha_cor=kbss_fha*np.sqrt((kbss_eha/kbss_fha)**2+(kbss_kscor_err/kbss_kscor)**2)
kbss_ehb_cor=kbss_fhb*np.sqrt((kbss_ehb/kbss_fhb)**2+(kbss_hscor_err/kbss_hscor)**2)
idx_lm_hahb=(1/np.sqrt((kbss_eha_cor/kbss_fha)**2+(kbss_ehb_cor/kbss_fhb)**2)[kbss_lmidx]>5)
#idx_lm_hkflag=(kbss_kscor[kbss_lmidx]<3)&(kbss_hscor[kbss_lmidx]<3)
idx_lm_hkflag=idx_lm_hahb&(np.abs(kbss_zh-kbss_zk)[kbss_lmidx]<0.01)

pl.axhline(0,color='k',ls='--')

kbss_ha_ext=kl.ebv2ext(6564,kl.ebv_neb((kbss_fha/kbss_fhb),2.86,'ccm'),'ccm')

cosmo=cosm.FlatLambdaCDM(H0=70,Om0=0.3)
kbss_dlum=cosmo.luminosity_distance(kbss_zneb).cgs.value
kbss_sfr=4*np.pi*kbss_dlum**2*1e-17/1.26e41*kbss_fha/kbss_ha_ext

pl.plot(np.log10(kbss_sfr)[kbss_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)],lbg_seqw[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)],'.',color='0.5',ms=10,mew=0)
#rstat,pstat=stats.spearmanr(np.log10(kbss_fo3_5008/kbss_fhb)[kbss_lmidx][idx_lm_snr|idx_lm_lim],lbg_seqw[lbg_lmidx][idx_lm_snr|idx_lm_lim])
pl.xlabel('$\mathrm{log(SFR}_{\mathrm{H}\\alpha})$ $[\mathrm{M}_\odot$ $\mathrm{yr}^{-1}]$',size=24)
pl.ylabel('$W_{\mathrm{Ly}\\alpha}$ $[\AA]$',size=24)
pl.xticks(size=16)
pl.yticks(size=16)
ax1=pl.gca()
ax1.xaxis.set_major_locator(MultipleLocator(0.5))
ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
ax1.yaxis.set_major_locator(MultipleLocator(50))
ax1.yaxis.set_minor_locator(MultipleLocator(10))
ax1.tick_params('both',which='major',length=5)
ax1.tick_params('both',which='minor',length=3)
pl.subplots_adjust(left=0.14,right=0.97,top=0.95,bottom=0.16)

pl.savefig('plots/2017/wlya_vs_SFR_yc.pdf')
#pl.close()
nstat=len(np.log10(kbss_sfr)[kbss_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)])
rstat,pstat=stats.spearmanr(np.log10(kbss_sfr)[kbss_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)],lbg_seqw[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)])
fout.write('$W_{\mathrm{Ly}\\alpha}$ vs SFR & %3d & %5.2f & %3.1f\\\\ \n' % (nstat,rstat,np.log10(pstat)))

pl.figure(1)
pl.clf()

pl.axhline(0,color='k',ls='--')

idx_lm_mass=kbss_lmstar[kbss_lmidx]>0

pl.plot(np.log10(kbss_sfr/10**kbss_lmstar)[kbss_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass],lbg_vwlya[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass],'.',color='0.5',ms=10,mew=0)
pl.xlabel('$\mathrm{log(sSFR}_{\mathrm{H}\\alpha})$ $[\mathrm{yr}^{-1}]$',size=24)
pl.ylabel('$W_{\mathrm{Ly}\\alpha}$ $[\AA]$',size=24)

pl.axis([-10.5,-6.5,-500,2000])

pl.xticks(size=16)
pl.yticks(size=16)
ax1=pl.gca()
ax1.xaxis.set_major_locator(MultipleLocator(0.5))
ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
ax1.yaxis.set_major_locator(MultipleLocator(500))
ax1.yaxis.set_minor_locator(MultipleLocator(100))
ax1.tick_params('both',which='major',length=5)
ax1.tick_params('both',which='minor',length=3)
pl.subplots_adjust(left=0.14,right=0.97,top=0.95,bottom=0.16)


pl.savefig('plots/2017/wlya_vs_sSFR_yc.pdf')
#pl.close()
nstat=len(np.log10(kbss_sfr/10**kbss_lmstar)[kbss_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass])
rstat,pstat=stats.spearmanr(np.log10(kbss_sfr/10**kbss_lmstar)[kbss_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass],lbg_seqw[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass])
fout.write('$W_{\mathrm{Ly}\\alpha}$ vs sSFR & %3d & %5.2f & %3.1f\\\\ \n' % (nstat,rstat,np.log10(pstat)))

pl.figure(1)
pl.clf()

pl.axhline(0,color='k',ls='--')

idx_lm_mass=kbss_lmstar[kbss_lmidx]>0

pl.plot(np.log10(kbss_sfr/10**kbss_lmstar)[kbss_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass],lbg_vwlya[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass],'.',color='0.5',ms=10,mew=0)
pl.plot(np.log10(kbss_sfr/10**kbss_lmstar)[kbss_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass],lbg_vwlya_em[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass],'.',color='b',ms=10,mew=0)

pl.axis([-10.5,-6.5,-500,2000])

pl.xlabel('$\mathrm{log(sSFR}_{\mathrm{H}\\alpha})$ $[\mathrm{yr}^{-1}]$',size=24)
pl.ylabel('$v_{\mathrm{Ly}\\alpha}$ $[\mathrm{km}$ $\mathrm{s}^{-1}]$',size=24)
pl.xticks(size=16)
pl.yticks(size=16)
ax1=pl.gca()
ax1.xaxis.set_major_locator(MultipleLocator(0.5))
ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
ax1.yaxis.set_major_locator(MultipleLocator(500))
ax1.yaxis.set_minor_locator(MultipleLocator(100))
ax1.tick_params('both',which='major',length=5)
ax1.tick_params('both',which='minor',length=3)
pl.subplots_adjust(left=0.14,right=0.97,top=0.95,bottom=0.16)


pl.savefig('plots/2017/vwlya_vs_sSFR_yc.pdf')

nstat=len(np.log10(kbss_sfr/10**kbss_lmstar)[kbss_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&(~np.isnan(lbg_vwlya_em[lbg_lmidx]))])
rstat,pstat=stats.spearmanr((kbss_sfr/10**kbss_lmstar)[kbss_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&(~np.isnan(lbg_vwlya_em[lbg_lmidx]))],lbg_vwlya[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&(~np.isnan(lbg_vwlya_em[lbg_lmidx]))])
fout.write('$v_{\mathrm{Ly}\\alpha\mathrm{,em}}$ vs sSFR & %3d & %5.2f & %3.1f\\\\ \n' % (nstat,rstat,np.log10(pstat)))

pl.figure(1)
pl.clf()

pl.axhline(0,color='k',ls='--')

idx_lm_z=((kbss_zh>2)|(kbss_zk>2))[kbss_lmidx]
idx_lm_mass=kbss_lmstar[kbss_lmidx]>0

pthresh=0.05
lbg_sig_em=lbg_pval<pthresh

pl.plot(np.log10(kbss_sfr/10**kbss_lmstar)[kbss_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass],lbg_vwlya[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass],'.',color='0.5',ms=10,mew=0)
pl.plot(np.log10(kbss_sfr/10**kbss_lmstar)[kbss_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&lbg_sig_em[lbg_lmidx]],lbg_vwlya[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&lbg_sig_em[lbg_lmidx]],'.',color='r',ms=10,mew=0)
pl.xlabel('$\mathrm{log(sSFR}_{\mathrm{H}\\alpha})$ $[\mathrm{yr}^{-1}]$',size=24)
pl.ylabel('$W_{\mathrm{Ly}\\alpha}$ $[\AA]$',size=24)
pl.xticks(size=16)
pl.yticks(size=16)

pl.axis([-10.5,-6.7,-1000,1500])

ax1=pl.gca()
ax1.xaxis.set_major_locator(MultipleLocator(0.5))
ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
ax1.yaxis.set_major_locator(MultipleLocator(1000))
ax1.yaxis.set_minor_locator(MultipleLocator(250))
ax1.tick_params('both',which='major',length=5)
ax1.tick_params('both',which='minor',length=3)
pl.subplots_adjust(left=0.14,right=0.97,top=0.95,bottom=0.16)

rstat,pstat=stats.spearmanr(np.log10(kbss_sfr/10**kbss_lmstar)[kbss_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&lbg_sig_em[lbg_lmidx]],lbg_vwlya[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&lbg_sig_em[lbg_lmidx]])

pl.savefig('plots/2017/vwlya_vs_sSFR_yc.pdf')
#pl.close()

nstat=len(np.log10(kbss_sfr/10**kbss_lmstar)[kbss_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&(~np.isnan(lbg_vwlya_em[lbg_lmidx]))])
rstat,pstat=stats.spearmanr((kbss_sfr/10**kbss_lmstar)[kbss_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&(~np.isnan(lbg_vwlya_em[lbg_lmidx]))],lbg_vwlya_em[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&(~np.isnan(lbg_vwlya_em[lbg_lmidx]))])
fout.write('$v_{\mathrm{Ly}\\alpha\mathrm{,tot}}$ vs sSFR & %3d & %5.2f & %3.1f\\\\ \n' % (nstat,rstat,np.log10(pstat)))

pl.figure(1)
pl.clf()

pl.axhline(0,color='k',ls='--')

#pl.plot(lbg_vwlya[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&lbg_sig_em[lbg_lmidx]],lbg_seqw[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&lbg_sig_em[lbg_lmidx]],'.',color='r',ms=10,mew=0)

pl.plot(lbg_vwlya[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)],lbg_seqw[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)],'.',color='g',ms=10,mew=0)

pl.plot(lbg_vwlya_em[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)],lbg_seqw[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)],'.',color='b',ms=10,mew=0)

pl.plot(lbg_vwlya_abs[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)],lbg_seqw[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)],'.',color='r',ms=10,mew=0)

pl.plot(lbg_vwlya[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&lbg_sig_em[lbg_lmidx]],lbg_seqw[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&lbg_sig_em[lbg_lmidx]],'.',color='b',ms=10,mew=0)

#pl.plot(lbg_vlya[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&(lbg_vlya[lbg_lmidx]>-500)&(lbg_vlya[lbg_lmidx]<2500)],lbg_seqw[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&(lbg_vlya[lbg_lmidx]>-500)&(lbg_vlya[lbg_lmidx]<2500)],'.',color='b',ms=10,mew=0)

pl.xlabel('$v_{\mathrm{Ly}\\alpha}$ $[\mathrm{km}$ $\mathrm{s}^{-1}]$',size=24)
pl.ylabel('$W_{\mathrm{Ly}\\alpha}$ $[\AA]$',size=24)
pl.xticks(size=16)
pl.yticks(size=16)

pl.axis([-1000,1700,-20,150])

ax1=pl.gca()
ax1.xaxis.set_major_locator(MultipleLocator(500))
ax1.xaxis.set_minor_locator(MultipleLocator(100))
ax1.yaxis.set_major_locator(MultipleLocator(25))
ax1.yaxis.set_minor_locator(MultipleLocator(5))
ax1.tick_params('both',which='major',length=5)
ax1.tick_params('both',which='minor',length=3)
pl.subplots_adjust(left=0.14,right=0.97,top=0.95,bottom=0.16)

pl.savefig('plots/2017/w_vs_vwlya_yc.pdf')


nstat=len(lbg_vwlya[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&lbg_sig_em[lbg_lmidx]])
rstat,pstat=stats.spearmanr(lbg_vwlya[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&lbg_sig_em[lbg_lmidx]],lbg_seqw[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&lbg_sig_em[lbg_lmidx]])
fout.write('$W_{\mathrm{Ly}\\alpha}$ vs $v_{\mathrm{Ly}\\alpha}$ & %3d & %5.2f & %3.1f\\\\ \n' % (nstat,rstat,np.log10(pstat)))



#lbg_sname[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&lbg_sig_em[lbg_lmidx]&(lbg_vlya[lbg_lmidx]<-150)]

bad_obj=['Q0142-BX67','Q0207-BX304','Q1442-MD53',
         ]
inverted_obj=['Q0449-BX141','Q0821-BX45','Q1700-BX711',
              ]

#pl.plot(lbg_vlya[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass],lbg_seqw[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass],'.',color='r',ms=10,mew=0)

bad_obj=['Q1623-BX366','Q1700-MD109','Q1549-BX58']


rstat,pstat=stats.spearmanr(lbg_vwlya[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&(lbg_vwlya[lbg_lmidx]>-500)&(lbg_vwlya[lbg_lmidx]<2500)],lbg_seqw[lbg_lmidx][(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass&(lbg_vwlya[lbg_lmidx]>-500)&(lbg_vwlya[lbg_lmidx]<2500)])


## Make ordered lists of objects for stacking rolling averages
cosmo=cosm.FlatLambdaCDM(H0=70,Om0=0.3)
kbss_dlum=cosmo.luminosity_distance(kbss_zh).cgs.value
kbss_lumcorr=4*np.pi*kbss_dlum**2/1e58

hk_mass_idx=(idx_lm_snr|idx_lm_lim)&(idx_lm_hkflag)&idx_lm_mass
kbsscut_ebv=kl.ebv_neb((kbss_fha/kbss_fhb)[kbss_lmidx][hk_mass_idx],2.86)
kbsscut_sfr=kbss_fha[kbss_lmidx][hk_mass_idx]/kl.ebv2ext(6564,kbsscut_ebv)

sfr_order_idx=np.argsort(kbsscut_sfr)
f1=open('lbg_spec_sfr_ordered.list','w')
for ii in range(hk_mass_idx.sum()):
    f1.write('/yuguang_lris/rest/%s 0.0\n' % lbg_spec[lbg_lmidx][hk_mass_idx][sfr_order_idx][ii])
f1.close()

mass_idx=idx_lm_mass&idx_lm_z
mass_order_idx=np.argsort(kbss_lmstar[kbss_lmidx][mass_idx])
f1=open('lbg_spec_mass_ordered.list','w')
for ii in range(mass_idx.sum()):
    f1.write('/yuguang_lris/rest/%s 0.0\n' % lbg_spec[lbg_lmidx][mass_idx][mass_order_idx][ii])
f1.close()

ssfr_order_idx=np.argsort(kbsscut_sfr/kbss_lmstar[kbss_lmidx][hk_mass_idx])
f1=open('lbg_spec_ssfr_ordered.list','w')
for ii in range(hk_mass_idx.sum()):
    f1.write('/yuguang_lris/rest/%s 0.0\n' % lbg_spec[lbg_lmidx][hk_mass_idx][ssfr_order_idx][ii])
f1.close()

o3_idx=((kbss_fhb>3*kbss_ehb)&(kbss_fo3_5008>3*kbss_eo3_5008)&(agnidx==False))[kbss_lmidx]&idx_lm_z
o3_order_idx=np.argsort((kbss_fo3_5008/kbss_fhb)[kbss_lmidx][o3_idx])
f1=open('lbg_spec_o3_ordered.list','w')
for ii in range(o3_idx.sum()):
    f1.write('/yuguang_lris/rest/%s 0.0\n' % lbg_spec[lbg_lmidx][o3_idx][o3_order_idx][ii])
f1.close()

ebv_order_idx=np.argsort(kbsscut_ebv)
f1=open('lbg_spec_ebv_ordered.list','w')
for ii in range(hk_mass_idx.sum()):
    f1.write('/yuguang_lris/rest/%s 0.0\n' % lbg_spec[lbg_lmidx][hk_mass_idx][ebv_order_idx][ii])
f1.close()

## Make Lya escape fraction histogram

pl.figure(4)
pl.clf()
pl.hist(lbg_fflux[lbg_lmidx][kbss_fha[kbss_lmidx]>0]/kbss_fha[kbss_lmidx][kbss_fha[kbss_lmidx]>0]/8.6,bins=np.arange(-2,5,0.1));

## Make Lya escape vs. EW plot
#lae_lyaha,lae_eqw=readcol('lae_fesc_eqw.dat',twod=False)
t1=ascii.read('lae_fesc_eqw.dat')
lae_lyaha,lae_eqw=(np.asarray(x) for x in t1.columns.values())

pl.figure(5)
pl.clf()
idx=(kbss_fha[kbss_lmidx]>0)&(lbg_fflux[lbg_lmidx]>0)&(lbg_seqw[lbg_lmidx]>0.1)
pl.loglog(lbg_seqw[lbg_lmidx][idx],lbg_fflux[lbg_lmidx][idx]/kbss_fha[kbss_lmidx][idx]/8.7,'.',ms=12)
pl.plot(lae_eqw,lae_lyaha/8.7,'rs')
pl.axis([0.1,1e3,3e-4,3])
pl.xticks([0.1,1,10,100,1000],[0.1,1,10,100,1000])
pl.yticks([1e-3,1e-2,1e-1,1],[1e-3,1e-2,1e-1,1])
pl.xlabel('$W_{\mathrm{Ly}\\alpha}$ $[\AA]$',size=24)
pl.ylabel('$f_{\mathrm{esc,Ly}\\alpha}$',size=24)
ax1=pl.gca()
#ax1.xaxis.set_major_locator(LogLocator(base=10.))
#ax1.yaxis.set_major_locator(LogLocator(base=10.))
ax1.tick_params('both',which='major',length=5)
ax1.tick_params('both',which='minor',length=3)
pl.subplots_adjust(left=0.18,right=0.95,top=0.95,bottom=0.17)
pl.savefig('plots/2017/fesc_vs_eqw_yc.pdf')

nstat=len(lbg_seqw[lbg_lmidx][idx])
rstat,pstat=stats.spearmanr(lbg_seqw[lbg_lmidx][idx],lbg_fflux[lbg_lmidx][idx]/kbss_fha[kbss_lmidx][idx]/8.7)
fout.write('$f_\mathrm{esc}$ vs $W_{\mathrm{Ly}\\alpha}$ & %3d & %5.2f & %3.1f\\\\ \n' % (nstat,rstat,np.log10(pstat)))


kbsscut_ha_dustcor=kbss_fha[kbss_lmidx][hk_mass_idx&idx]/kl.ebv2ext(6564,kbsscut_ebv[idx[hk_mass_idx]])
kbsscut_lya_dustcor=lbg_fflux[lbg_lmidx][hk_mass_idx&idx]/kl.ebv2ext(1216,kbsscut_ebv[idx[hk_mass_idx]])

pl.figure(5)
pl.clf()
idx=(kbss_fha[kbss_lmidx]>0)&(lbg_fflux[lbg_lmidx]>0)#&(lbg_seqw[lbg_lmidx]>0.1)
pl.loglog(10**kbss_lmstar[kbss_lmidx][idx],lbg_fflux[lbg_lmidx][idx]/kbss_fha[kbss_lmidx][idx]/8.7,'.')
#pl.plot(lae_eqw,lae_lyaha/8.7,'rs')
pl.axis([1e8,1e12,3e-4,3])
#pl.xticks([0.1,1,10,100,1000],[0.1,1,10,100,1000])
pl.yticks([1e-3,1e-2,1e-1,1],[1e-3,1e-2,1e-1,1])
pl.xlabel('$M_*$ $[M_\odot]$',size=24)
pl.ylabel('$f_{\mathrm{esc,Ly}\\alpha}$',size=24)
ax1=pl.gca()
#ax1.xaxis.set_major_locator(LogLocator(base=10.))
#ax1.yaxis.set_major_locator(LogLocator(base=10.))
ax1.tick_params('both',which='major',length=5)
ax1.tick_params('both',which='minor',length=3)
pl.subplots_adjust(left=0.18,right=0.95,top=0.95,bottom=0.17)
pl.savefig('plots/2017/fesc_vs_mass_yc.pdf')

nstat=len(kbss_lmstar[kbss_lmidx][idx])
rstat,pstat=stats.spearmanr(10**kbss_lmstar[kbss_lmidx][idx],lbg_fflux[lbg_lmidx][idx]/kbss_fha[kbss_lmidx][idx]/8.7)
fout.write('$f_\mathrm{esc}$ vs $M_*$ & %3d & %5.2f & %3.1f\\\\ \n' % (nstat,rstat,np.log10(pstat)))

pl.figure(5)
pl.clf()
idx=(kbss_fhb[kbss_lmidx]>0)&(kbss_fo3_5008[kbss_lmidx]>0)&(kbss_fha[kbss_lmidx]>0)&(lbg_fflux[lbg_lmidx]>0)
pl.semilogy(np.log10(kbss_fo3_5008/kbss_fhb)[kbss_lmidx][idx],lbg_fflux[lbg_lmidx][idx]/kbss_fha[kbss_lmidx][idx]/8.7,'.')
#pl.plot(lae_eqw,lae_lyaha/8.7,'rs')
pl.axis([-0.1,1.2,2e-4,3])
#pl.xticks([0.1,1,10,100,1000],[0.1,1,10,100,1000])
pl.yticks([1e-3,1e-2,1e-1,1],[1e-3,1e-2,1e-1,1])
pl.xlabel('$\mathrm{log}([\mathrm{O III}]/\mathrm{H}\\beta)$',size=24)
pl.ylabel('$f_{\mathrm{esc,Ly}\\alpha}$',size=24)
ax1=pl.gca()
#ax1.xaxis.set_major_locator(LogLocator(base=10.))
#ax1.yaxis.set_major_locator(LogLocator(base=10.))
ax1.tick_params('both',which='major',length=5)
ax1.tick_params('both',which='minor',length=3)
pl.subplots_adjust(left=0.18,right=0.95,top=0.95,bottom=0.17)
pl.savefig('plots/2017/fesc_vs_o3_yc.pdf')

nstat=len(np.log10(kbss_fo3_5008/kbss_fhb)[kbss_lmidx][idx])
rstat,pstat=stats.spearmanr(np.log10(kbss_fo3_5008/kbss_fhb)[kbss_lmidx][idx],lbg_fflux[lbg_lmidx][idx]/kbss_fha[kbss_lmidx][idx]/8.7)
fout.write('$f_\mathrm{esc}$ vs O3 & %3d & %5.2f & %3.1f\\\\ \n' % (nstat,rstat,np.log10(pstat)))


pl.figure(5)
pl.clf()
idx=(kbss_fhb[kbss_lmidx]>0)&(kbss_fo3_5008[kbss_lmidx]>0)
#pl.semilogx(10**kbss_lmstar[kbss_lmidx][idx],np.log10(kbss_fo3_5008/kbss_fhb)[kbss_lmidx][idx],'.')
pl.scatter(10**kbss_lmstar[kbss_lmidx][idx],np.log10(kbss_fo3_5008/kbss_fhb)[kbss_lmidx][idx],cmap=mpl.cm.coolwarm_r,c=lbg_seqw[lbg_lmidx][idx],s=10+np.abs(lbg_seqw[lbg_lmidx][idx]),vmin=-25,vmax=25)
pl.semilogx()
#idx=(kbss_fhb>0)&(kbss_fo3_5008>0)
#pl.semilogx(10**kbss_lmstar[idx],np.log10(kbss_fo3_5008/kbss_fhb)[idx],'.')
pl.axis([4e8,3e11,-0.2,1.3])
#pl.xticks([0.1,1,10,100,1000],[0.1,1,10,100,1000])
#pl.yticks([1e-3,1e-2,1e-1,1],[1e-3,1e-2,1e-1,1])
pl.xlabel('$M_*$ $[M_\odot]$',size=24)
pl.ylabel('$\mathrm{log}([\mathrm{O III}]/\mathrm{H}\\beta)$',size=24)
ax1=pl.gca()
ax1.xaxis.set_major_locator(LogLocator(base=10.))
ax1.yaxis.set_major_locator(MultipleLocator(0.2))
ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
ax1.tick_params('both',which='major',length=5)
ax1.tick_params('both',which='minor',length=3)
cb1=pl.colorbar(extend='both',fraction=0.05)
cb1.solids.set_edgecolor("face")
pl.subplots_adjust(left=0.17,right=0.92,top=0.95,bottom=0.17)
pl.savefig('plots/2017/o3_vs_mass_yc.pdf')

nstat=idx.sum()
rstat,pstat=stats.spearmanr(kbss_lmstar[kbss_lmidx][idx],np.log10(kbss_fo3_5008/kbss_fhb)[kbss_lmidx][idx])
fout.write('O3 vs $M_*$ & %3d & %5.2f & %3.1f\\\\ \n' % (nstat,rstat,np.log10(pstat)))


pl.figure(5)
pl.clf()

fesc_idx=(kbss_fha[kbss_lmidx]>0)&(lbg_fflux[lbg_lmidx]>0)
idx_lm_o32=((kbss_fo3_5008>3*kbss_eo3_5008)&(kbss_fo2>3*kbss_eo2)&(kbss_fha>5*kbss_eha)&(kbss_fhb>3*kbss_ehb))[kbss_lmidx]&(idx_lm_agn==False)
idx_lm_o32_lim=((kbss_fo3_5008>3*kbss_eo3_5008)&(kbss_fo2<=3*kbss_eo2)&(kbss_fha>5*kbss_eha)&(kbss_fhb>3*kbss_ehb))[kbss_lmidx]&(idx_lm_agn==False)
idx_lm_jhflag=kbss_jh_idx[kbss_lmidx]

o32_ind_ext=kl.ebv2ext([5008,4960,3728],kl.ebv_neb((kbss_fha/kbss_fhb)[kbss_lmidx],2.86,'ccm'),'ccm')
o32_corr=o32_ind_ext[:,2]/(0.75*o32_ind_ext[:,0]+0.25*o32_ind_ext[:,1])

pl.semilogy(np.log10(o32_corr*((kbss_fo3_5008+kbss_fo3_4960)/kbss_fo2)[kbss_lmidx])[idx_lm_o32&fesc_idx],(lbg_fflux[lbg_lmidx]/kbss_fha[kbss_lmidx]/8.7)[idx_lm_o32&fesc_idx],'.')

pl.axis([-0.6,1.3,2e-4,3])
#pl.xticks([0.1,1,10,100,1000],[0.1,1,10,100,1000])
pl.yticks([1e-3,1e-2,1e-1,1],[1e-3,1e-2,1e-1,1])
pl.xlabel('$\mathrm{log}([\mathrm{O III}]/[\mathrm{O II}])$',size=24)
pl.ylabel('$f_{\mathrm{esc,Ly}\\alpha}$',size=24)
ax1=pl.gca()
#ax1.xaxis.set_major_locator(LogLocator(base=10.))
#ax1.yaxis.set_major_locator(LogLocator(base=10.))
ax1.xaxis.set_major_locator(MultipleLocator(0.5))
ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
ax1.tick_params('both',which='major',length=5)
ax1.tick_params('both',which='minor',length=3)
pl.subplots_adjust(left=0.18,right=0.95,top=0.95,bottom=0.17)
pl.savefig('plots/2017/fesc_vs_o32_yc.pdf')

nstat=len(np.log10(o32_corr*((kbss_fo3_5008+kbss_fo3_4960)/kbss_fo2)[kbss_lmidx])[idx_lm_o32&fesc_idx])
rstat,pstat=stats.spearmanr(np.log10(o32_corr*((kbss_fo3_5008+kbss_fo3_4960)/kbss_fo2)[kbss_lmidx])[idx_lm_o32&fesc_idx],(lbg_fflux[lbg_lmidx]/kbss_fha[kbss_lmidx]/8.7)[idx_lm_o32&fesc_idx])
fout.write('$f_\mathrm{esc}$ vs O32 & %3d & %5.2f & %3.1f\\\\ \n' % (nstat,rstat,np.log10(pstat)))


# Make EW_lya vs EW_lis plots

lis_idx=np.isnan(lbg_ew_lis)==False
list_ewlya = np.arange(-50,201,1.25)
list_ewlis = np.arange(-5,10.01,0.1)
Z=stack_gauss_densities(lbg_seqw[lis_idx],lbg_ew_lis[lis_idx],np.maximum(lbg_seqwerr[lis_idx],1),lbg_ew_lis_err[lis_idx],list_ewlya,list_ewlis)

pl.figure(6)
pl.clf()
pl.imshow(np.log10(Z),cmap=viridis)
xticks=np.arange(-50,201,50,dtype=int)
pl.xticks((xticks+50)/1.25,xticks)
yticks=np.arange(-5,10.1,5,dtype=int)
pl.yticks((yticks+5)/0.1,yticks)

pl.axhline(5/0.1,color='k',ls='--')
pl.axvline(50/1.25,color='k',ls='--')

pl.figure(7)
pl.clf()
#pl.contour(np.log10(Z),cmap=viridis,levels=np.arange(-5,1.5,0.5))
pl.contour(Z,cmap=viridis,levels=np.arange(1,15))

sns.set(style='whitegrid', font='serif',font_scale=2.0,palette='deep')
pl.figure(8)
pl.clf()
pl.errorbar(lbg_seqw[lis_idx],lbg_ew_lis[lis_idx],xerr=np.maximum(lbg_seqwerr[lis_idx],1),yerr=lbg_ew_lis_err[lis_idx],ls='none',zorder=1,c='0.5')

pl.scatter(lbg_seqw[lis_idx],lbg_ew_lis[lis_idx],c=np.log10(lbg_lis_cont/lbg_lis_cont_err)[lis_idx],cmap=viridis,vmin=1,vmax=2.5,s=30,zorder=2)

# redo plot with cut on SNR in continuum
sns.set(style='white', font='serif',font_scale=2.0,palette='deep')
pl.figure(8)
pl.clf()

lis_idx_good=lis_idx&(lbg_lis_cont/lbg_lis_cont_err>10)

pl.errorbar(lbg_seqw[lis_idx_good],-lbg_ew_lis[lis_idx_good],xerr=np.maximum(lbg_seqwerr[lis_idx_good],1),yerr=lbg_ew_lis_err[lis_idx_good],ls='none',zorder=1,c='0.5')

pl.scatter(lbg_seqw[lis_idx_good],-lbg_ew_lis[lis_idx_good],c=np.log10(lbg_lis_cont/lbg_lis_cont_err)[lis_idx_good],cmap=viridis,vmin=1,vmax=2.5,s=30,zorder=2)

pl.axis([-50,220,8,-13])

pl.xlabel('$\mathrm{EW}_{\mathrm{Ly}\\alpha}$ $[\AA]$',size=24)
pl.ylabel('$\mathrm{EW}_\mathrm{LIS}$ $[\AA]$',size=24,labelpad=-5)

ax1=pl.gca()
ax1.xaxis.set_major_locator(MultipleLocator(50))
ax1.xaxis.set_minor_locator(MultipleLocator(10))
ax1.yaxis.set_major_locator(MultipleLocator(5))
ax1.yaxis.set_minor_locator(MultipleLocator(1))
ax1.tick_params('both',which='major',length=5)
ax1.tick_params('both',which='minor',length=3)
pl.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.17)
pl.savefig('plots/2017/lis_vs_lya_yc.pdf')

nstat=len(lbg_seqw[lis_idx_good])
rstat,pstat=stats.spearmanr(lbg_seqw[lis_idx_good],-lbg_ew_lis[lis_idx_good])
fout.write('$W_\mathrm{LIS}$ vs $W_{\mathrm{Ly}\\alpha}$ & %3d & %5.2f & %3.1f\\\\ \n' % (nstat,rstat,np.log10(pstat)))

# redo with black and white

pl.figure(8)
pl.clf()

lis_idx_good=lis_idx&(lbg_lis_cont/lbg_lis_cont_err>10)

pl.axvline(0,c='k',ls='--',zorder=-1,lw=1)
pl.axhline(0,c='k',ls='--',zorder=-1,lw=1)

pl.errorbar(lbg_seqw[lis_idx_good],-lbg_ew_lis[lis_idx_good],xerr=np.maximum(lbg_seqwerr[lis_idx_good],1),yerr=lbg_ew_lis_err[lis_idx_good],ls='none',zorder=1,c='0.5',lw=1)

pl.scatter(lbg_seqw[lis_idx_good],-lbg_ew_lis[lis_idx_good],c=sns_deep[2],s=20,zorder=2)

pl.axis([-50,220,8,-13])

pl.xlabel('$\mathrm{EW}_{\mathrm{Ly}\\alpha}$ $[\AA]$',size=24)
pl.ylabel('$\mathrm{EW}_\mathrm{LIS}$ $[\AA]$',size=24,labelpad=-5)

ax1=pl.gca()
ax1.xaxis.set_major_locator(MultipleLocator(50))
ax1.xaxis.set_minor_locator(MultipleLocator(10))
ax1.yaxis.set_major_locator(MultipleLocator(5))
ax1.yaxis.set_minor_locator(MultipleLocator(1))
ax1.tick_params('both',which='major',length=5)
ax1.tick_params('both',which='minor',length=3)
pl.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.17)
pl.savefig('plots/2017/lis_vs_lya_yc_bw.pdf')

## redo with color-coding by O3 ratio

pl.figure(8)
pl.clf()

idx=lis_idx_good[lbg_lmidx]&(kbss_fhb[kbss_lmidx]>kbss_ehb[kbss_lmidx])&(kbss_fo3_5008[kbss_lmidx]>kbss_eo3_5008[kbss_lmidx])

pl.axvline(0,c='k',ls='--',zorder=-1,lw=1)
pl.axhline(0,c='k',ls='--',zorder=-1,lw=1)

pl.errorbar(lbg_seqw[lbg_lmidx][idx],-lbg_ew_lis[lbg_lmidx][idx],xerr=lbg_seqwerr[lbg_lmidx][idx],yerr=lbg_ew_lis_err[lbg_lmidx][idx],ls='none',zorder=1,c='0.5',lw=1)

pl.scatter(lbg_seqw[lbg_lmidx][idx],-lbg_ew_lis[lbg_lmidx][idx],c=np.log10(kbss_fo3_5008/kbss_fhb)[kbss_lmidx][idx],cmap=viridis,s=30,vmin=-0.2,vmax=1,zorder=2)

pl.axis([-50,220,8,-13])

cb=pl.colorbar(extend='both')
cb.solids.set_edgecolor('face')
cb.set_label('$\mathrm{O3}$',fontsize=20,rotation=270,labelpad=15)
cb.set_ticks(np.arange(-0.2,1.1,0.2))

pl.xlabel('$\mathrm{EW}_{\mathrm{Ly}\\alpha}$ $[\AA]$',size=24)
pl.ylabel('$\mathrm{EW}_\mathrm{LIS}$ $[\AA]$',size=24,labelpad=-5)

ax1=pl.gca()
ax1.xaxis.set_major_locator(MultipleLocator(50))
ax1.xaxis.set_minor_locator(MultipleLocator(10))
ax1.yaxis.set_major_locator(MultipleLocator(5))
ax1.yaxis.set_minor_locator(MultipleLocator(1))
ax1.tick_params('both',which='major',length=5)
ax1.tick_params('both',which='minor',length=3)
pl.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.17)
pl.savefig('plots/2017/lis_vs_lya_yc_o3.pdf')

nstat=len(lbg_seqw[lbg_lmidx][idx])
rstat,pstat=stats.spearmanr(lbg_seqw[lbg_lmidx][idx],-lbg_ew_lis[lbg_lmidx][idx])
fout.write('$W_\mathrm{LIS}$ vs $W_{\mathrm{Ly}\\alpha}$ & %3d & %5.2f & %3.1f\\\\ \n' % (nstat,rstat,np.log10(pstat)))


## make inverse plot: Lya vs O3 with LIS colors


pl.figure(9)
pl.clf()

idx=lis_idx_good[lbg_lmidx]&(kbss_fhb[kbss_lmidx]>kbss_ehb[kbss_lmidx])&(kbss_fo3_5008[kbss_lmidx]>kbss_eo3_5008[kbss_lmidx])

pl.axvline(0,c='k',ls='--',zorder=-1,lw=1)

pl.errorbar(lbg_seqw[lbg_lmidx][idx],np.log10(kbss_fo3_5008/kbss_fhb)[kbss_lmidx][idx],xerr=lbg_seqwerr[lbg_lmidx][idx],yerr=np.sqrt((kbss_eo3_5008/kbss_fo3_5008)**2+(kbss_ehb/kbss_fhb)**2)[kbss_lmidx][idx],ls='none',zorder=1,c='0.5',lw=1)

pl.scatter(lbg_seqw[lbg_lmidx][idx],np.log10(kbss_fo3_5008/kbss_fhb)[kbss_lmidx][idx],c=-lbg_ew_lis[lbg_lmidx][idx],cmap=viridis,s=30,vmin=-5,vmax=0,zorder=2)

pl.axis([-50,220,-1,2])

cb=pl.colorbar(extend='both')
cb.solids.set_edgecolor('face')
cb.set_label('$\mathrm{EW}_\mathrm{LIS}$ $[\AA]$',fontsize=20,rotation=270,labelpad=30)
cb.set_ticks(np.arange(-5,0.1,1))

pl.xlabel('$\mathrm{EW}_{\mathrm{Ly}\\alpha}$ $[\AA]$',size=24)
pl.ylabel('$\mathrm{O3}$',size=24,labelpad=-5)

ax1=pl.gca()
ax1.xaxis.set_major_locator(MultipleLocator(50))
ax1.xaxis.set_minor_locator(MultipleLocator(10))
ax1.yaxis.set_major_locator(MultipleLocator(0.5))
ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
ax1.tick_params('both',which='major',length=5)
ax1.tick_params('both',which='minor',length=3)
pl.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.17)
pl.savefig('plots/2017/o3_vs_lya_yc_lis.pdf')

nstat=len(lbg_seqw[lbg_lmidx][idx])
rstat,pstat=stats.spearmanr(lbg_seqw[lbg_lmidx][idx],np.log10(kbss_fo3_5008/kbss_fhb)[kbss_lmidx][idx])
fout.write('$W_{\mathrm{Ly}\\alpha}$ vs O3 & %3d & %5.2f & %3.1f\\\\ \n' % (nstat,rstat,np.log10(pstat)))

## make third version: LIS vs O3

pl.figure(10)
pl.clf()

idx=lis_idx_good[lbg_lmidx]&(kbss_fhb[kbss_lmidx]>kbss_ehb[kbss_lmidx])&(kbss_fo3_5008[kbss_lmidx]>kbss_eo3_5008[kbss_lmidx])

lbg_ew_lis_best=lbg_ew_lis
sig_idx=lbg_sname=='Q1442-BX333'
lbg_ew_lis_best[sig_idx]=lbg_ew_lis_sig[sig_idx]

#pl.axvline(0,c='k',ls='--',zorder=-1,lw=1)


pl.errorbar(-lbg_ew_lis_best[lbg_lmidx][idx],np.log10(kbss_fo3_5008/kbss_fhb)[kbss_lmidx][idx],xerr=lbg_ew_lis_err[lbg_lmidx][idx],yerr=np.sqrt((kbss_eo3_5008/kbss_fo3_5008)**2+(kbss_ehb/kbss_fhb)**2)[kbss_lmidx][idx],ls='none',zorder=1,c='0.5',lw=1)

sort_lya_idx=np.argsort(lbg_seqw[lbg_lmidx][idx])
pl.scatter(-lbg_ew_lis_best[lbg_lmidx][idx][sort_lya_idx],np.log10(kbss_fo3_5008/kbss_fhb)[kbss_lmidx][idx][sort_lya_idx],c=lbg_seqw[lbg_lmidx][idx][sort_lya_idx],cmap=viridis,s=30,vmin=-25,vmax=50,zorder=2)

#pl.axis([8,-13,-1,2])
#pl.axis([8,-8,-1,2])
pl.axis([4,-7,-1,2])

cb=pl.colorbar(extend='both')
cb.solids.set_edgecolor('face')
cb.set_label('$\mathrm{EW}_{\mathrm{Ly}\\alpha}$ $[\AA]$',fontsize=20,rotation=270,labelpad=20)
cb.set_ticks(np.arange(-25,51,25))

pl.xlabel('$\mathrm{EW}_\mathrm{LIS}$ $[\AA]$',size=24)
pl.ylabel('$\mathrm{O3}$',size=24,labelpad=-5)

ax1=pl.gca()
ax1.xaxis.set_major_locator(MultipleLocator(2))
ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
ax1.yaxis.set_major_locator(MultipleLocator(0.5))
ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
ax1.tick_params('both',which='major',length=5)
ax1.tick_params('both',which='minor',length=3)
pl.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.17)
pl.savefig('plots/2017/o3_vs_lis_yc_lya.pdf')

nstat=idx.sum()
rstat,pstat=stats.spearmanr(-lbg_ew_lis[lbg_lmidx][idx],np.log10(kbss_fo3_5008/kbss_fhb)[kbss_lmidx][idx])
fout.write('$W_\mathrm{LIS}$ vs O3 & %3d & %5.2f & %3.1f\\\\ \n' % (nstat,rstat,np.log10(pstat)))


## Measure EW_LIS vs v_lya

pl.figure(11)
pl.clf()
pl.errorbar(3e5*(lbg_wlya[lis_idx]-1215.67)/1215.67,lbg_ew_lis[lis_idx],xerr=np.ones(lis_idx.sum()),yerr=lbg_ew_lis_err[lis_idx],ls='none',zorder=1,c='0.5')

pl.scatter(3e5*(lbg_wlya[lis_idx]-1215.67)/1215.67,lbg_ew_lis[lis_idx],c=np.log10(lbg_lis_cont/lbg_lis_cont_err)[lis_idx],cmap=viridis,vmin=1,vmax=2.5,s=30,zorder=2)

pl.axis([-5000,5000,8,-13])
#pl.rc('font',**{'family':'serif','serif':'Computer Modern Roman'})
#pl.rc('text',usetex=True)
#pl.xlabel('$\mathrm{EW}_{\mathrm{Ly}\\alpha}$ $[\AA]$',size=24)
#pl.ylabel('$\mathrm{EW}_\mathrm{LIS}$ $[\AA]$',size=24,labelpad=-5)
pl.xlabel(r'$\mathrm{EW}_{\mathrm{Ly}\alpha}$ $[\AA]$',size=24)
pl.ylabel(r'$\mathrm{EW}_\mathrm{LIS}$ $[\AA]$',size=24,labelpad=0)

ax1=pl.gca()
#ax1.xaxis.set_major_locator(MultipleLocator(50))
#ax1.xaxis.set_minor_locator(MultipleLocator(10))
#ax1.yaxis.set_major_locator(MultipleLocator(5))
#ax1.yaxis.set_minor_locator(MultipleLocator(1))
ax1.tick_params('both',which='major',length=5)
ax1.tick_params('both',which='minor',length=3)
pl.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.17)
pl.savefig('plots/2017/lis_vs_vlya_yc.pdf')
#pl.rc('text',usetex=False)

nstat=len(lbg_wlya[lis_idx])
rstat,pstat=stats.spearmanr(3e5*(lbg_wlya[lis_idx]-1215.67)/1215.67,lbg_ew_lis[lis_idx])
fout.write('$v_{\mathrm{Ly}\\alpha}$ vs $W_\mathrm{LIS}$ & %3d & %5.2f & %3.1f\\\\ \n' % (nstat,rstat,np.log10(pstat)))


# Fesc vs LIS
pl.figure(5)
pl.clf()
fesc_idx=(kbss_fha[kbss_lmidx]>0)&(lbg_fflux[lbg_lmidx]>0)
idx=fesc_idx&(lis_idx[lbg_lmidx])
pl.semilogy(lbg_ew_lis[lbg_lmidx][idx],(lbg_fflux[lbg_lmidx]/kbss_fha[kbss_lmidx]/8.7)[idx],'.')
#pl.plot(lae_eqw,lae_lyaha/8.7,'rs')
#pl.xticks([0.1,1,10,100,1000],[0.1,1,10,100,1000])
#pl.yticks([1e-3,1e-2,1e-1,1],[1e-3,1e-2,1e-1,1])
pl.xlabel(r'$\mathrm{EW}_\mathrm{LIS}$ $[\AA]$',size=24)
pl.ylabel('$f_{\mathrm{esc,Ly}\\alpha}$',size=24,labelpad=10)
ax1=pl.gca()
ax1.xaxis.set_major_locator(MultipleLocator(1))
ax1.xaxis.set_minor_locator(MultipleLocator(0.25))
#ax1.yaxis.set_major_locator(LogLocator(base=10.))
ax1.tick_params('both',which='major',length=5)
ax1.tick_params('both',which='minor',length=3)
pl.axis([-2,5.5,2e-4,3])
pl.subplots_adjust(left=0.16,right=0.97,top=0.95,bottom=0.17)
pl.savefig('plots/2017/fesc_vs_lis_yc.pdf')

nstat=idx.sum()
rstat,pstat=stats.spearmanr(lbg_ew_lis[lbg_lmidx][idx],(lbg_fflux[lbg_lmidx]/kbss_fha[kbss_lmidx]/8.7)[idx])
fout.write('$W_\mathrm{LIS}$ vs $f_\mathrm{out}$ & %3d & %5.2f & %3.1f\\\\ \n' % (nstat,rstat,np.log10(pstat)))

fout.close()

# Calculate fraction with seqw>20 vs R mag
rmag_order_idx=np.argsort(kbss_rmag[kbss_lmidx])

nsamp=25
rmag_med=np.empty(len(kbss_lmidx)-nsamp)
rmag_laefrac1=np.empty(len(kbss_lmidx)-nsamp)
rmag_laefrac2=np.empty(len(kbss_lmidx)-nsamp)
rmag_laefrac3=np.empty(len(kbss_lmidx)-nsamp)
for ii in range(len(kbss_lmidx)-nsamp):
    rmag_med[ii]=np.median(kbss_rmag[kbss_lmidx][rmag_order_idx][ii:ii+nsamp])
    rmag_laefrac1[ii]=np.sum(lbg_seqw[lbg_lmidx][rmag_order_idx][ii:ii+nsamp]>20)/np.float(nsamp)
    rmag_laefrac2[ii]=np.sum(lbg_seqw[lbg_lmidx][rmag_order_idx][ii:ii+nsamp]>10)/np.float(nsamp)
    rmag_laefrac3[ii]=np.sum(lbg_seqw[lbg_lmidx][rmag_order_idx][ii:ii+nsamp]>0)/np.float(nsamp)

pl.figure()
pl.clf()
pl.plot(rmag_med,100*rmag_laefrac3,'k-')
pl.plot(rmag_med,100*rmag_laefrac2,'b-')
pl.plot(rmag_med,100*rmag_laefrac1,'r-')
pl.xlabel('R mag')
pl.ylabel('% LAEs')
pl.text(23,60,'$>0\AA$',color='k')
pl.text(23,52,'$>10\AA$',color='b')
pl.text(23,44,'$>20\AA$',color='r')
pl.subplots_adjust(bottom=0.15,top=0.95,right=0.95)


## Make ascii list of galaxies with M_UV and W(Lya)

cosmo=cosm.FlatLambdaCDM(H0=70,Om0=0.3)
kbss_dlum=cosmo.luminosity_distance(kbss_zneb).cgs.value
kbss_fg=(1450*(1+kbss_zneb)-4731)/(6417-4731)
kbss_fg=np.maximum(kbss_fg,0)
kbss_fg=np.minimum(kbss_fg,1)
kbss_m1450=kbss_gmag*(1-kbss_fg)+kbss_rmag*kbss_fg
kbss_muv=kbss_m1450-5*np.log10(kbss_dlum/(10*3.086e18))+2.5*np.log10(1+kbss_zneb) # plus sign for final term because it's energy per unit Hz, not wavelength!
#kbss_sfr=4*np.pi*kbss_dlum**2*1e-17/1.26e41*kbss_fha/kbss_ha_ext

f1=open('lbg_name_muv_wlya.dat','w')
for ii in range(len(kbss_lmidx)):
    f1.write('%s %5.2f %5.1f\n' % (lbg_sname[lbg_lmidx][ii],kbss_muv[kbss_lmidx][ii],lbg_seqw[lbg_lmidx][ii]))
f1.close()

f1=open('lbg_name_muv_wlya_z.dat','w')
for ii in range(len(kbss_lmidx)):
    f1.write('%s %5.2f %5.1f %6.4f\n' % (lbg_sname[lbg_lmidx][ii],kbss_muv[kbss_lmidx][ii],lbg_seqw[lbg_lmidx][ii],kbss_zneb[kbss_lmidx][ii]))
f1.close()
