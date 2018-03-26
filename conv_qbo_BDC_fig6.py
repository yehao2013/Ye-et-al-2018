'''
This program is to read convective cloud frequency, GEOSCCM convective ice water content,
and GEOSCCM convective ice evaporation rate and then to plot the scatter plots vs tropospheric temperature.
'''
from netCDF4 import Dataset
import numpy as np
from multi_regress import *
import matplotlib.pyplot as plt
from scipy import stats

fdir = '/ice3/hao/TTL/code&data_ACP2018/data/'
outdir = '/ice3/hao/TTL/code&data_ACP2018/figures/'

time_lim = [200501,201612]
t_lim = [int(x/100)+round((x%100-0.5)/12,3) for x in time_lim]

# set parameter limit
th_lim    = 390                                                                ## cloud frequency potential temperature
lat_lim   = 30                                                                 ## latitude limit (tropics)
prT = 500                                                                      ## tropospheric temperature pressure (500 hPa)
qpr_lim = 100                                                                  ## GEOSCCM convective ice pressure level (100 hPa)
pr_evap = 100                                                                  ## GEOSCCM convective ice evaporate rate pressure level (100 hPa)

## read tropospheric temperature anomalies from ERAi and cloud frequency at 390 K over the tropics
erai_file = fdir+'ECMWF_ERAi_monthly_t_hr_2000-2016_full_resolution.nc'
erai_tem   = read_reanalysis(erai_file,t_lim,lat_lim,prT,'isob_t')             ## read t500 from ERAi
t500_anom  = anom_calc(erai_tem)['anom']                                       ## calculate tropical average t500 anomalies

## read cloud frequency potential temperatures
conv_file = fdir+'freq_z_th_avg_2x2.nc'
f = Dataset(conv_file,'r')
latc     = f.variables['lat'][:]
lonc     = f.variables['lon'][:]
thf      = f.variables['th_freq'][:]
mon      = f.variables['time'][:]
mon      = np.array([round(int(x/100)+(x%100-0.5)/12,3) for x in mon])

## choose tropical values and then calculate tropical averages
lat_indc = np.where(abs(latc) <= lat_lim)[0]
latc     = latc[lat_indc]

th_ind = np.where(f.variables['theta'][:] == th_lim)[0]
thf = thf[:,:,lat_indc,:][:,th_ind,:,:].squeeze()

thf_lat  = np.average(thf.mean(axis=2),axis=1,weights= np.cos(latc*np.pi/180))
thf_anom = np.array([thf_lat[i]-thf_lat[i%12::12].mean() for i in range(len(thf_lat))])

## calculate the linear regression coefficients
x1 = t500_anom.copy()
x1.sort()
a1,b1 = stats.linregress(t500_anom,thf_anom/thf_avg.mean())[:2]
y1 = a1*x1+b1

##-------------------------------------------------------------------------------
## read GEOSCCM convective ice water content at 100 hPa
qi_file = fdir+'GEOSCCM_monthly_qian_2000-2020.nc'
f1      = Dataset(qi_file,'r')
qlat    = f1.variables['lat'][:]
qlon    = f1.variables['lon'][:]
qtime   = f1.variables['time'][:]
qpr     = f1.variables['pr'][:]

tind    = np.where((qtime >= t_lim[0]) & (qtime <= t_lim[1]))[0]
latind  = np.where(abs(qlat) <= lat_lim)[0]
qpr_ind = np.where(abs(qpr-qpr_lim) < 1)[0]

qtime = qtime[tind] ; qlat = qlat[latind]
qi    = f1.variables['qicn'][tind,:,:,:][:,:,latind,:][:,qpr_ind,:,:].squeeze()
f1.close()

# calculate tropical average and anomalies
qi_avg  = np.average(qi.mean(axis=2),axis=1,weights= np.cos(qlat*np.pi/180))
qi_anom = np.array([qi_avg[i]-qi_avg[i%12::12].mean() for i in range(len(qi_avg))])

## Read GEOSCCM temperature at 500 hPa and calculate anomalies
ccm_file      = fdir+'P1dm2b_GEOSCCM_monthly_2000_2099.nc'
ccm_tem       = read_reanalysis(ccm_file,t_lim,lat_lim,prT,'isob_t')
ccm_tem_anom  = anom_calc(geosccm_tem)['anom']

## calculate the linear regression coefficients
x2 = ccm_tem_anom.copy()
x2.sort()
a2,b2 = stats.linregress(ccm_tem_anom,qi_anom)[:2]
y2 = a2*x2+b2

##-------------------------------------------------------------------------------
## read GEOSCCM convective ice evaporation rate from trajectory model run
f1 = Dataset(fdir+'geosccm_s100_i370_2005-2016_6hrly_anvil_evap_rate.nc')
evp   = f1.variables['Ice_vp_mix'][:]*32                                      ## turn the amount into daily
elat  = f1.variables['Lat'][:]
elon  = f1.variables['Lon'][:]
epr   = f1.variables['Lev'][:]
etime = f1.variables['Time'][:]
f1.close()
etime = np.array([round(int(x/100)+(x%100-0.5)/12,3) for x in etime])

etimeind = np.where((etime >= t_lim[0]) & (etime <= t_lim[1]))[0]
eprind   = np.where(abs(epr-pr_evap) < 1)[0]
elatind  = np.where(abs(elat) <= lat_lim)[0]
evp      = evp[etimeind,:,:,:][:,:,elatind,:][:,eprind,:,:].squeeze()
evp[np.isnan(evp)] = 0

elat    = elat[elatind]

## calculate GEOSCCM evaporation rate anomalies
evp_avg = np.average(evp.mean(axis=2),axis=1,weights= np.cos(elat*np.pi/180))
evp_anom = np.array([evp_avg[i]-evp_avg[i%12::12].mean() for i in range(len(etime))])

## calculate the linear regression coefficients
x3 = ccm_tem_anom.copy()
x3.sort()
a3,b3 = stats.linregress(ccm_tem_anom,evp_anom)[:2]
y3 = a3*x3+b3

##-------------------------------------------------------------------------------
# plot figure 6

fig,ax = plt.subplots(2,2,figsize=[10,8])
plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=0.3, hspace=0.2)
ax[1,1].axis('off')
ax[0,0].scatter(t500_anom,thf_anom/thf_avg.mean())
ax[0,0].set_xlabel('ERAi T @ 500 hPa (K)',fontsize=10)
ax[0,0].set_ylabel('Relative convective cloud freq anomaly',fontsize=10)
ax[0,0].set_ylim(-2,4)
ax[0,0].set_xlim(-1,1.5)
ax[0,0].text(-0.9,3.3,'(a) Y = '+format(a1,'.1f')+'*X+'+format(b1,'.1e'),fontsize=10)
ax[0,0].plot(x1,y1)

ax[0,1].scatter(ccm_tem_anom,qi_anom)
ax[0,1].set_xlabel('GEOSCCM T @ 500 hPa (K)',fontsize=10)
ax[0,1].set_ylabel('GEOSCCM convective IWC anomaly (ppmv)',fontsize=10)
ax[0,1].set_ylim(-0.2,0.3)
ax[0,1].set_yticks([-0.2,-0.1,0,0.1,0.2,0.3])
ax[0,1].set_xlim(-1,1)
ax[0,1].set_xticks([-1,-0.5,0,0.5,1])
ax[0,1].text(-0.9,0.25,'(b) Y = '+format(a2,'.3f')+'*X+'+format(b2,'.1e'),fontsize=10)
ax[0,1].plot(x2,y2)


ax[1,0].scatter(ccm_tem_anom,evp_anom)
ax[1,0].set_xlabel('GEOSCCM T @ 500 hPa (K)',fontsize=10)
ax[1,0].set_ylabel('Evaporation rate anomaly (ppmv day$^{-1}$)',fontsize=10)
ax[1,0].set_ylim(-0.08,0.12)
ax[1,0].set_yticks([-0.08,-0.04,0,0.04,0.08,0.12])
ax[1,0].set_xlim(-1,1)
ax[1,0].set_xticks([-1,-0.5,0,0.5,1])
ax[1,0].text(-0.9,0.1,'(c) Y = '+format(a3,'.3f')+'*X+'+format(b3,'.1e'),fontsize=10)
ax[1,0].plot(x3,y3)

plt.savefig(outdir+'fig6.pdf')
