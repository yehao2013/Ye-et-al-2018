import numpy as np
import statsmodels.api as sm
from scipy import stats
from multi_regress import read_geosccm, read_traj,read_reanalysis, anom_calc, multi_reg
import pickle
from pandas.core import datetools


ym_lim = [200501,201612]            # time range
time_lim = [round(int(x/100)+(x%100-0.5)/12.,3) for x in ym_lim]
lat_lim_ccm = 30                      # ccm latitude range
lat_lim_traj = 30                     # trajectory model simulation latitude range
lat_lim = 30                          # tropical latitude range for regressors
pr_lim = 100                          # water vapor pressure level
pr_hr = 85.4                          # heating rate pressure level for BDC
prT = 500                             # troposperic temperature pressure level

# set lags for regressors T, BDC(hr), QBO
if pr_lim == 85.4:
    lag = [1, 1, 3]
else:
    lag = [0, 0, 2]
run = 'std'  #'std_anv'              # choose with or without ice trajectory run
qbo = False                          # choose without qbo index in the regression

fdir       ='/ice3/hao/TTL/code&data_ACP2018/data/'             # dir of input files
outdir     ='/ice3/hao/TTL/code&data_ACP2018/data/'             # dir of output files

# geosccm and trajectory simulations files
ccm_file     = fdir+'GEOSCCM_SCF2_5_P1dm2b_isob_q_2000-2100.nc'
if run == 'std_anv':
    traj_ccm_file     = fdir+'isob_170312_traj_s100_i370_P1dm2b_GEOSCCM_2004_2016_6hrly_new_anvil.nc'
if run == 'std':
    traj_ccm_file     = fdir+'isob_150907_traj_s100_i370_P1dm2b_GEOSCCM_2000_2099_6hrly_noanvil.nc'

## read geosccm water vapor, calculate water vapor anomaly for each grid, weighting by latitude.
h2o_ccm      = read_geosccm(ccm_file,time_lim,lat_lim_ccm,pr_lim,'q')

h2o_ccm_anom = np.zeros(h2o_ccm['value'].shape)
for i in range(h2o_ccm['value'].shape[0]):
    h2o_ccm_anom[i,:,:] = h2o_ccm['value'][i,:,:]-np.mean(h2o_ccm['value'][i%12::12,:,:],axis=0)

for j in range(h2o_ccm['value'].shape[1]):
    h2o_ccm_anom[:,j,:]   = h2o_ccm_anom[:,j,:]*np.cos(h2o_ccm['lat'][j]*np.pi/180)/(np.mean(np.cos(h2o_ccm['lat']*np.pi/180)))


## read trajectory model water vapor, calculate water vapor anomaly for each grid, weighting by latitude.
h2o_traj = read_traj(time_lim,lat_lim_traj,pr_lim,traj_ccm_file,'isob_h2oM')

h2o_traj_anom = np.zeros(h2o_traj['value'].shape)
for i in range(h2o_traj['value'].shape[0]):
    h2o_traj_anom[i,:,:] = h2o_traj['value'][i,:,:]-np.mean(h2o_traj['value'][i%12::12,:,:],axis=0)

for j in range(h2o_traj['value'].shape[1]):
    h2o_traj_anom[:,j,:]   = h2o_traj_anom[:,j,:]*np.cos(h2o_traj['lat'][j]*np.pi/180)/(np.mean(np.cos(h2o_traj['lat']*np.pi/180)))

nlat  = h2o_traj_anom.shape[1]  ;   nlon = h2o_traj_anom.shape[2]

## read qbo if used
if qbo:
    qbo_file     = fdir+'P1dm2b_GEOSCCM_qbo_2005-2016.dat'
    with open(qbo_file,'r') as f:
        qbo_anom = pickle.load(f)

## read regressors BDC and tropospheric temperature
ccm_file2 = fdir+'P1dm2b_GEOSCCM_monthly_2000_2099.nc'
ccm_hr          = read_reanalysis(ccm_file2,time_lim,lat_lim,pr_hr,'isob_dthetadt')
ccm_hr_anom     = anom_calc(ccm_hr)
ccm_tem         = read_reanalysis(ccm_file2,time_lim,lat_lim,prT,'isob_t')
ccm_tem_anom    = anom_calc(ccm_tem)

# set regressors dict
if qbo:
    nvar  = 3
    x_ccm = np.column_stack((ccm_tem_anom['anom'],ccm_hr_anom['anom'],qbo_anom))
    ntout = len(h2o_ccm['time'])-max(lag)
else:
    nvar  = 2
    lag   = lag[:-1]
    x_ccm = np.column_stack((ccm_tem_anom['anom'],ccm_hr_anom['anom']))
    ntout = len(h2o_ccm['time'])-max(lag)

## multi-regression for all regressors
traj_coefs_ccm  = np.zeros([nvar+1,nlat,nlon])
traj_reg_ccm    = np.zeros([nvar+1,nlat,nlon])
traj_r2_ccm     = np.zeros([nlat,nlon])
traj_y_ccm      = np.zeros([ntout,nlat,nlon])
traj_yfit_ccm   = np.zeros([ntout,nlat,nlon])
traj_ypart_ccm  = np.zeros([nvar,ntout,nlat,nlon])
traj_pval_ccm   = np.zeros([nvar+1,nlat,nlon])
traj_conf95_ccm = np.zeros([nvar+1,nlat,nlon])

ccm_coefs = np.zeros([nvar+1,nlat,nlon])
ccm_reg   = np.zeros([nvar+1,nlat,nlon])
ccm_r2    = np.zeros([nlat,nlon])
ccm_y     = np.zeros([ntout,nlat,nlon])
ccm_yfit  = np.zeros([ntout,nlat,nlon])
ccm_ypart = np.zeros([nvar,ntout,nlat,nlon])
ccm_pval  = np.zeros([nvar+1,nlat,nlon])
ccm_conf95= np.zeros([nvar+1,nlat,nlon])

## regress for each grid box
for ilat in range(nlat):
    for ilon in range(nlon):
        reg = multi_reg(h2o_traj_anom[:,ilat,ilon],x_ccm,lag)
        traj_coefs_ccm[:,ilat,ilon]   = reg.b
        traj_y_ccm[:,ilat,ilon]       = reg.y
        traj_yfit_ccm[:,ilat,ilon]    = reg.yhat
        traj_ypart_ccm[:,:,ilat,ilon] = reg.yhat_part.T
        traj_r2_ccm[ilat,ilon]        = reg.R2adj
        traj_pval_ccm[:,ilat,ilon]    = reg.p
        traj_conf95_ccm[:,ilat,ilon]  = reg.conf95

        reg = multi_reg(h2o_ccm_anom[:,ilat,ilon],x_ccm,lag)
        ccm_coefs[:,ilat,ilon]        = reg.b
        ccm_y[:,ilat,ilon]            = reg.y
        ccm_yfit[:,ilat,ilon]         = reg.yhat
        ccm_ypart[:,:,ilat,ilon]      = reg.yhat_part.T
        ccm_r2[ilat,ilon]             = reg.R2adj
        ccm_pval[:,ilat,ilon]         = reg.p
        ccm_conf95[:,ilat,ilon]       = reg.conf95

## pack up the coefficients into a dictionary
data = dict(lon=h2o_traj['lon'], lat=h2o_traj['lat'],nvar=nvar,\
            gcm_coefs=ccm_coefs,gcm_y=ccm_y,gcm_yfit=ccm_yfit,gcm_ypart=ccm_ypart,\
            gcm_r2=ccm_r2,gcm_pval=ccm_pval,gcm_conf95=ccm_conf95, \
            traj_coefs=traj_coefs_ccm,traj_y=traj_y_ccm,traj_yfit=traj_yfit_ccm,traj_ypart=traj_ypart_ccm,\
            traj_r2=traj_r2_ccm,traj_pval=traj_pval_ccm, traj_conf95=traj_conf95_ccm)

if  qbo:
    with open(outdir+'reg_geosccm_traj_coefs_'+run+'_'+str(int(pr_lim))+'hpa_'+str(time_lim[0])+'_'+str(time_lim[1])+'_qbo.dat', 'wb') as f:
        pickle.dump(data,f)
else:
    with open(outdir+'reg_geosccm_traj_coefs_'+run+'_'+str(int(pr_lim))+'hpa_'+str(time_lim[0])+'_'+str(time_lim[1])+'.dat', 'wb') as f:
        pickle.dump(data,f)
