import numpy as np
import statsmodels.api as sm
from scipy import stats
from multi_regress import read_mls, read_traj,read_reanalysis, anom_calc, multi_reg, read_qbo_anom
import pickle
from pandas.core import datetools

ym_lim = [200408,201612]            # time range
time_lim = [round(int(x/100)+(x%100-0.5)/12.,3) for x in ym_lim]
lat_lim_mls = 30                      # MLS latitude range
lat_lim_traj = 30                     # trajectory model simulation latitude range
lat_lim = 30                          # tropical latitude range for regressors
pr_lim = 100                          # water vapor pressure level
pr_hr = 82.54                         # heating rate pressure level for BDC
prT = 500                             # troposperic temperature pressure level

# set lags for regressors T, BDC(hr), QBO
if pr_lim == 82.54:
    lag = [1, 1, 3]
else:
    lag = [0, 0, 2]

####  choose trajectory runs: 1 for standard run; 2 for s140 run
run = 1
if run == 1:
    run_str = 'std'
if run == 2:
    run_str = 's140'
qbo = True                       # choose without qbo index in the regression

fdir       ='/ice3/hao/TTL/code&data_ACP2018/data/'             # dir of input files
outdir     ='/ice3/hao/TTL/code&data_ACP2018/data/'             # dir of output files

## set directory of files
mls_file     = fdir+'MLS-Aura_L2GP-H2O_v04_2004_2016.nc'
if run == 1:
    traj_erai_file     = fdir+'isob_traj_s100_i370_ERAi_200408_201612_6hrly_forward_mls_ak.nc'
    traj_mer2_file     = fdir+'isob_traj_s100_i370_merra2_200408_2016_6hrly_forward_mls_ak.nc'
if run == 2:
    traj_erai_file     = fdir+'isob_traj_s140_i370_ERAi_200408_201612_6hrly_forward_mls_ak_0N.nc'
    traj_mer2_file     = fdir+'isob_traj_s140_i370_merra2_200408_201605_6hrly_forward_mls_ak_0N.nc'

## read MLS and trajectory model water vapor
h2o_mls  = read_mls(time_lim,lat_lim_mls,pr_lim,mls_file,'h2o_mix')
h2o_erai = read_traj(time_lim,lat_lim_traj,pr_lim,traj_erai_file,'isob_h2oM')
h2o_mer2 = read_traj(time_lim,lat_lim_traj,pr_lim,traj_mer2_file,'isob_h2oM')

## calculate water vapor anomaly for each grid, weighting by latitude.
def anom_grid_calc(h2o,nt,lat):
    nlat = len(lat)
    h2o_anom = np.zeros(h2o.shape)
    for i in range(nt):
        h2o_anom[i,:,:] = h2o[i,:,:]-np.mean(h2o[i%12::12,:,:],axis=0)

    for j in range(nlat):
        h2o_anom[:,j,:]   = h2o_anom[:,j,:]*np.cos(lat[j]*np.pi/180)/(np.mean(np.cos(lat*np.pi/180)))  # weighted by latitude
    return h2o_anom

h2o_mls_anom  = anom_grid_calc(h2o_mls['value'],h2o_mls['value'].shape[0],h2o_mls['lat'])
h2o_erai_anom = anom_grid_calc(h2o_erai['value'],h2o_erai['value'].shape[0],h2o_erai['lat'])
h2o_mer2_anom = anom_grid_calc(h2o_erai['value'],h2o_erai['value'].shape[0],h2o_erai['lat'])

[nt,nlat,nlon] = h2o_erai_anom.shape

## read qbo index for regressor
if qbo:
    qbo_file     = '/ice3/hao/TTL/qbo_u50_index.txt'
    qbo_anom = read_qbo_anom(qbo_file,ym_lim)

mer2_file = fdir+'MERRA2_day2mon_tavg3_dtdttot_t_Np_2000-201708.nc'
erai_file = fdir+'ECMWF_ERAi_monthly_t_hr_2000-2016_full_resolution.nc'

## read BDC and dT regressors from MERRA-2 and ERAi
mer2_hr          = read_reanalysis(mer2_file,time_lim,lat_lim,pr_hr,'isob_dtdttot')
mer2_hr['value'] = mer2_hr['value']*(1000./pr_hr)**(2./7.)
mer2_hr_anom     = anom_calc(mer2_hr)['anom']
mer2_tem         = read_reanalysis(mer2_file,time_lim,lat_lim,prT,'isob_t')
mer2_tem_anom    = anom_calc(mer2_tem)['anom']

erai_hr          = read_reanalysis(erai_file,time_lim,lat_lim,pr_hr,'isob_dtdttot')
erai_hr['value'] = erai_hr['value']*(1000./pr_hr)**(2./7.)
erai_hr_anom     = anom_calc(erai_hr)['anom']
erai_tem         = read_reanalysis(erai_file,time_lim,lat_lim,prT,'isob_t')
erai_tem_anom    = anom_calc(erai_tem)['anom']


# set regressors dict
if qbo:
    nvar  = 3
    x_mer2  = np.column_stack((mer2_tem_anom,mer2_hr_anom,qbo_anom['stdAnom']))
    x_erai  = np.column_stack((erai_tem_anom,erai_hr_anom,qbo_anom['stdAnom']))
    ntout = nt-max(lag)
else:
    nvar  = 2
    x_mer2 = np.column_stack((mer2_tem_anom,mer2_hr_anom))
    x_erai = np.column_stack((erai_tem_anom,erai_hr_anom))
    ntout = nt-max(lag[:-1])

## multi-regression for MLS and trajectory water vapor with regressors from ERAi
traj_coefs_erai   = np.zeros([nvar+1,nlat,nlon])
traj_reg_erai     = np.zeros([nvar+1,nlat,nlon])
traj_r2_erai      = np.zeros([nlat,nlon])
traj_y_erai       = np.zeros([ntout,nlat,nlon])
traj_yfit_erai    = np.zeros([ntout,nlat,nlon])
traj_ypart_erai   = np.zeros([nvar,ntout,nlat,nlon])
traj_pval_erai    = np.zeros([nvar+1,nlat,nlon])
traj_conf95_erai  = np.zeros([nvar+1,nlat,nlon])

mls_coefs_erai    = np.zeros([nvar+1,nlat,nlon])
mls_reg_erai      = np.zeros([nvar+1,nlat,nlon])
mls_r2_erai       = np.zeros([nlat,nlon])
mls_y_erai        = np.zeros([ntout,nlat,nlon])
mls_yfit_erai     = np.zeros([ntout,nlat,nlon])
mls_ypart_erai    = np.zeros([nvar,ntout,nlat,nlon])
mls_pval_erai     = np.zeros([nvar+1,nlat,nlon])
mls_conf95_erai   = np.zeros([nvar+1,nlat,nlon])

for ilat in range(nlat):
    for ilon in range(nlon):
        reg = multi_reg(h2o_erai_anom[:,ilat,ilon],x_erai,lag)
        traj_coefs_erai[:,ilat,ilon]    = reg.b
        traj_y_erai[:,ilat,ilon]        = reg.y
        traj_yfit_erai[:,ilat,ilon]     = reg.yhat
        traj_ypart_erai[:,:,ilat,ilon]  = reg.yhat_part.T
        traj_r2_erai[ilat,ilon]         = reg.R2adj
        traj_pval_erai[:,ilat,ilon]     = reg.p
        traj_conf95_erai[:,ilat,ilon]   = reg.conf95

        reg = multi_reg(h2o_mls_anom[:,ilat,ilon],x_erai,lag)
        mls_coefs_erai[:,ilat,ilon]     = reg.b
        mls_y_erai[:,ilat,ilon]         = reg.y
        mls_yfit_erai[:,ilat,ilon]      = reg.yhat
        mls_ypart_erai[:,:,ilat,ilon]   = reg.yhat_part.T
        mls_r2_erai[ilat,ilon]          = reg.R2adj
        mls_pval_erai[:,ilat,ilon]      = reg.p
        mls_conf95_erai[:,ilat,ilon]    = reg.conf95

## multi-regression for MLS and trajectory water vapor with regressors from MERRA-2
traj_coefs_mer2   = np.zeros([nvar+1,nlat,nlon])
traj_reg_mer2     = np.zeros([nvar+1,nlat,nlon])
traj_r2_mer2      = np.zeros([nlat,nlon])
traj_y_mer2       = np.zeros([ntout,nlat,nlon])
traj_yfit_mer2    = np.zeros([ntout,nlat,nlon])
traj_ypart_mer2   = np.zeros([nvar,ntout,nlat,nlon])
traj_pval_mer2    = np.zeros([nvar+1,nlat,nlon])
traj_conf95_mer2  = np.zeros([nvar+1,nlat,nlon])

mls_coefs_mer2   = np.zeros([nvar+1,nlat,nlon])
mls_reg_mer2     = np.zeros([nvar+1,nlat,nlon])
mls_r2_mer2      = np.zeros([nlat,nlon])
mls_y_mer2       = np.zeros([ntout,nlat,nlon])
mls_yfit_mer2    = np.zeros([ntout,nlat,nlon])
mls_ypart_mer2   = np.zeros([nvar,ntout,nlat,nlon])
mls_pval_mer2    = np.zeros([nvar+1,nlat,nlon])
mls_conf95_mer2  = np.zeros([nvar+1,nlat,nlon])

for ilat in range(nlat):
    for ilon in range(nlon):
        reg = multi_reg(h2o_mer2_anom[:,ilat,ilon],x_mer2,lag)
        traj_coefs_mer2[:,ilat,ilon]     = reg.b
        traj_y_mer2[:,ilat,ilon]         = reg.y
        traj_yfit_mer2[:,ilat,ilon]      = reg.yhat
        traj_ypart_mer2[:,:,ilat,ilon]   = reg.yhat_part.T
        traj_r2_mer2[ilat,ilon]          = reg.R2adj
        traj_pval_mer2[:,ilat,ilon]      = reg.p
        traj_conf95_mer2[:,ilat,ilon]    = reg.conf95

        reg = multi_reg(h2o_mls_anom[:,ilat,ilon],x_mer2,lag)
        mls_coefs_mer2[:,ilat,ilon]      = reg.b
        mls_y_mer2[:,ilat,ilon]          = reg.y
        mls_yfit_mer2[:,ilat,ilon]       = reg.yhat
        mls_ypart_mer2[:,:,ilat,ilon]    = reg.yhat_part.T
        mls_r2_mer2[ilat,ilon]           = reg.R2adj
        mls_pval_mer2[:,ilat,ilon]       = reg.p
        mls_conf95_mer2[:,ilat,ilon]     = reg.conf95

## pack up the data for plotting later
erai = dict(lon=h2o_erai['lon'], lat=h2o_erai['lat'],nvar=nvar, \
            erai_coefs=traj_coefs_erai,erai_y=traj_y_erai,erai_yfit=traj_yfit_erai,erai_ypart=traj_ypart_erai,\
            erai_r2=traj_r2_erai,erai_pval=traj_pval_erai,erai_conf95=traj_conf95_erai,\
            mls_coefs =mls_coefs_erai, mls_y =mls_y_erai, mls_yfit =mls_yfit_erai, mls_ypart =mls_ypart_erai, \
            mls_r2 =mls_r2_erai, mls_pval=mls_pval_erai,mls_conf95=mls_conf95_erai)
mer2 = dict(lon=h2o_mer2['lon'], lat=h2o_mer2['lat'],nvar=nvar, \
            mer2_coefs=traj_coefs_mer2,mer2_y=traj_y_mer2,mer2_yfit=traj_yfit_mer2,mer2_ypart=traj_ypart_mer2,\
            mer2_r2=traj_r2_mer2,mer2_pval=traj_pval_mer2,mer2_conf95=traj_conf95_mer2,\
            mls_coefs =mls_coefs_mer2, mls_y =mls_y_mer2, mls_yfit =mls_yfit_mer2, mls_ypart =mls_ypart_mer2, \
            mls_r2 =mls_r2_mer2, mls_pval=mls_pval_mer2,mls_conf95=mls_conf95_mer2)
data = dict(erai=erai,mer2=mer2)

## save coefficients into files
if  qbo:
    with open(outdir+'reg_mls_traj_coefs_'+run_str+'_'+str(int(pr_lim))+'hpa_'+str(time_lim[0])+'_'+str(time_lim[1])+'_qbo.dat', 'wb') as f:
        pickle.dump(data,f)
else:
    with open(outdir+'reg_mls_traj_coefs_'+run_str+'_'+str(int(pr_lim))+'hpa_'+str(time_lim[0])+'_'+str(time_lim[1])+'.dat', 'wb') as f:
        pickle.dump(data,f)
