from netCDF4 import Dataset
import numpy as np
from lonto360 import lonto360,lonrearrange
from scipy.io import readsav

from scipy import stats,c_,ones,diff,dot
from scipy.linalg import inv
from numpy import log, pi, sqrt, square, diagonal

class multi_reg:
    '''
    Author: HAO YE
    Email: yehaolzu@gmail.com
    Last modified: Sep 4th 2017
    '''

    def __init__(self,y,x,lag):
        '''
        Initializing the regression class.
        '''

        self.y = y
        self.x = c_[ones(x.shape[0]),x]             #add one column with ones for the constant in the regression
        self.lag = np.append(max(lag),lag)

        #Estimate model with multi_regression
        self.lag_process()
        self.regress_auto()

    def lag_process(self):

        ncoef = self.x.shape[1]
        nobs  = self.y.shape[0]
        nlag  = int(max(self.lag))

        ## prepare x,y with consideration of lags
        self.y = self.y[nlag:]
        self.x = np.array([self.x[(nlag-self.lag[i]):(nobs-self.lag[i]),i] for i in range(ncoef)]).T

#    def regress(self):
#        # estimate coefficients and basic stats
#        self.x=np.array(self.x)
#        self.inv_xx = inv(np.array(dot(self.x.T,self.x)))
#        xy = dot(self.x.T,self.y)
#        self.b = dot(self.inv_xx,xy)                            # regression coefficients
#
#        self.nobs = self.y.shape[0]                             # number of observations
#        self.ncoef = self.x.shape[1]                            # number of coefficients
#        self.df_e  = self.nobs - self.ncoef                     # degrees of freedom, error
#        self.df_r  = self.ncoef - 1                             # degrees of freedom, regression
#
#        self.e = self.y - dot(self.x,self.b)                    # residuals
#        self.yhat = dot(self.x,self.b)                          # y value predicted by regression
#        self.sse = dot(self.e,self.e)/self.df_e                 # estimate for sigma^2, standard error of regression
#        self.se = sqrt(diagonal(self.sse*self.inv_xx))          # coefficient standard errors
#
#        self.t = self.b/self.se                                 # coefficient t-statistics
#        self.p = (1-stats.t.cdf(abs(self.t), self.df_e)) * 2    # coefficient p-values
#
#        self.R2 = 1 - self.e.var()/self.y.var()                 # model R-squared
#        self.R2adj = 1-(1-self.R2)*((self.nobs-1)/(self.nobs-self.ncoef))   # adjusted R-square
#
#        self.F = (self.R2/self.df_r) / ((1-self.R2)/self.df_e)  # model F-statistic
#        self.Fpv = 1-stats.f.cdf(self.F, self.df_r, self.df_e)  # F-statistic p-value
#
#        # conf95 is the 95% confidence interval based on the adjusted number of DOF
#        self.conf95 = stats.t(self.df_e-1).isf(0.025)*self.se
#        self.conf90 = stats.t(self.df_e-1).isf(0.05)*self.se
#        self.conf68 = stats.t(self.df_e-1).isf(0.16)*self.se


    def regress_auto(self):     # statistical adjusting for autocorrelation in residual
        self.x = np.array(self.x)
        self.inv_xx = inv(np.array(dot(self.x.T,self.x)))
        xy = dot(self.x.T,self.y)
        self.b = dot(self.inv_xx,xy)                            # estimate coefficients
        self.e = self.y - dot(self.x,self.b)                    # residuals

        self.nobs = self.y.shape[0]                             # number of observations
        self.ncoef = self.x.shape[1]                            # number of coefficients

        # Based on Santer et al. (2000), calculate adjusted degrees of freedom
        # adjx = factor that reduces # of obs. to # of independent obs.
        lag1cov = lambda a1: (np.average(a1[1:]*a1[:-1])/np.var(a1))    # lagged covariance routines
        adjx = lag1cov(self.e)
        adjx = (1-adjx)/(1+adjx)

        self.df_e = self.nobs*adjx - self.ncoef                 # degrees of freedom
        self.df_r = self.ncoef - 1                              # degrees of freedom, regression
        self.yhat = dot(self.x,self.b)                          # y value predicted by regression

        self.yhat_part = np.zeros([self.nobs,self.ncoef-1])
        for i in range(self.ncoef-1):
            self.yhat_part[:,i] = self.b[0]*self.x[:,0] + self.b[i+1]*self.x[:,i+1]

        self.sse = dot(self.e,self.e)/self.df_e                 # estimate for sigma^2, standard error of regression
        self.se = sqrt(diagonal(self.sse*self.inv_xx))          # coefficient standard errors

        self.t = self.b/self.se                                 # coefficient t-statistics
        self.p = (1-stats.t.cdf(abs(self.t), self.df_e)) * 2    # coefficient p-values

        self.R2 = 1 - self.e.var()/self.y.var()                 # model R-squared
        self.R2adj = 1-(1-self.R2)*((self.nobs-1)/(self.nobs-self.ncoef))   # adjusted R-square

        self.F = (self.R2/self.df_r) / ((1-self.R2)/self.df_e)  # model F-statistic
        self.Fpv = 1-stats.f.cdf(self.F, self.df_r, self.df_e)  # F-statistic p-value

        # conf95 is the 95% confidence interval based on the adjusted number of DOF
        self.conf95 = stats.t(self.df_e-1).isf(0.025)*self.se
        self.conf90 = stats.t(self.df_e-1).isf(0.05)*self.se
        self.conf68 = stats.t(self.df_e-1).isf(0.16)*self.se


## read trajectory simulations
def read_traj(time_lim,lat_lim,pr_lim,indir,var):
    fin  = Dataset(indir)
    var1 = fin.variables[var][:]
    lat  = fin.variables['Latitude'][:]
    lon  = fin.variables['Longitude'][:]
    pr   = fin.variables['Pressure'][:]
    time   = fin.variables['Time'][:]
    fin.close()

    ## if time is in yyyymm format, turn it into year format
    if int(time[0]/100) > 1000:
        time = np.array([round(int(x/100)+(x%100-0.5)/12.,3) for x in time])
    else:
        time = np.array([round(x,3) for x in time])
    time_ind = np.where((time>=time_lim[0]) & (time<=time_lim[1]))[0]        # choose data in time range
    time = time[time_ind]
    lat_ind = np.where(abs(lat) <= lat_lim)[0]                   # choose data in latitude range
    lat = lat[lat_ind]
    pr_ind = np.where(abs(pr-pr_lim) < 1)[0]                     # choose data in pressure range
    pr = pr[pr_ind]
    var1 =  var1[:,:,lat_ind,:][:,pr_ind,:,:][time_ind,:,:,:].squeeze()
    return dict(lat=lat,lon=lon,time=time,pr=pr,value=var1)

## read MLS water vapor observations
def read_mls(time_lim,lat_lim,pr_lim,indir,var):
    fin  = Dataset(indir)
    var1 = fin.variables[var][:].T
    lat  = fin.variables['latitude'][:]
    lon  = fin.variables['longitude'][:]
    pr   = fin.variables['pressure'][:]
    time   = fin.variables['time'][:]
    fin.close()

    ## turn coordinates into right format: longitude 0~360; latitude -90~90
    if lon[0] < 0:
        lon  = lonto360(lon)
        lon  = lonrearrange(lon)
        var1 = lonrearrange(var1)
    if lon[-1]== 0:
        lon  = np.roll(lon,1)
        var1 = np.roll(var1,1,axis=-1)
    if lon[0] == 360:
        lon[0] = 0
    if lat[0] > lat[1]:
        lat  = lat[::-1]
        var1 = var1[:,:,::-1,:]

    ## if time is in yyyymm format, turn it into year format
    if int(time[0]/100) > 1000:
        time = np.array([round(int(x/100)+(x%100-0.5)/12.,3) for x in time])
    else:
        time = np.array([round(x,3) for x in time])

    # choose data in the right range
    time_ind = np.where((time>=time_lim[0]) & (time<=time_lim[1]))[0]        # choose data in time range
    time = time[time_ind]
    lat_ind = np.where(abs(lat) <= lat_lim)[0]
    lat = lat[lat_ind]
    pr_ind = np.where(abs(pr-pr_lim) < 1)[0]
    pr = pr[pr_ind]

    var1 = var1[:,:,lat_ind,:][:,pr_ind,:,:][time_ind,:,:].squeeze()
    return dict(lat=lat,lon=lon,time=time,pr=pr,value=var1)


## calculate tropical average anomaly
def anom_calc(var):
    lat  = var['lat']
    lon  = var['lon']
    pr   = var['pr']
    var1  = var['value']
    time = var['time']
    var1_mean = np.nanmean(np.average(var1,axis=1,weights=np.cos(lat*np.pi/180)),axis=1)  # calculate tropical average

    var1_mon  = np.array([np.nanmean(var1_mean[i::12],axis=0) for i in range(12)])       # calculate monthly mean
    var1_anom = np.array([var1_mean[i]-var1_mon[i%12] for i in range(len(time))])        # calculate anomaly
    return dict(lat=lat,lon=lon,time=time,pr=pr,mean=var1_mean,anom=var1_anom)

## read water vapor from trajectory model simulations
def read_reanalysis(file_dir,time_lim,lat_lim,pr_lim,var):
    fin  = Dataset(file_dir)
    lat  = fin.variables['latitude'][:]
    lon  = fin.variables['longitude'][:]
    pr   = fin.variables['pressure'][:]
    time = fin.variables['time'][:]
    var1 = fin.variables[var][:]
    fin.close()

    ## turn coordinates into right format: longitude 0~360; latitude -90~90
    if lon[0] < 0:
        lon  = lonto360(lon)
        lon  = lonrearrange(lon)
        var1 = lonrearrange(var1)
    if lon[-1]== 0:
        lon  = np.roll(lon,1)
        var1 = np.roll(var1,1,axis=-1)
    if lon[0] == 360:
        lon[0] = 0
    if lat[0] > lat[1]:
        lat  = lat[::-1]
        var1 = var1[:,:,::-1,:]

    ## if time is in yyyymm format, turn it into year format
    if int(time[0]/100) > 1000:
        time = np.array([round(int(x/100)+(x%100-0.5)/12.,3) for x in time])
    else:
        time = np.array([round(x,3) for x in time])

    # choose data in the right range
    time_ind = np.where((time>=time_lim[0]) & (time<=time_lim[1]))[0]
    time = time[time_ind]
    lat_ind = np.where(abs(lat) <= lat_lim)[0]
    lat = lat[lat_ind]
    pr_ind = np.where(abs(pr-pr_lim) < 1)[0]
    pr = pr[pr_ind]

    var1 = var1[:,:,lat_ind,:][:,pr_ind,:,:][time_ind,:,:].squeeze()
    return dict(lat=lat,lon=lon,time=time,pr=pr,value=var1)

## read water vapor from GEOSCCM
def read_geosccm(file_dir,time_lim,lat_lim,pr_lim,var):
    fin  = Dataset(file_dir)
    lat  = fin.variables['lat'][:]
    lon  = fin.variables['lon'][:]
    pr   = fin.variables['pr'][:]
    time = fin.variables['time'][:]
    var1 = fin.variables[var][:]
    fin.close()

    ## turn coordinates into right format: longitude 0~360; latitude -90~90
    if lon[0] < 0:
        lon  = lonto360(lon)
        lon  = lonrearrange(lon)
        var1 = lonrearrange(var1)
    if lon[-1]== 0:
        lon  = np.roll(lon,1)
        var1 = np.roll(var1,1,axis=-1)
    if lon[0] == 360:
        lon[0] = 0
    if lat[0] > lat[1]:
        lat  = lat[::-1]
        var1 = var1[:,:,:,::-1,:]

    ## if time is in yyyymm format, turn it into year format
    if int(time[0]/100) > 1000:
        time = np.array([round(int(x/100)+(x%100-0.5)/12.,3) for x in time])
    else:
        time = np.array([round(x,3) for x in time])

    # choose data in the right range
    time_ind = np.where((time>=time_lim[0]) & (time<=time_lim[1]))[0]
    time = time[time_ind]
    lat_ind = np.where(abs(lat) <= lat_lim)[0]
    lat = lat[lat_ind]
    pr_ind = np.where(abs(pr-pr_lim) < 1)[0]
    pr = pr[pr_ind]

    var1 = var1[:,:,lat_ind,:][:,pr_ind,:,:][time_ind,:,:].squeeze()
    return dict(lat=lat,lon=lon,time=time,pr=pr,value=var1)

## read qbo index from downloaded text file and calculate the anomaly, time_lim with format yyyymm
def read_qbo_anom(qbo_dir,time_lim):
    with open(qbo_dir,'r') as f:
        content = f.readlines()

    n1 = 4+(int(time_lim[0]/100)-1979)                            # the first line to read data (4 lines text and the 5th line with 1979)
    nyr       = int(time_lim[1]/100)-int(time_lim[0]/100)+1  # calculate number of years
    mon_last  = time_lim[1]%100
    mon_first = time_lim[0]%100
    nmon      = (nyr-1)*12+mon_last-mon_first+1              # calculate number of months

    data1 = np.ones([nyr,12])*np.nan
    mon   = np.ones([nyr,12])*np.nan
    for i in range(nyr):
        mon[i,:]   = float(content[n1+i][0:4])+np.array([round((x+0.5)/12,3) for x in range(12)])                      # read year and combine with month
        data1[i,:] = np.array([float(content[n1+i][4+x*7:4+(x+1)*7]) for x in range(12)])          # read data column by column

    ## reshape month and data into one dimensional format
    yrmon = mon.reshape(nyr*12)[mon_first-1:mon_first+nmon-1]
    qbo   = data1.reshape(nyr*12)[mon_first-1:mon_first+nmon-1]

    ## calculate anomalies and standardized anomalies
    qbo_anom = np.array([qbo[i]-np.nanmean(qbo[i%12::12]) for i in range(nmon)])
    std = np.sqrt(np.sum((qbo-qbo.mean())**2)/nmon)
    std_anom = qbo_anom/std
    return dict(time=yrmon,value=qbo,anom=qbo_anom,stdAnom = std_anom)


## read geosccm cloud ice content
def read_cld(file_dir,time_lim,lat_lim,pr_lim,var):
    f1  = Dataset(file_dir,'r')
    lat = f1.variables['lat'][:]
    lon = f1.variables['lon'][:]
    pr  = f1.variables['pr'][:]
    cld = f1.variables[var][:]
    time = f1.variables['time'][:]
    f1.close()

    ## if time is in yyyymm format, turn it into year format
    if int(time[0]/100) > 1000:
        time = np.array([round(int(x/100)+(x%100-0.5)/12.,3) for x in time])
    else:
        time = np.array([round(x,3) for x in time])

    ## choose data with ranges
    time_ind  = np.where((time >= time_lim[0]) & (time <= time_lim[1]))[0]
    time      = time[time_ind]
    pr_ind    = np.where(pr == pr_lim)[0]
    pr        = pr[pr_ind]
    lat_ind   = np.where(abs(lat) <=lat_lim)[0]
    lat       = lat[lat_ind]
    var1      = cld[:,:,lat_ind,:][:,pr_ind,:,:][time_ind,:,:].squeeze()

    return dict(lat=lat,lon=lon,time=time,pr=pr,value=var1)

## read GEOSCCM surface temperature to calculate ONI index
def cal_ccm_oni(file_dir,time_lim):
    f1 = Dataset(file_dir,'r')
    t  = f1.variables['T'][:]
    lon = f1.variables['longitude'][:]
    lat = f1.variables['latitude'][:]
    time = f1.variables['time'][:]
    f1.close()

    ## choose nino3.4 region
    latind = np.where(abs(lat) <= 5)[0]
    lonind = np.where((lon >= 190) & (lon <= 240))[0]

    ## average temperature over nino3.4 region and calculate climate state with first 20 years
    lat = lat[latind]  ;  lon = lon[lonind]
    t_surf  = t[:,:,lonind][:,latind,:]

    ## calculate mean surface temperature in Nino3.4 region
    t_mean = np.average(t_surf.mean(axis=2),axis=1,weights=np.cos(np.pi*lat/180))

    ## calculate climate state between 2000 and 2020 in Nino3.4 region
    t_clim = np.array([t_mean[:240][i::12].mean() for i in range(12)])

    ## calculate three month moving-average of surface temperature
    N = 3
    cumsum, moving_aves = [0], []
    for i, x in enumerate(t_mean, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    t_mean=np.array(moving_aves)
    time = time[1:-1]

    ## if time is in yyyymm format, turn it into year format
    if int(time[0]/100) > 1000:
        time = np.array([round(int(x/100)+(x%100-0.5)/12.,3) for x in time])
    else:
        time = np.array([round(x,3) for x in time])

    ## choose time range
    tind = np.where((time >= time_lim[0]) & (time <= time_lim[1]))[0]
    time = time[tind]
    t_mon = t_mean[tind]

    ## calculate temperature anomaly by subtracting climate state
    t_anom = np.array([t_mon[i]-t_clim[i%12] for i in range(len(time))])
    return t_anom
