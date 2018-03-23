'''
This code is used to read the cloud top theta and height from Pfister's data.
The data are three hourly and 0.25*0.25 in lat*lon from 60S to 60N,
thus the total grid boxes are 480*1440 in lat*lon.
We re-grid the data into 2*2 in lat*lon
The frequencies are calculated in vertical direction with 1km or 2K interval.

The original data files are in IDL .sav format and the results are saved in .dat format with pickle module.
'''

from scipy.io import readsav
import pickle
from datetime import datetime,timedelta
from calendar import monthrange
from glob import glob
import numpy as np

fdir = '/co3/schoeberl/pfister_conv/'
outdir = '/ice3/hao/TTL/code\&data_ACP2018/data/'

date0 = datetime(2005,1,1)
datef = datetime(2016,12,31)
nmon  = (datef.year-date0.year)*12+datef.month-date0.month+1

## set re-gridded latitude, longitude, height and theta intervals, here 2 (degree)*2 (degree)
lat_int = 2   ;   lon_int = 2
z_int   = 1   ;   th_int  = 2
lat_lim = [-59,59]  ;   lon_lim = [0,358]
z_lim   = [8,20]    ;   th_lim  = [350,390]

nlat_tot = 480  ;  nlon_tot = 1440
n1 = int(lat_int/0.25)   ;   n2 = int(lon_int/0.25)                                     # numbers of original grids in one new grid

## calculate numbers of latitude, longitude, height and theta grids
nlat = int((lat_lim[1]-lat_lim[0])/lat_int+1)
nlon = int((lon_lim[1]-lon_lim[0])/lon_int+1)
nz   = int((z_lim[1]-z_lim[0])/z_int+1)
nth  = int((th_lim[1]-th_lim[0])/th_int+1)

## set lat, lon, z, th grids for results
lat_grid = np.arange(lat_lim[0],lat_lim[1]+1,lat_int)
lon_grid = np.arange(lon_lim[0],lon_lim[1]+1,lon_int)
z_grid   = np.arange(z_lim[0],z_lim[1]+1,z_int)
th_grid  = np.arange(th_lim[0],th_lim[1]+1,th_int)

## create variables to save results: clouds top height and theta frequencies, average clouds top height and theta
ctopz_freq  = np.zeros([nmon,nz,nlat,nlon])
ctopth_freq = np.zeros([nmon,nth,nlat,nlon])
alt_avg     = np.zeros([nmon,nlat,nlon])
th_avg      = np.zeros([nmon,nlat,nlon])
cnt         = np.zeros([nmon])                                  # this cnt variable is the total number of grids in each month for future re-average frequency
mon         = np.zeros([nmon])


date = date0
imon = 0
while (datef-date).days >= 0:
    ym = date.strftime('%Y%m')
    mon[imon] = int(ym)

    ## find out cloud files each month, read cloud top height and theta
    fn = glob(fdir+'cloudaltthetkm_'+ym+'*.sav')
    nfile = len(fn)
    fn.sort()
    alt = np.zeros([nfile,nlat_tot,nlon_tot])*np.nan
    th  = np.zeros([nfile,nlat_tot,nlon_tot])*np.nan
    for i in range(nfile):
        ## the cloud file in 2006/03/25 03:00 is in different name format
        if fn[i] == '/co3/schoeberl/pfister_conv/cloudaltthetkm_2006032503_reanmix_noice_ocean_trmm.15.9_srch.3_offs1.00_tropw0.70.sav':
            fn[i] = '/co3/schoeberl/pfister_conv/cloudaltthetkm_2006032503_reanmix_noice_ocean_trmm.15.9_srch.3_offs1.00_tropw0.70.sav.1'
        try:
            f1  = readsav(fn[i])
            lat = f1.rainlat
            lon = f1.rainlon
            alt[i,:,:] = f1.rainalt
            th[i,:,:]  = f1.rainthet
        except ValueError:
            pass

    ## calculate the frequencies and average height and theta in each new grid
    for ilat in range(nlat):
        for ilon in range(nlon):
            a = alt[:,ilat*n1:(ilat+1)*n1,ilon*n2:(ilon+1)*n2]
            b =  th[:,ilat*n1:(ilat+1)*n1,ilon*n2:(ilon+1)*n2]
            ctopz_freq[imon,:,ilat,ilon]  = (np.histogram(a,range=(z_lim[0]-z_int/2.,z_lim[1]+z_int/2.),bins=nz))[0]*1./np.size(a)
            ctopth_freq[imon,:,ilat,ilon] = (np.histogram(b,range=(th_lim[0]-th_int/2.,th_lim[1]+th_int/2.),bins=nth))[0]*1./np.size(b)
            a[a==0] = np.nan
            b[b==0] = np.nan
            alt_avg[imon,ilat,ilon]  =  np.nanmean(a)
            th_avg[imon,ilat,ilon]   =  np.nanmean(b)
    cnt[imon] = np.size(a)                                              ## a.shape is (ndays*8, 8, 8), total points in a new grids in a month

    ## increase the time to next month
    nday = monthrange(date.year,date.month)[1]
    date = date+timedelta(days=nday)
    imon += 1
    print('finish '+ym)

## pack up the results into a dictionary and save into a .dat file with pickle
data = dict(mon=mon,lat=lat_grid,lon=lon_grid,z=z_grid,th=th_grid,\
            z_freq=ctopz_freq,th_freq=ctopth_freq,z_avg=alt_avg,th_avg=th_avg,cnt=cnt)

with open(outdir+'freq_z_th_avg_'+str(lat_int)+'x'+str(lon_int)+'.dat','wb') as f:
    pickle.dump(data,f)
