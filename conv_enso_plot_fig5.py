from netCDF4 import Dataset
import numpy as np
import pickle
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from multi_regress import read_oni,cal_ccm_oni,read_reanalysis

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


fdir = '/ice3/hao/TTL/code&data_ACP2018/data/'
outdir = '/ice3/hao/TTL/code&data_ACP2018/figures/'

time_lim1 = [200408,201612]                                                         # time limit for observations
t_lim1 = [int(x/100)+round((x%100-0.5)/12,3) for x in time_lim1]                    # turn time limit into year form

time_lim2 = [200501,201612]                                                         # time limit for GEOSCCM
t_lim2 = [int(x/100)+round((x%100-0.5)/12,3) for x in time_lim2]                    # turn time limit into year form
th_lim    = 365                                                                     # theta limit for cloud percentage
lat_lim   = 30
pr_lim    = 100                                                                     # pressure limit for TTL temperature, here use 100 hPa
qpr_lim   = 118                                                                     # pressure limit for GEOSCCM convective ice water content, here use 118 hPa

## read nino3.4 temperature ENSO ONI index
oni = read_oni(fdir+'oni_detrend.nino34.ascii.txt',t_lim1)

## read temperature at 100 hPa from ERAi with high resolution, 1x1
f1 = fdir+'ECMWF_ERAi_monthly_t_hr_2000-2016_full_resolution.nc'
erai_t100 = read_reanalysis(f1,t_lim1,lat_lim,pr_lim,'isob_t')

t100_lon = erai_t100['lon']
t100_lat = erai_t100['lat']
nlat = len(t100_lat)            ;    nlon = len(t100_lon)
nmon = len(erai_t100['time'])

## calculate temperature anomalies at 100 hPa
t100_anom = np.array([erai_t100['value'][i,:,:] - erai_t100['value'][i%12::12,:,:].mean(axis=0) for i in range(nmon)])


## read cloud frequency potential temperatures and heights
conv_file = fdir+'freq_z_th_avg_2x2.nc'
f = Dataset(conv_file,'r')
latc     = f.variables['lat'][:]
lat_indc = np.where(abs(latc) <= lat_lim)[0]
latc     = latc[lat_indc]
lonc     = f.variables['lon'][:]
thf      = f.variables['th_freq'][:,:,lat_indc,:]
mon      = f.variables['time'][:]
mon      = np.array([round(int(x/100)+(x%100-0.5)/12,3) for x in mon])

## here consider total cloud frequency above a potential temperature level, same as choosing a certain level, then calculate cloud frequency anomalies
th_ind = np.where(f.variables['theta'][:] >= th_lim)[0]
thf_sum = np.sum(thf[:,th_ind,:,:],axis=1)
thf_anom = np.array([thf_sum[i,:]-np.mean(thf_sum[i%12::12,:],axis=0) for i in range(len(mon))])

## choose cloud frequency anomalies and T@100 hPa for El Nino and La Nina conditions based on ONI index and average
tind = np.where(oni >= 0.5)[0]
thf_nino = np.nanmean(thf_anom[tind,:,:],axis=0)*100                            ## turn into percentage
t100_nino = t100_anom[tind,:,:].mean(axis=0)

tind = np.where(oni <= -0.5)[0]
thf_nina = np.nanmean(thf_anom[tind,:,:],axis=0)*100
t100_nina = t100_anom[tind,:,:].mean(axis=0)


## read GEOSCCM qicn at 118 hPa
with Dataset(fdir+'GEOSCCM_monthly_qian_2000-2020.nc','r') as f:
    qlat = f.variables['lat'][:]
    qlon = f.variables['lon'][:]
    qtime = f.variables['time'][:]
    qpr  = f.variables['pr'][:]

    ## apply time, latitude, and pressure limit
    tind = np.where((qtime > t_lim2[0]) & (qtime < t_lim2[1]+1))[0]
    latind = np.where(abs(qlat <= lat_lim))[0]
    qprind = np.where(abs(qpr-qpr_lim)<1)[0]
    qi      = f.variables['qicn'][:,qprind,:,:][tind,:,:,:][:,:,latind,:].squeeze()

qtime = qtime[tind] ; qlat = qlat[latind]

# calculate anomalies by subtracting the annual cycle
qi_anom =np.array([qi[i,:,:]-qi[i%12::12,:,:].mean(axis=0) for i in range(len(qtime))])

# read GEOSCCM temperature @100 hpa
with Dataset(fdir+'P1dm2b_GEOSCCM_monthly_2000_2099.nc','r') as f:
    lat_tem = f.variables['latitude'][:]
    lon_tem = f.variables['longitude'][:]
    pr_tem  = f.variables['pressure'][:]
    time_tem = f.variables['time'][:]
    pr_ind = np.where(pr_tem == pr_lim)[0]
    latind = np.where(abs(lat_tem) <= lat_lim)[0]
    time_tem = np.array([round(int(x/100)+(x%100-0.5)/12,3) for x in time_tem])
    tind = np.where((time_tem >= t_lim2[0]) & (time_tem < t_lim2[1]+1))[0]
    tem = f.variables['isob_t'][tind,:][:,pr_ind,:][:,:,latind,:].squeeze()

lat_tem = lat_tem[latind]

# calculate anomalies by subtracting the annual cycle
tem_anom = np.array([tem[i,:,:]-tem[i%12::12,:,:].mean(axis=0) for i in range(tem.shape[0])])

## read surface temperature from GEOSCCM to calculate ONI index
t_anom = cal_ccm_oni(fdir+'GEOSCCM_monthly_surf_t_2000-2099.nc',t_lim2)

## choose GEOSCCM convective ice water content and 100hPa temperature anomalies based on GEOSCCM surface temperature index
nino_ind = np.where(t_anom>=0.5)[0]
qi_nino = qi_anom[nino_ind,:,:].mean(axis=0)
tem_nino = tem_anom[nino_ind,:,:].mean(axis=0)

nina_ind = np.where(t_anom<=-0.5)[0]
qi_nina = qi_anom[nina_ind,:,:].mean(axis=0)
tem_nina = tem_anom[nina_ind,:,:].mean(axis=0)


## set lon, lat, and contourf levels for figures
lons1, lats1 = np.meshgrid(lonc,latc)
lons_t100, lats_t100 = np.meshgrid(t100_lon,t100_lat)
lev1  = np.arange(-0.3,0.51,0.05)

lons2, lats2 = np.meshgrid(qlon,qlat)
lons_tem, lats_tem = np.meshgrid(lon_tem,lat_tem)
lev2  = np.arange(-3.5,6.6,0.5)

## plot figure 5
fig,ax=plt.subplots(nrows=2,ncols=2,figsize=[12,6])
plt.rc('font', weight='bold')
plt.subplots_adjust(left=0.05, bottom=None, right=0.95, top=None, wspace=0.05, hspace=0.1)
m =Basemap(llcrnrlon=0,llcrnrlat=-30,urcrnrlon=360.01,urcrnrlat=30.01,projection='cyl',fix_aspect=False,ax=ax[0,0])
x1, y1 = m(lons1, lats1)
m.drawcoastlines(linewidth=0.75)
m.drawparallels(np.arange(-30,30.01,15),labels=[1,0,0,0],fontsize=10,fontweight='bold')
m.drawmeridians(np.arange(0,360,90),labels=[0,0,0,0],fontsize=10,fontweight='bold')
ax[0,0].tick_params(axis='both',which ='major',labelsize=10)
im=m.contourf(x1,y1,thf_nina,levels=lev1,extend='both',norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r')
cs = m.contour(lons_t100,lats_t100,t100_nina,levels=[-0.8,-0.4,0,0.4,0.8,1.2,1.6,2],colors=['magenta','magenta','k','k','k','k','k','k'])
plt.clabel(cs, fontsize=6, fmt='%1.1f',inline=1)

ax[0,0].text(4,25.5,'(a) La Nina',size=10,fontweight='bold',bbox=dict(fc='white', ec='white',boxstyle="square",fill='True'),zorder=10)

m =Basemap(llcrnrlon=0,llcrnrlat=-30,urcrnrlon=360.01,urcrnrlat=30.01,projection='cyl',fix_aspect=False,ax=ax[0,1])
x1, y1 = m(lons1, lats1)
m.drawcoastlines(linewidth=0.75)
m.drawparallels(np.arange(-30,30.01,15),labels=[0,0,0,0],fontsize=10,fontweight='bold')
m.drawmeridians(np.arange(0,360,90),labels=[0,0,0,0],fontsize=10,fontweight='bold')
ax[0,1].tick_params(axis='both',which ='major',labelsize=10)
im=m.contourf(x1,y1,thf_nino,levels=lev1,extend='both',norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r')
cs = m.contour(lons_t100,lats_t100,t100_nino,levels=[-2,-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2],colors=['magenta','magenta','magenta','magenta','magenta','k','k','k','k'])
plt.clabel(cs, fontsize=6, fmt='%1.1f',inline=1)
ax[0,1].text(4,25.5,'(b) El Nino',fontsize=10,fontweight='bold',bbox=dict(fc='white', ec='white',boxstyle="square"),zorder=10)

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.55, 0.01, 0.35])
cb=fig.colorbar(im, cax=cbar_ax,extend='both',ticks=[-0.2,0,0.2,0.4])
cb.ax.set_yticklabels(['-0.2','0','0.2','0.4'],fontsize=10)
cb.ax.tick_params(labelsize=10)
cb.ax.set_ylabel('Cloud pct anom (%)\nabove 365 K',fontsize=9,fontweight='bold')


m =Basemap(llcrnrlon=0,llcrnrlat=-30,urcrnrlon=360.01,urcrnrlat=30.01,projection='cyl',fix_aspect=False,ax=ax[1,0])
x1, y1 = m(lons2, lats2)
m.drawcoastlines(linewidth=0.75)
m.drawparallels(np.arange(-30,30.01,15),labels=[1,0,0,0],fontsize=10,fontweight='bold')
m.drawmeridians(np.arange(0,360,90),labels=[0,0,0,0],fontsize=10,fontweight='bold')
ax[1,0].tick_params(axis='both',which ='major',labelsize=10)
im=m.contourf(x1,y1,qi_nina,levels=lev2,extend='both',norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r')
cs = m.contour(lons_tem,lats_tem,tem_nina,levels=[-1,-0.5,0,0.5,1,1.5,2],colors=['magenta','magenta','k','k','k','k','k'])
plt.clabel(cs, fontsize=6, fmt='%1.1f',inline=1)

ax[1,0].text(4,25.5,'(c) Cold phase',size=10,fontweight='bold',bbox=dict(fc='white', ec='white',boxstyle="square",fill='True'),zorder=10)

m =Basemap(llcrnrlon=0,llcrnrlat=-30,urcrnrlon=360.01,urcrnrlat=30.01,projection='cyl',fix_aspect=False,ax=ax[1,1])
x1, y1 = m(lons2, lats2)
m.drawcoastlines(linewidth=0.75)
m.drawparallels(np.arange(-30,30.01,15),labels=[0,0,0,0],fontsize=10,fontweight='bold')
m.drawmeridians(np.arange(0,360,90),labels=[0,0,0,0],fontsize=10,fontweight='bold')
ax[1,1].tick_params(axis='both',which ='major',labelsize=10)
im=m.contourf(x1,y1,qi_nino,levels=lev2,extend='both',norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r')
cs = m.contour(lons_tem,lats_tem,tem_nino,levels=[-2,-1.5,-1,-0.5,0,0.5,1,1.5],colors=['magenta','magenta','magenta','magenta','k','k','k','k'])
plt.clabel(cs, fontsize=6, fmt='%1.1f',inline=1)
ax[1,1].text(4,25.5,'(d) Warm phase',fontsize=10,fontweight='bold',bbox=dict(fc='white', ec='white',boxstyle="square"),zorder=10)


cbar_ax = fig.add_axes([0.92, 0.1, 0.01, 0.35])
cb=fig.colorbar(im, cax=cbar_ax,extend='both',ticks=[-3,0,3,6])
cb.ax.set_yticklabels(['-3','0','3','6'],fontsize=10)
cb.ax.tick_params(labelsize=10)
cb.ax.set_ylabel('Cloud IWC anom (ppmv) \nGEOSCCM at 118 hPa',fontsize=9,fontweight='bold')

#plt.savefig(outdir+'cloud_pct_ccm_anvil_enso.pdf')
