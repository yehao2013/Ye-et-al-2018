import numpy as np
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from multi_regress import cal_ccm_oni

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

fdir   = '/ice3/hao/TTL/code&data_ACP2018/data/'
outdir = '/ice3/hao/TTL/code&data_ACP2018/figures/'
time_lim = [200501,201612]
t_lim = np.array([round(int(x/100)+(x%100-0.5)/12,3) for x in time_lim])
lat_lim = 30
pr_evap = 100

## read GEOSCCM evaporation rate date
f1 = Dataset(fdir+'isob_traj_s100_i370_geosccm_2005-2016_6hrly_anvil_evaporation_rate.nc')
evp = f1.variables['Ice_vp_mix'][:]*32                                                       # turn the evaporation rate unit into ppv/day
lat = f1.variables['Lat'][:]
lon = f1.variables['Lon'][:]
lev = f1.variables['Lev'][:]
time = f1.variables['Time'][:]
f1.close()

## choose evaporation rate date over tropics
prind = np.where(abs(lev-pr_evap) < 1)[0]
latind = np.where(abs(lat) <= lat_lim)[0]
lat = lat[latind]
tind = np.where((time >= time_lim[0]) & (time <= time_lim[1]))[0]
evp = evp[tind,:,:,:][:,prind,:,:][:,:,latind,:].squeeze()

evp[np.isnan(evp)] = 0                                                                       # turn NaN to zero
evp_anom = np.array([evp[i,:,:]-evp[i%12::12,:,:].mean(axis=0) for i in range(len(time))])   # calculate evaporation rate anomaly

time = np.array([round(int(x/100)+(x%100-0.5)/12,3) for x in time])                          # turn time into year format

## read surface temperature from GEOSCCM to calculate ONI index
f1 = fdir+'GEOSCCM_monthly_surf_t_2000-2099.nc'
t_anom = cal_ccm_oni(f1,t_lim)

## read horizontal winds at 100 hPa from GEOSCCM over tropics
with Dataset(fdir+'P1dm2b_GEOSCCM_monthly_u_v_2005-2016.nc','r') as f:
    lat_uv  = f.variables['latitude'][:]
    lon_uv  = f.variables['longitude'][:]
    pr_uv   = f.variables['pressure'][:]
    time_uv = f.variables['time'][:]

    pr_ind  = np.where(pr_uv == 100)[0]
    latind  = np.where(abs(lat_uv) <= 30)[0]
    time_uv = np.array([round(int(x/100)+(x%100-0.5)/12,3) for x in time_uv])
    tind = np.where((time_uv >= 2005) & (time_uv < 2017))[0]

    u = f.variables['isob_u'][tind,:][:,pr_ind,:][:,:,latind,:].squeeze()
    v = f.variables['isob_v'][tind,:][:,pr_ind,:][:,:,latind,:].squeeze()

lat_uv = lat_uv[latind]

## calculate horizontal wind anomalies
u_anom = np.array([u[i,:]-u[i%12::12,:].mean(axis=0) for i in range(len(time_uv))])
v_anom = np.array([v[i,:]-v[i%12::12,:].mean(axis=0) for i in range(len(time_uv)) ])

## average evaporation rate and u&v in warm and cold conditions
nino_ind = np.where(t_anom>=0.5)[0]
evp_nino = evp_anom[nino_ind,:,:].mean(axis=0)
u_nino   = u_anom[nino_ind,:,:].mean(axis=0)
v_nino   = v_anom[nino_ind,:,:].mean(axis=0)

nina_ind = np.where(t_anom<=-0.5)[0]
evp_nina = evp_anom[nina_ind,:,:].mean(axis=0)
u_nina   = u_anom[nina_ind,:,:].mean(axis=0)
v_nina   = v_anom[nina_ind,:,:].mean(axis=0)


## plot evaporation rate anomalies and horizontal winds anomalies over warm and cold phases
lons1, lats1 = np.meshgrid(lon,lat)
lev1 = np.linspace(-0.3,0.3,21)
lons2, lats2 = np.meshgrid(lon_uv,lat_uv)

fig,ax=plt.subplots(nrows=2,figsize=[8,6])
ax[0].set_title('GEOSCCM convective cloud evaporation rate anom at 100 hPa',weight='bold',fontsize=12)
plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=0.05, hspace=0.1)

## plot evaporation rate anomalies and wind vector in cold phase
m =Basemap(llcrnrlon=0,llcrnrlat=-30,urcrnrlon=360.01,urcrnrlat=30.01,projection='cyl',fix_aspect=False,ax=ax[0])
x1, y1 = m(lons1, lats1)
x2, y2 = m(lons2, lats2)
m.drawcoastlines(linewidth=0.75)
m.drawparallels(np.arange(-30,30.01,15),labels=[1,0,0,0],fontsize=10,fontweight='bold')
m.drawmeridians(np.arange(0,360,90),labels=[0,0,0,0],fontsize=10,fontweight='bold')
ax[0].tick_params(axis='both',which ='major',labelsize=10)
im1=m.contourf(x1,y1,evp_nina,levels=lev1,norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r')                                            # plot evaporation rate anomalies in cold condition
Q = ax[0].quiver(x2[::3, ::3], y2[::3, ::3], u_nina[::3, ::3], v_nina[::3, ::3],width=0.002)                                            # plot wind vectors
ax[0].text(4,25.5,'(a) Cold Phase',size=10,fontweight='bold',bbox=dict(fc='white', ec='white',boxstyle="square",fill='True'),zorder=10)  # add text


## plot evaporation rate anomalies and wind vector in warm phase
m =Basemap(llcrnrlon=0,llcrnrlat=-30,urcrnrlon=360.01,urcrnrlat=30.01,projection='cyl',fix_aspect=False,ax=ax[1])
x1, y1 = m(lons1, lats1)
m.drawcoastlines(linewidth=0.75)
m.drawparallels(np.arange(-30,30.01,15),labels=[1,0,0,0],fontsize=10,fontweight='bold')
m.drawmeridians(np.arange(0,360,90),labels=[0,0,0,1],fontsize=10,fontweight='bold')
ax[1].tick_params(axis='both',which ='major',labelsize=10)
im=m.contourf(x1,y1,evp_nino,levels=lev1,norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r')
Q = ax[1].quiver(x2[::3, ::3], y2[::3, ::3], u_nino[::3, ::3], v_nino[::3, ::3],width=0.002)
ax[1].text(4,25.5,'(b) Warm Phase',fontsize=10,fontweight='bold',bbox=dict(fc='white', ec='white',boxstyle="square"),zorder=10)

## add colorbar to the right of the plots
fig.subplots_adjust(right=0.86)
cbar_ax = fig.add_axes([0.88, 0.1, 0.02, 0.8])
cb=fig.colorbar(im, cax=cbar_ax,ticks=[-0.3,-0.2,-0.1,0,0.1,0.2,0.3])#,orientation='horizontal')
cb.ax.set_yticklabels(['-0.3','-0.2','-0.1','0','0.1','0.2','0.3'],fontsize=10)
cb.ax.tick_params(labelsize=10)
cb.ax.set_ylabel('ppmv day$^{-1}$',fontsize=9,fontweight='bold')

plt.savefig(outdir+'geosccm_evap_rate_enso.pdf')
