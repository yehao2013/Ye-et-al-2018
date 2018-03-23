import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from multi_regress import *

indir = '/ice3/hao/TTL/regress_MLS/data/'
outdir = '/ice3/hao/TTL/regress_MLS/total/'

f1 = Dataset(indir+'GEOSCCM_monthly_surf_t_2000-2099.nc','r')
t  = f1.variables['T'][:]
lon = f1.variables['longitude'][:]
lat = f1.variables['latitude'][:]
time = f1.variables['time'][:]
f1.close()

latind = np.where(abs(lat) <= 5)[0]
lonind = np.where((lon >= 190) & (lon <= 240))[0]

lat = lat[latind]  ;  lon = lon[lonind]
t  = t[:,:,lonind][:,latind,:]
t_mean = np.average(t.mean(axis=2),axis=1,weights=np.cos(np.pi*lat/180))

t_clim = np.array([t_mean[:240][i::12].mean() for i in range(12)])


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


tind = np.where((time >= 2005) & (time <= 2017))[0]
time = time[tind]
t_mon = t_mean[tind]

t_anom = np.array([t_mon[i]-t_clim[i%12] for i in range(len(t_mon))])


qi_file = indir+'GEOSCCM_monthly_qian_2000-2020.nc'
f1 = Dataset(qi_file,'r')
qi = f1.variables['qicn'][:]
qlat = f1.variables['lat'][:]
qlon = f1.variables['lon'][:]
qtime = f1.variables['time'][:]
qpr  = f1.variables['pr'][:]
f1.close()

tind = np.where((qtime > 2005) & (qtime < 2017))[0]
latind = np.where((qlat >= -30) & (qlat <= 30))[0]
qtime = qtime[tind] ; qlat = qlat[latind]
qi = qi[tind,:,:,:][:,:,latind,:]

# calculte annual cycle of the qi
qi_mon = np.zeros(qi.shape)
for i in range(len(qtime)):
    qi_mon[i,:] = np.mean(qi[i%12::12,:],axis=0)

ind = np.where(qi_mon < 0.0001)
qi_mon[ind] = 0

qi_mon_sum = np.sum(qi_mon[:,0:3,:,:],axis=1)
qi_sum = np.sum(qi[:,0:3,:,:],axis=1)

qi_anom =np.array([qi_sum[i,:,:]-qi_sum[i%12::12,:,:].mean(axis=0) for i in range(len(qtime))])

# read temperature @100 hpa GEOSCCM
with Dataset('/sn3/geosccm/geosccm_SCF2_5_P1dm2b/P1dm2b_GEOSCCM_monthly_2000_2099.nc','r') as f:
    lat_t = f.variables['latitude'][:]
    lon_t = f.variables['longitude'][:]
    pr_t  = f.variables['pressure'][:]
    time_t = f.variables['time'][:]
    pr_ind = np.where(pr_t == 100)[0]
    latind = np.where(abs(lat_t) <= 30)[0]
    time_t = np.array([round(int(x/100)+(x%100-0.5)/12,3) for x in time_t])
    tind = np.where((time_t >= 2005) & (time_t < 2017))[0]

    tem = f.variables['isob_t'][tind,:][:,pr_ind,:][:,:,latind,:].squeeze()

lat_t = lat_t[latind]
tem_anom = np.zeros(tem.shape)
for i in range(tem.shape[0]):
     tem_anom[i,:] = tem[i,:] - tem[i%12::12,:].mean(axis=0)

nino_ind = np.where(t_anom>=0.5)[0]
qi_nino = qi_anom[nino_ind,:,:].mean(axis=0)
tem_nino = tem_anom[nino_ind,:,:].mean(axis=0)

nina_ind = np.where(t_anom<=-0.5)[0]
qi_nina = qi_anom[nina_ind,:,:].mean(axis=0)
tem_nina = tem_anom[nina_ind,:,:].mean(axis=0)

norm_ind = np.where(abs(t_anom)<=0.5)[0]
qi_norm = qi_anom[norm_ind,:,:].mean(axis=0)
tem_norm = tem_anom[norm_ind,:,:].mean(axis=0)


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

lons, lats = np.meshgrid(qlon,qlat)
lons_t, lats_t = np.meshgrid(lon_t,lat_t)
lev  = np.arange(-5,7.1,1)
fig,ax=plt.subplots(nrows=2,figsize=[8,6])
plt.rc('font', weight='bold')
plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=0.1, hspace=0.1)
m =Basemap(llcrnrlon=0,llcrnrlat=-30,urcrnrlon=360.01,urcrnrlat=30.01,projection='cyl',fix_aspect=False,ax=ax[0])
x1, y1 = m(lons, lats)
m.drawcoastlines(linewidth=0.75)
m.drawparallels(np.arange(-30,30.01,15),labels=[1,0,0,0],fontsize=10,fontweight='bold')
m.drawmeridians(np.arange(0,360,90),labels=[0,0,0,0],fontsize=10,fontweight='bold')
ax[0].set_title('GEOSCCM anvil cloud content anom above 118 hPa',fontsize=12,fontweight='bold')
ax[0].tick_params(axis='both',which ='major',labelsize=10)
im=m.contourf(lons,lats,qi_nino,levels=lev,extend='both',norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r')
cs = m.contour(lons_t,lats_t,tem_nino,levels=[-2,-1.5,-1,-0.5,0,0.5,1,1.5],colors=['magenta','magenta','magenta','magenta','k','k','k','k'])
plt.clabel(cs, fontsize=6, fmt='%1.1f',inline=1)

ax[0].text(3.5,25.5,'a) Warm phase',size=10,fontweight='bold',bbox=dict(fc='white', ec='white',boxstyle="square",fill='True'),zorder=10)

m =Basemap(llcrnrlon=0,llcrnrlat=-30,urcrnrlon=360.01,urcrnrlat=30.01,projection='cyl',fix_aspect=False,ax=ax[1])
x1, y1 = m(lons, lats)
m.drawcoastlines(linewidth=0.75)
m.drawparallels(np.arange(-30,30.01,15),labels=[1,0,0,0],fontsize=10,fontweight='bold')
m.drawmeridians(np.arange(0,360,90),labels=[0,0,0,1],fontsize=10,fontweight='bold')
ax[1].tick_params(axis='both',which ='major',labelsize=10)
im=m.contourf(lons,lats,qi_nina,levels=lev,extend='both',norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r')
cs = m.contour(lons_t,lats_t,tem_nina,levels=[-1,-0.5,0,0.5,1,1.5,2],colors=['magenta','magenta','k','k','k','k','k'])
plt.clabel(cs, fontsize=6, fmt='%1.1f',inline=1)
ax[1].text(3.5,25.5,'b) Cold phase',fontsize=10,fontweight='bold',bbox=dict(fc='white', ec='white',boxstyle="square"),zorder=10)

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
cb=fig.colorbar(im, cax=cbar_ax,extend='both',ticks=[-4,-2,0,2,4,6])
cb.ax.set_yticklabels(['-4','-2','0','2','4','6'],fontsize=10)
cb.ax.tick_params(labelsize=10)
cb.ax.set_ylabel('Cloud content anom (ppmv)',fontsize=10,fontweight='bold')

#plt.savefig(outdir+'geosccm_anvil_cloud_t100.pdf')


lev  = np.arange(-1.5,2.6,0.5)
fig,ax=plt.subplots(nrows=1,figsize=[8,4])
plt.rc('font', weight='bold')
plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=0.1, hspace=0.1)
m =Basemap(llcrnrlon=0,llcrnrlat=-30,urcrnrlon=360.01,urcrnrlat=30.01,projection='cyl',fix_aspect=False,ax=ax)
x1, y1 = m(lons, lats)
m.drawcoastlines(linewidth=0.75)
m.drawparallels(np.arange(-30,30.01,15),labels=[1,0,0,0],fontsize=10,fontweight='bold')
m.drawmeridians(np.arange(0,360,90),labels=[0,0,0,0],fontsize=10,fontweight='bold')
ax.set_title('GEOSCCM convective cloud content anom above 118 hPa',fontsize=12,fontweight='bold')
ax.tick_params(axis='both',which ='major',labelsize=10)
im=m.contourf(lons,lats,qi_norm,levels=lev,extend='both',norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r')
cs = m.contour(lons_t,lats_t,tem_norm,levels=[-0.3,-0.2,-0.1,0,0.1,0.2,0.3],colors=['magenta','magenta','magenta','magenta','k','k','k','k'])
plt.clabel(cs, fontsize=6, fmt='%1.1f',inline=1)

ax.text(3.5,25.5,'Normal',size=10,fontweight='bold',bbox=dict(fc='white', ec='white',boxstyle="square",fill='True'),zorder=10)

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
cb=fig.colorbar(im, cax=cbar_ax,extend='both',ticks=[-1,0,1,2])
cb.ax.set_yticklabels(['-1','0','1','2'],fontsize=10)
cb.ax.tick_params(labelsize=10)
cb.ax.set_ylabel('Cloud content anom (ppmv)',fontsize=10,fontweight='bold')

plt.savefig(outdir+'geosccm_anvil_cloud_t100_norm.pdf')



import numpy as np
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.colors as colors

indir = '/ice3/hao/TTL/regress_MLS/'

f1 = Dataset(indir+'170710_s100_i370_geosccm_2005-2016_6hrly_anvil.nc')
evp = f1.variables['Ice_vp_mix'][:]*32
lat = f1.variables['Lat'][:]
lon = f1.variables['Lon'][:]
lev = f1.variables['Lev'][:]
time = f1.variables['Time'][:]
f1.close()

ind = np.where(lev == 100.51439667)[0]
latind = np.where(abs(lat) <= 30)[0]
evp = evp[:,:,latind,:][:,ind,:,:].squeeze()
lat = lat[latind]
evp[np.isnan(evp)] = 0

time = np.array([round(int(x/100)+(x%100-0.5)/12,3) for x in time])

evp_anom = np.array([evp[i,:,:]-evp[i%12::12,:,:].mean(axis=0) for i in range(len(time))])

evp_nino = np.nanmean(evp_anom[nino_ind,:],axis=0)
evp_nina = np.nanmean(evp_anom[nina_ind,:],axis=0)


f1   = Dataset('/ice1/hao/geosccm/P1dm2b_GEOSCCM_monthly_u_v_2005-2016.nc','r')

ulon  =f1.variables['longitude'][:]
ulat  =f1.variables['latitude'][:]
time =f1.variables['time'][:]
pr   =f1.variables['pressure'][:]

pr_ind = np.where(pr==100)[0]
ulat_ind = np.where(abs(ulat) <=30)[0]
ulat = ulat[ulat_ind]

u  =f1.variables['isob_u'][:,:,ulat_ind,:][:,pr_ind,:,:].squeeze()
v  =f1.variables['isob_v'][:,:,ulat_ind,:][:,pr_ind,:,:].squeeze()

u_anom = np.zeros(u.shape)
v_anom = np.zeros(v.shape)
for i in range(u.shape[0]):
     u_anom[i,:] = u[i,:] - u[i%12::12,:].mean(axis=0)
     v_anom[i,:] = v[i,:] - v[i%12::12,:].mean(axis=0)

u_nino = u_anom[nino_ind,:,:].mean(axis=0)
v_nino = v_anom[nino_ind,:,:].mean(axis=0)

spd_nino=np.sqrt(u_nino*u_nino+v_nino*v_nino)

u_nina = u_anom[nina_ind,:,:].mean(axis=0)
v_nina = v_anom[nina_ind,:,:].mean(axis=0)

spd_nina=np.sqrt(u_nina*u_nina+v_nina*v_nina)

ulons,ulats=np.meshgrid(ulon,ulat)

#time_lim = [200501,201612]
#lat_lim = 30
#prT = 500
#
#outdir       ='/ice3/hao/TTL/regress_MLS/total/'
#
#geosccm_file2 = '/ice3/hao/TTL/regress_MLS/data/P1dm2b_GEOSCCM_monthly_2000_2099.nc'
#geosccm_tem         = read_reanalysis(geosccm_file2,time_lim,lat_lim,prT,'isob_t')
#lat_t500 = geosccm_tem['lat']   ;  lon_t500 = geosccm_tem['lon']
#t500 = geosccm_tem['value']
#
#t500_anom = np.zeros(t500.shape)
#for i in range(len(geosccm_tem['time'])):
#    t500_anom[i,:] = t500[i,:]-np.nanmean(t500[i%12::12,:],axis=0)
#
#t500_nino = t500_anom[nino_ind,:].mean(axis=0)
#t500_nina = t500_anom[nina_ind,:].mean(axis=0)

lons, lats = np.meshgrid(lon,lat)
#lons_t500, lats_t500 = np.meshgrid(lon_t500,lat_t500)
lev  = np.arange(-0.3,0.301,0.03)
fig,ax=plt.subplots(nrows=2,figsize=[8,6])
plt.rc('font', weight='bold')
plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=0.1, hspace=0.1)
m =Basemap(llcrnrlon=0,llcrnrlat=-30,urcrnrlon=360.01,urcrnrlat=30.01,projection='cyl',fix_aspect=False,ax=ax[0])
x1, y1 = m(lons, lats)
m.drawcoastlines(linewidth=0.75)
m.drawparallels(np.arange(-30,30.01,15),labels=[1,0,0,0],fontsize=10,fontweight='bold')
m.drawmeridians(np.arange(0,360,90),labels=[0,0,0,0],fontsize=10,fontweight='bold')
ax[0].set_title('GEOSCCM convective cloud evaporation rate anom at 100 hPa',fontsize=12,fontweight='bold')
ax[0].tick_params(axis='both',which ='major',labelsize=10)
im=m.contourf(lons,lats,evp_nino,levels=lev,norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r')
#cs = m.contour(lons_t500,lats_t500,t500_nino,levels=[-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8],colors=['magenta','magenta','magenta','magenta','k','k','k','k','k'])
#plt.clabel(cs, fontsize=6, fmt='%1.1f',inline=1)

ax[0].text(3.5,25.5,'a) Warm phase',size=10,fontweight='bold',bbox=dict(fc='white', ec='white',boxstyle="square",fill='True'),zorder=10)

X,Y=m(ulons,ulats)
yy=np.arange(0,len(ulat),3)
xx=np.arange(0,len(ulon),3)
pts=np.meshgrid(yy,xx)
m.quiver(X[pts],Y[pts],u_nino[pts],v_nino[pts],latlon=True,width=0.0015)

m =Basemap(llcrnrlon=0,llcrnrlat=-30,urcrnrlon=360.01,urcrnrlat=30.01,projection='cyl',fix_aspect=False,ax=ax[1])
x1, y1 = m(lons, lats)
m.drawcoastlines(linewidth=0.75)
m.drawparallels(np.arange(-30,30.01,15),labels=[1,0,0,0],fontsize=10,fontweight='bold')
m.drawmeridians(np.arange(0,360,90),labels=[0,0,0,1],fontsize=10,fontweight='bold')
ax[1].tick_params(axis='both',which ='major',labelsize=10)
im=m.contourf(lons,lats,evp_nina,levels=lev,norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r')
#cs = m.contour(lons_t500,lats_t500,t500_nina,levels=[-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8],colors=['magenta','magenta','magenta','magenta','k','k','k','k','k'])
#plt.clabel(cs, fontsize=6, fmt='%1.1f',inline=1)
ax[1].text(3.5,25.5,'b) Cold phase',fontsize=10,fontweight='bold',bbox=dict(fc='white', ec='white',boxstyle="square"),zorder=10)

X,Y=m(ulons,ulats)
yy=np.arange(0,len(ulat),3)
xx=np.arange(0,len(ulon),3)
pts=np.meshgrid(yy,xx)
m.quiver(X[pts],Y[pts],u_nina[pts],v_nina[pts],latlon=True,width=0.0015)

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
cb=fig.colorbar(im, cax=cbar_ax,ticks=[-0.3,-0.2,-0.1,0,0.1,0.2,0.3])
cb.ax.set_yticklabels(['-0.3','-0.2','-0.1','0','0.1','0.2','0.3'],fontsize=10)
cb.ax.tick_params(labelsize=10)
cb.ax.set_ylabel('ppmv/day',fontsize=10,fontweight='bold')

plt.savefig(outdir+'geosccm_evap_rate_enso.pdf')
