import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib import colors
import pickle
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec

## choose colormap midpoint with a certain value (usually used for use white color for zero)
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


time_lim = [200408,201612]

## set files directory
fdir     = '/ice3/hao/TTL/code&data_ACP2018/data/'
outdir   = '/ice3/hao/TTL/code&data_ACP2018/figures/'

## set coefficient number (4 trajectory runs) and variable number (3 variables: dT, BDC, QBO)
ncoef = 4
nvar  = 3

## read coefficients from MLS and geosccm regressions as well as ERAi and MERRA-2
reg_file = fdir+'reg_mls_traj_coefs_std_100hpa_200408_201612_qbo.dat'
with open(reg_file,'rb') as f:
    data = pickle.load(f)
erai = data['erai']
mer2 = data['mer2']

reg_file = fdir+'reg_geosccm_traj_coefs_std_100hpa_200501_201612.dat'
with open(reg_file,'rb') as f:
    ccm = pickle.load(f)

reg_file = fdir+'reg_geosccm_traj_coefs_std_anv_100hpa_200501_201612.dat'
with open(reg_file,'rb') as f:
    ccm2 = pickle.load(f)

lons, lats = np.meshgrid(erai['lon'],erai['lat'])
nlon = len(erai['lon'])    ;  nlat = len(erai['lat'])

## pack up the coefficients and pvalues into two variables for plotting
coefs  = np.zeros([ncoef,2,nvar,nlat,nlon])
pvalue = np.zeros([ncoef,2,nvar,nlat,nlon])
for i in range(nvar):
    coefs[0,0,:]      = erai['mls_coefs'][1:,:,:]                # MLS coefficients with ERAi regressors
    pvalue[0,0,:]     = erai['mls_pval'][1:,:,:]                 # MLS coefficient pvalues with ERAi regressors
    coefs[0,1,:]      = erai['erai_coefs'][1:,:,:]               # trajectory coefficients with ERAi regressors
    pvalue[0,1,:]     = erai['erai_pval'][1:,:,:]                # trajectory coefficient pvalues with ERAi regressors
    coefs[1,0,:]      = mer2['mls_coefs'][1:,:,:]                # MLS coefficients with MERRA-2 regressors
    pvalue[1,0,:]     = mer2['mls_pval'][1:,:,:]                 # MLS coefficient pvalues with MERRA-2 regressors
    coefs[1,1,:]      = mer2['mer2_coefs'][1:,:,:]               # trajectory coefficients with MERRA-2 regressors
    pvalue[1,1,:]     = mer2['mer2_pval'][1:,:,:]                # trajectory coefficient pvalues with ERAi regressors

    coefs[2,0,:-1,:]  = ccm['gcm_coefs'][1:,:,:]                 # GEOSCCM coefficients with GEOSCCM regressors
    pvalue[2,0,:-1,:] = ccm['gcm_pval'][1:,:,:]                  # GEOSCCM coefficient pvalues with GEOSCCM regressors
    coefs[2,1,:-1,:]  = ccm['traj_coefs'][1:,:,:]                # trajectory without ice coefficients with GEOSCCM regressors
    pvalue[2,1,:-1,:] = ccm['traj_pval'][1:,:,:]                 # trajectory without ice coefficient pvalues with GEOSCCM regressors
    coefs[3,0,:-1,:]  = ccm2['gcm_coefs'][1:,:,:]                # GEOSCCM coefficients with GEOSCCM regressors
    pvalue[3,0,:-1,:] = ccm2['gcm_pval'][1:,:,:]                 # GEOSCCM coefficient pvalues with GEOSCCM regressors
    coefs[3,1,:-1,:]  = ccm2['traj_coefs'][1:,:,:]               # trajectory with ice coefficients with GEOSCCM regressors
    pvalue[3,1,:-1,:] = ccm2['traj_pval'][1:,:,:]                # trajectory with ice coefficient pvalues with GEOSCCM regressors

## if pvalues <= 0.05, mark on the plots with black dots to indicate that the coefficient is statistically different from zero.
pval_ind = np.where(pvalue >= 0.05)
pvalue[pval_ind] = np.nan

pval_ind = np.where(pvalue <= 0.05)
pvalue[pval_ind] = 1

## set lab and title for plots
lab = np.array(['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(i)','(j)','(k)']).reshape(4,3)
title = np.array(['MLS_ERAi','traj_ERAi','MLS_MER2','traj_MER2',\
                'GEOSCCM','traj_CCM','GEOSCCM','traj_CCM_ice']).reshape(4,2)

## plot dT coefficients, three columns: MLS/GEOSCCM, trajectory models, and scatter plots
fig,ax=plt.subplots(nrows=4,ncols=4,figsize=[9,8])
gs = gridspec.GridSpec(4, 5)
for i1 in range(4):
    ax[i1,0] = plt.subplot(gs[i1, 0:2])
    ax[i1,1] = plt.subplot(gs[i1, 2:4])
    ax[i1,2] = plt.subplot(gs[i1, 4:5])

plt.subplots_adjust(left=0.05, bottom=None, right=0.95, top=None, wspace=0.1, hspace=0.3)                                      # change spaces between subplots
t = fig.text(0.5, 0.95, '$\Delta$T coefficients [ppmv/K]',horizontalalignment='center',fontproperties=FontProperties(size=14)) # add title
ax[-1,0].axis('off')                                                                                                           # do not plot at bottom left

s_min, s_max, inter = [-0.25, 0.85, 0.2]                                                                                       # set colorbar range

for ii in range(ncoef):
    jj = 0                            # MLS/GEOSCCM coefficients
    if ii != ncoef-1:
        m =Basemap(llcrnrlon=0,llcrnrlat=-30,urcrnrlon=360.01,urcrnrlat=30.01,projection='cyl',fix_aspect=False,ax=ax[ii,jj])
        x1, y1 = m(lons, lats)
        m.drawcoastlines(linewidth=0.75)
        m.drawparallels(np.arange(-30,30.01,15),labels=[1,0,0,0],fontsize=8)
        m.drawmeridians(np.arange(0,360,90),labels=[0,0,0,0],fontsize=8)
        ax[ii,jj].tick_params(axis='both',which ='major',labelsize=8)                                                          # set ticks label size
        ax[ii,jj].set_title(title[ii,jj],size=10)

        img0=m.pcolormesh(x1,y1,coefs[ii,0,0,:,:],norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r', vmin=s_min, vmax=s_max)   # plot coefficients with colormap midpoint at 0.0
        cb=plt.colorbar(img0,ax=ax[ii,jj],ticks=np.arange(-0.2,0.81,0.2),extend='both')
        cb.ax.tick_params(labelsize=8)                                                                                         # set colorbar ticks label size
        #im = m.scatter(x1,y1,pvalue[ii,0,0,:,:])         ## add pvalues indicating the coefficients are statistically different from zeros
        plt.text(0.05, 1.09,lab[ii,jj],fontsize=10, ha='center', va='center', transform=ax[ii,jj].transAxes)                   ## add label at the top left for each plot

    jj += 1                          # trajectory coefficients
    m =Basemap(llcrnrlon=0,llcrnrlat=-30,urcrnrlon=360.01,urcrnrlat=30.01,projection='cyl',fix_aspect=False,ax=ax[ii,jj])
    x1, y1 = m(lons, lats)
    m.drawcoastlines(linewidth=0.75)
    m.drawparallels(np.arange(-30,30.01,15),labels=[0,0,0,0],fontsize=8)
    m.drawmeridians(np.arange(0,360,90),labels=[0,0,0,0],fontsize=8)
    ax[ii,jj].tick_params(axis='both',which ='major',labelsize=8)                                                              # set ticks label size
    ax[ii,jj].set_title(title[ii,jj],size=10)

    img1=m.pcolormesh(x1,y1,coefs[ii,1,0,:,:],norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r', vmin=s_min, vmax=s_max)       # plot coefficients with colormap midpoint at 0.0

    cb=plt.colorbar(img1,ax=ax[ii,jj],ticks=np.arange(-0.2,0.81,0.2),extend='both')
    cb.ax.tick_params(labelsize=8)                                                                                             # set colorbar ticks label size
    #im = m.scatter(x1,y1,pvalue[ii,1,0,:,:])            ## add pvalues indicating the coefficients are statistically different from zeros
    plt.text(0.05, 1.09,lab[ii,jj],fontsize=10, ha='center', va='center', transform=ax[ii,jj].transAxes)                       ## add label at the top left for each plot

    jj += 1                         # scatter plots
    if ii in [0,2,3]:
        axis_lim=[-0.4,0.8]
        axis_tick = [-0.4,0,0.4,0.8]
    else:
        axis_lim=[-0.8,0.8]
        axis_tick = [-0.8,-0.4,0,0.4,0.8]
    ax[ii,jj].tick_params(axis='both',which ='major',labelsize=8)
    ax[ii,jj].scatter(coefs[ii,0,0,:,:],coefs[ii,1,0,:,:],s=0.75)
    ax[ii,jj].plot(axis_lim,axis_lim,'k--')
    ax[ii,jj].yaxis.tick_right()
    ax[ii,jj].set_xlim(axis_lim)
    ax[ii,jj].set_xticks(axis_tick)
    ax[ii,jj].set_ylim(axis_lim)
    ax[ii,jj].set_yticks(axis_tick)
    ax[ii,jj].set_title(title[ii,0],fontsize=10)
    ax[ii,jj].set_ylabel(title[ii,1],fontsize=10)
    plt.text(0.05, 1.09,lab[ii,jj],fontsize=10, ha='center', va='center', transform=ax[ii,jj].transAxes)

#plt.savefig(outdir+'T_coefficients_mls_erai_merra2_ccm.pdf')

## plot BDC coefficients, three columns: MLS/GEOSCCM, trajectory models, and scatter plots
fig,ax=plt.subplots(nrows=4,ncols=4,figsize=[9,8])
gs = gridspec.GridSpec(4, 5)
for i1 in range(4):
    ax[i1,0] = plt.subplot(gs[i1, 0:2])
    ax[i1,1] = plt.subplot(gs[i1, 2:4])
    ax[i1,2] = plt.subplot(gs[i1, 4:5])

plt.subplots_adjust(left=0.05, bottom=None, right=0.95, top=0.9, wspace=0.1, hspace=0.3)                                       # change spaces between subplots
t = fig.text(0.5, 0.95, 'BDC coefficients [ppmv/(K/day)]',horizontalalignment='center',fontproperties=FontProperties(size=12)) # add title
ax[-1,0].axis('off')                                                                                                           # do not plot at bottom left

s_min, s_max, inter = [[-4, -4, -12, -12],[0, 0, 0, 0],[1, 1, 4,4]]                                                            # set colorbar range
for ii in range(ncoef):
    jj = 0                            # MLS/GEOSCCM coefficients
    if ii != ncoef-1:
        m =Basemap(llcrnrlon=0,llcrnrlat=-30,urcrnrlon=360.01,urcrnrlat=30.01,projection='cyl',fix_aspect=False,ax=ax[ii,jj])
        x1, y1 = m(lons, lats)
        m.drawcoastlines(linewidth=0.75)
        m.drawparallels(np.arange(-30,30.01,15),labels=[1,0,0,0],fontsize=8)
        m.drawmeridians(np.arange(0,360,90),labels=[0,0,0,0],fontsize=8)
        ax[ii,jj].tick_params(axis='both',which ='major',labelsize=8)
        ax[ii,jj].set_title(title[ii,jj],size=10)

        ## when ii == 1, extend the colorbar in the minimum direction
        if ii == 1:
            img0=m.pcolor(x1,y1,coefs[ii,0,1,:,:],norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r', vmin=s_min[ii]-0.05, vmax=s_max[ii])   # plot coefficients with colormap midpoint at 0.0
            #im = m.scatter(x1,y1,pvalue[ii,0,1,:,:],marker='.')                                 ## add pvalues indicating the coefficients are statistically different from zeros
            cb=plt.colorbar(img0,ax=ax[ii,jj],ticks=np.arange(s_min[ii],s_max[ii]+0.001,inter[ii]),extend='min')
        else:
            img0=m.pcolor(x1,y1,coefs[ii,0,1,:,:],norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r', vmin=s_min[ii], vmax=s_max[ii])
            #im = m.scatter(x1,y1,pvalue[ii,0,1,:,:],marker='.')
            cb=plt.colorbar(img0,ax=ax[ii,jj],ticks=np.arange(s_min[ii],s_max[ii]+0.001,inter[ii]))
        cb.ax.tick_params(labelsize=8)
        plt.text(0.05, 1.09,lab[ii,jj],fontsize=10, ha='center', va='center', transform=ax[ii,0].transAxes)

    jj += 1                         # trajectory coefficients
    m =Basemap(llcrnrlon=0,llcrnrlat=-30,urcrnrlon=360.01,urcrnrlat=30.01,projection='cyl',fix_aspect=False,ax=ax[ii,jj])
    x1, y1 = m(lons, lats)
    m.drawcoastlines(linewidth=0.75)
    m.drawparallels(np.arange(-30,30.01,15),labels=[0,0,0,0],fontsize=8)
    m.drawmeridians(np.arange(0,360,90),labels=[0,0,0,0],fontsize=8)
    ax[ii,jj].tick_params(axis='both',which ='major',labelsize=8)
    ax[ii,jj].set_title(title[ii,jj],size=10)

    if ii == 1:
        img0=m.pcolor(x1,y1,coefs[ii,1,1,:,:],norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r', vmin=s_min[ii]-0.05, vmax=s_max[ii]+0.05)   # plot coefficients with colormap midpoint at 0.0
        #im = m.scatter(x1,y1,pvalue[ii,1,1,:,:],marker='.')                                     ## add pvalues indicating the coefficients are statistically different from zeros
        cb=plt.colorbar(img0,ax=ax[ii,jj],ticks=np.arange(s_min[ii],s_max[ii]+0.001,inter[ii]),extend='both')
    else:
        img0=m.pcolor(x1,y1,coefs[ii,1,1,:,:],norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r', vmin=s_min[ii], vmax=s_max[ii])
        #im = m.scatter(x1,y1,pvalue[ii,1,1,:,:],marker='.')
        cb=plt.colorbar(img0,ax=ax[ii,jj],ticks=np.arange(s_min[ii],s_max[ii]+0.001,inter[ii]))
    cb.ax.tick_params(labelsize=8)
    plt.text(0.05, 1.09,lab[ii,jj],fontsize=10, ha='center', va='center', transform=ax[ii,jj].transAxes)

    jj += 1
    if ii in [0,1]:
        axis_lim=[-4.5,0.5]
        axis_tick = [-4,-3,-2,-1,0]
    else:
        axis_lim=[-12,0]
        axis_tick = [-12,-8,-4,0]
    ax[ii,jj].tick_params(axis='both',which ='major',labelsize=8)
    ax[ii,jj].scatter(coefs[ii,0,1,:,:],coefs[ii,1,1,:,:],s=0.75)
    ax[ii,jj].plot(axis_lim,axis_lim,'k--')
    ax[ii,jj].yaxis.tick_right()
    ax[ii,jj].set_xlim(axis_lim)
    ax[ii,jj].set_xticks(axis_tick)
    ax[ii,jj].set_ylim(axis_lim)
    ax[ii,jj].set_yticks(axis_tick)
    ax[ii,jj].set_title(title[ii,0],fontsize=10)
    ax[ii,jj].set_ylabel(title[ii,1],fontsize=10)
    plt.text(0.05, 1.09,lab[ii,jj],fontsize=10, ha='center', va='center', transform=ax[ii,jj].transAxes)
#plt.savefig(outdir+'BDC_coefficients_mls_erai_merra2_ccm.pdf')


## plot QBO coefficients, three columns: MLS, trajectory models, and scatter plots
fig,ax=plt.subplots(nrows=2,ncols=3,figsize=[9,4])
gs = gridspec.GridSpec(2, 5)
for i1 in range(2):
    ax[i1,0] = plt.subplot(gs[i1, 0:2])
    ax[i1,1] = plt.subplot(gs[i1, 2:4])
    ax[i1,2] = plt.subplot(gs[i1, 4:5])

plt.subplots_adjust(left=0.05, bottom=None, right=0.95, top=0.87, wspace=0.1, hspace=0.3)
t = fig.text(0.5, 0.95, 'QBO coefficients [ppmv/(m/s)]',horizontalalignment='center',fontproperties=FontProperties(size=12))

for ii in range(ncoef-2):
    jj = 0
    if ii == 0:
        s_min, s_max, inter = [-0.1,0.3,0.1]
    else:
        s_min, s_max, inter = [0,0.3,0.1]

    m =Basemap(llcrnrlon=0,llcrnrlat=-30,urcrnrlon=360.01,urcrnrlat=30.01,projection='cyl',fix_aspect=False,ax=ax[ii,jj])
    x1, y1 = m(lons, lats)
    m.drawcoastlines(linewidth=0.75)
    m.drawparallels(np.arange(-30,30.01,15),labels=[1,0,0,0],fontsize=8)
    m.drawmeridians(np.arange(0,360,90),labels=[0,0,0,0],fontsize=8)
    ax[ii,jj].tick_params(axis='both',which ='major',labelsize=8)
    ax[ii,jj].set_title(title[ii,jj],size=10)

    img0=m.pcolor(x1,y1,coefs[ii,0,2,:,:],norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r', vmin=s_min, vmax=s_max)
    #im = m.scatter(x1,y1,pvalue[ii,0,2,:,:],marker='.')
    cb=plt.colorbar(img0,ax=ax[ii,jj],ticks=np.arange(s_min,s_max+0.001,inter))
    cb.ax.tick_params(labelsize=8)
    plt.text(0.05, 1.07,lab[ii,jj],fontsize=10, ha='center', va='center', transform=ax[ii,0].transAxes)

    jj += 1
    m =Basemap(llcrnrlon=0,llcrnrlat=-30,urcrnrlon=360.01,urcrnrlat=30.01,projection='cyl',fix_aspect=False,ax=ax[ii,jj])
    x1, y1 = m(lons, lats)
    m.drawcoastlines(linewidth=0.75)
    m.drawparallels(np.arange(-30,30.01,15),labels=[0,0,0,0],fontsize=8)
    m.drawmeridians(np.arange(0,360,90),labels=[0,0,0,0],fontsize=8)
    ax[ii,jj].tick_params(axis='both',which ='major',labelsize=8)
    ax[ii,jj].set_title(title[ii,jj],size=10)

    img0=m.pcolor(x1,y1,coefs[ii,1,2,:,:],norm=MidpointNormalize(midpoint=0.),cmap='RdBu_r', vmin=s_min, vmax=s_max)
    #im = m.scatter(x1,y1,pvalue[ii,1,2,:,:],marker='.')
    cb=plt.colorbar(img0,ax=ax[ii,1],ticks=np.arange(s_min,s_max+0.001,inter))
    cb.ax.tick_params(labelsize=8)
    plt.text(0.05, 1.07,lab[ii,jj],fontsize=10, ha='center', va='center', transform=ax[ii,jj].transAxes)

    jj += 1
    if ii == 0:
        axis_lim=[-0.1,0.3]
        axis_tick = [-0.1,0,0.1,0.2,0.3]
    else:
        axis_lim=[0,0.3]
        axis_tick = [0,0.1,0.2,0.3]
    ax[ii,jj].tick_params(axis='both',which ='major',labelsize=8)
    ax[ii,jj].scatter(coefs[ii,0,2,:,:],coefs[ii,1,2,:,:],s=0.75)
    ax[ii,jj].plot(axis_lim,axis_lim,'k--')
    ax[ii,jj].yaxis.tick_right()
    ax[ii,jj].set_xlim(axis_lim)
    ax[ii,jj].set_xticks(axis_tick)
    ax[ii,jj].set_ylim(axis_lim)
    ax[ii,jj].set_yticks(axis_tick)
    ax[ii,jj].set_title(title[ii,0],fontsize=10)
    ax[ii,jj].set_ylabel(title[ii,1],fontsize=10)
    plt.text(0.05, 1.07,lab[ii,jj],fontsize=10, ha='center', va='center', transform=ax[ii,jj].transAxes)
#plt.savefig(outdir+'qbo_coefficients_mls_erai_merra2.pdf')
