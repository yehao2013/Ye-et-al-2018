import numpy as np
from multi_regress import read_traj,read_mls,anom_calc
import matplotlib.pyplot as plt

time_lim = [200408,201612]
lat_lim = 30
pr_lim = 100

fdir       ='/ice3/hao/TTL/code&data_ACP2018/data/'
outdir       ='/ice3/hao/TTL/code&data_ACP2018/figures/'

mls_file        = fdir+'MLS-Aura_L2GP-H2O_v04_2004_2016.nc'
traj_erai_file  = fdir+'isob_traj_s100_i370_ERAi_200408_201612_6hrly_forward_mls_ak.nc'
traj_mer2_file  = fdir+'isob_traj_s100_i370_merra2_200408_2016_6hrly_forward_mls_ak.nc'

## read water vapor from MLS, trajectory runs
h2o_mls  = read_mls(time_lim,lat_lim,pr_lim,mls_file,'h2o_mix')
h2o_erai = read_traj(time_lim,lat_lim,pr_lim,traj_erai_file,'isob_h2oM')
h2o_mer2 = read_traj(time_lim,lat_lim,pr_lim,traj_mer2_file,'isob_h2oM')

## calculate tropical average water vapor anomalies
h2o_mls_anom  = anom_calc(h2o_mls)['anom']
h2o_erai_anom = anom_calc(h2o_erai)['anom']
h2o_mer2_anom = anom_calc(h2o_mer2)['anom']
time = np.array([round(x,3) for x in h2o_mls['time']])

## plot time series of tropical average water vapor anomalies
fig,ax = plt.subplots(figsize=[8,3.5])
ax.plot(time,h2o_mls_anom,'k-',lw=1.5,label='MLS')
ax.plot(time,h2o_erai_anom,'b-',lw=1.5,label='traj_ERAi')
ax.plot(time,h2o_mer2_anom,'r-',lw=1.5,label='traj_MERRA2')
ax.legend(fontsize=9,loc='upper left')

tkw = dict(size=4, width=1.5)
ax.tick_params(axis='both',which ='major',labelsize=12,**tkw)                ## set axis tick parameters

ax.set_xlim(2004,2017)
ax.set_xticks(np.arange(2005,2018,2))
ax.set_xlabel('Year',fontsize=12)
ax.set_ylim(-0.8,.8)
ax.set_yticks([-0.8,-0.4,0,0.4,0.8])
ax.set_ylabel('H$_2$O anomaly (ppmv) ',fontsize=12)
ax.set_title('H$_2$O anomaly at 100 hPa',fontsize=14)
ax.grid()

plt.tight_layout()
plt.savefig(outdir+'tropical_h2o_anom.pdf')
