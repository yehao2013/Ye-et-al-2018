Code files introduction

1. multi_regress.py
  
   This file contains functions used in the calculations
   Functions included: multi_reg, read_traj, read_mls, read_reanalysis, read_geosccm, read_qbo_anom,
		       read_cld, anom_calc, cal_ccm_oni

2. read_pfister_conv.py

   Read convective cloud top theta and height, calculate and save cloud frequency

3. regress_geosccm_traj_coefs_calc.py
   
   Regress water vapor anomalies from GEOSCCM and trajectory model, save regression coefficients

4. regress_mls_traj_coefs_calc.py

   Regress water vapor anomalies from MLS and trajectory model, save regression coefficients

5. regress_traj_coefs_plot.py

   Plots regression coefficients from MLS, GEOSCCM, and trajectory model (Figs. 2-4)

6. tropical_h2o_anom.py

   Plots tropical average TTL water vapor anomalies from MLS, traj_ERAi, and traj_MER2

7. 
