import numpy as np
SHMF_model = 'BestFit_z'
snapNum = 2
simulation_set = 'TNG50-1'

#bestfit parameters for ratio of selected HMF to total HMF (a,b,c,d,e,f), used in HMF_ratio_2Dbestfit()
hmf_ratio_params = [8.3729, 0.5120, -0.0197, 6.7591, 8.0099, 6.1885]
# hmf_ratio_params_useM200 = [8.3307, 0.5627, -0.0155, 23.1858, 0.0000, 0.0000]

#Jiang & van den Bosch 2016 SHMF parameters
p_evolved = [0.86, 50/np.log(10), 4, np.log10(0.065)]
p_unevolved = [0.91, 6/np.log(10), 3, np.log10(0.22*np.log(10))]

#redshift dependence of BestFit SHMF parameters 
#(see results in analysis/SHMF_redshift_evolution_alpha_lgA.png and
#analysis/SHMF_redshift_evolution_omega_beta.png)
alpha_z_params = [-0.02, 0.82] #alpha = alpha_z_params[0]*z + alpha_z_params[1]
lgA_z_params = [0.06, -1.11] #lgA = lgA_z_params[0]*z + lgA_z_params[1]
omega_z_params = [3.71, -0.80]  #[c, m]: omega = c for z >= 6, m*z + b for z < 6 (b+6*m = c)
lnbeta_z_params = [4.14, -0.61] #[c, m]: ln(beta) = c for z >= 6, m*z + b for z < 6 (b+6*m = c)
# beta_ln10_z = beta_z/np.log(10)

#Kim 2005 heating rate for massive cluster (mass [Msun], heating rate[erg/s])
Kim2005_result = np.array([6.6e14, 4.0e44])

