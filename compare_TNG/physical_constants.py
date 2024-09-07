import numpy as np
from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck15')

T0_CMB = 2.7255
h_planck = 6.62607015e-34
me = 9.1093837e-31
c = 2.99792458e8
kB = 1.380649e-23
sigma_T = 6.65e-29  #Thomson scattering cross section
mp = 1.67262192e-27
sigma_SB = 5.670374419e-8  #Stefan-Boltzmann constant

H0 = cosmo.H0
H0_s = H0 / 3.086e19  # Convert H0 from km/s/Mpc to s^(-1)
h_Hubble = cosmo.h


delta_c = 1.686 #spherical collapse

#cosmological parameters
Omega_m = cosmo.Om0
Omega_lambda = 1 - Omega_m
Omega_r = 0  #ignore cosmo.Or0
Omega_k = 0

Omega_b = cosmo.Ob0
sigma8 = cosmo.sigma8
G_grav = 6.674e-11

Msun = 1.988e30
Mpc = 3.086e22

#Furlanetto 2006; Barkana 2001, Tvir = 10^4 K
Tvir_crit = 1e4
mu = 1.22 #mean molecular weight for neutral primordial gas  !invalid for ionised gas

#Power_Spectrum
rho_crit_z0_kgm3 = 3*H0_s**2/(8*np.pi*G_grav)
#convert form kg/m^3 to Msun/Mpc^3
rho_crit_z0 = rho_crit_z0_kgm3 * Mpc**3/Msun
rho_m0 = Omega_m*rho_crit_z0  #Msun/Mpc^3
rho_b0 = Omega_b*rho_crit_z0  #Msun/Mpc^3