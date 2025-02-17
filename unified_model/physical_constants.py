import numpy as np
from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck18')

T0_CMB = np.float64(2.7255)
h_planck = np.float64(6.62607015e-34)
me = np.float64(9.1093837e-31)
c = np.float64(2.99792458e8)
kB = np.float64(1.380649e-23)
sigma_T = np.float64(6.65e-29)  # Thomson scattering cross section
mp = np.float64(1.67262192e-27)
sigma_SB = np.float64(5.670374419e-8)  # Stefan-Boltzmann constant

H0 = np.float64(cosmo.H0)
H0_s = np.float64(H0 / 3.086e19)  # Convert H0 from km/s/Mpc to s^(-1)
h_Hubble = np.float64(cosmo.h)

delta_c = np.float64(1.686)  # spherical collapse

Omega_m = np.float64(cosmo.Om0)
Omega_lambda = np.float64(1.0 - Omega_m)
Omega_r = np.float64(0.0)  # ignore cosmo.Or0
Omega_k = np.float64(0.0)

Omega_b = np.float64(cosmo.Ob0)
sigma8 = np.float64(cosmo.sigma8)
G_grav = np.float64(6.674e-11)

Msun = np.float64(1.988e30)
Mpc = np.float64(3.086e22)
Myr = np.float64(3.1536e13)  # seconds

Tvir_crit = np.float64(1e4)
mu = np.float64(1.23)  # mean molecular weight for neutral primordial gas

rho_crit_z0_kgm3 = 3.0 * H0_s**2 / (8.0 * np.pi * G_grav)
# convert form kg/m^3 to Msun/Mpc^3
rho_crit_z0 = rho_crit_z0_kgm3 * Mpc**3 / Msun
rho_m0 = Omega_m * rho_crit_z0  # Msun/Mpc^3
rho_b0 = Omega_b * rho_crit_z0  # Msun/Mpc^3

freefall_factor = np.float64(np.sqrt(3 * np.pi / 32))
