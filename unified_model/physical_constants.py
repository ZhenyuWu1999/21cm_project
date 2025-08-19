import numpy as np
from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck15') # Planck 2015 cosmology as used in TNG

T0_CMB = np.float64(2.7255)
h_planck = np.float64(6.62607015e-34)
me = np.float64(9.1093837e-31)
c_light = np.float64(2.99792458e8)
kB = np.float64(1.380649e-23)
sigma_T = np.float64(6.65e-29)  # Thomson scattering cross section
mp = np.float64(1.67262192e-27)
sigma_SB = np.float64(5.670374419e-8)  # Stefan-Boltzmann constant
eV = np.float64(1.60217662e-19) # electronvolt in Joules

H0 = np.float64(cosmo.H0)
H0_s = np.float64(H0 / 3.086e19)  # Convert H0 from km/s/Mpc to s^(-1)
h_Hubble = np.float64(cosmo.h)

delta_c = np.float64(1.686)  # spherical collapse

Omega_m = np.float64(cosmo.Om0)
Omega_lambda = np.float64(cosmo.Ode0)
Omega_r = np.float64(0.0)  # ignore cosmo.Or0
Omega_k = np.float64(0.0)
Omega_b = np.float64(cosmo.Ob0)
Omega_c = Omega_m - Omega_b  # cold dark matter
Ombh2 = Omega_b * h_Hubble**2  
Omch2 = Omega_c * h_Hubble**2

#table 4 of https://escholarship.org/content/qt9hz5p0hv/qt9hz5p0hv.pdf
#initial power spectrum parameters
ns = np.float64(cosmo.ns)
As = np.float64(2.142e-9)
tau = np.float64(0.066)
sigma8 = np.float64(cosmo.sigma8)

G_grav = np.float64(6.674e-11)
Msun = np.float64(1.988e30)
Mpc = np.float64(3.086e22)
kpc = np.float64(3.086e19)
Myr = np.float64(3.1536e13)  # seconds
Gyr = Myr * 1e3  # seconds
Zsun = 0.01295 # solar metallicity, grackle default
#Zsun = 0.0127 # solar metallicity, according to TNG data specifications

Tvir_crit = np.float64(1e4)
mu_neutral = np.float64(1.22)  # mean molecular weight for neutral primordial gas
mu = np.float64(0.6) # mean molecular weight for ionized gas
mu_minihalo = np.float64(1.22) # mean molecular weight for gas cloud ~200K - 1e4K
hydrogen_mass_fraction = np.float64(0.76)  # hydrogen mass fraction in primordial gas

rho_crit_z0_kgm3 = 3.0 * H0_s**2 / (8.0 * np.pi * G_grav)
# convert form kg/m^3 to Msun/Mpc^3
rho_crit_z0 = rho_crit_z0_kgm3 * Mpc**3 / Msun
rho_m0 = Omega_m * rho_crit_z0  # Msun/Mpc^3
rho_b0 = Omega_b * rho_crit_z0  # Msun/Mpc^3

freefall_factor = np.float64(np.sqrt(3 * np.pi / 32))

a_radiation = 4*sigma_SB/c_light
# t_gamma_inv = 8.55e-13 / 3.1536e7  # Compton scattering timescale inverse, Convert t_gamma_inv from yr^(-1) to s^(-1)
t_gamma_inv = 8.0 *a_radiation*T0_CMB**4 * sigma_T /(3.0*me*c_light)

if __name__ == "__main__":
    print("H0:", H0)
    print("Omega_lambda:", cosmo.Ode0)
    print("Omega_b:", Omega_b)
    print(Omega_b/Omega_m)
    print("sigma8:", sigma8)

    print("1/H0:", 1./H0_s/Myr, "Myr")
    print("Ombh2:", Ombh2)
    print("Omch2:", Omch2)
    print(cosmo.ns)
    print(f"age of the universe: {cosmology.Cosmology.age(cosmo, z = 17)} billion years")

    print(Omega_m)
    print(Omega_b)
    print(H0)
