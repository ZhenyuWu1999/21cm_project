
from physical_constants import *


def Vel_Virial_analytic_oldversion(M_vir_in_Msun, z):  #van den Bosch Lecture 11
    #M_vir in solar mass, return virial velocity in m/s    
    #Delta_vir = Overdensity_Virial(z)
    Delta_vir = 200
    V_vir = 163*1e3 * (M_vir_in_Msun/1e12*h_Hubble)**(1/3) * (Delta_vir/200)**(1/6) * Omega_m**(1/6) *(1+z)**(1/2)
    return V_vir #m/s

def get_Rvir_analytic(M_vir_in_Msun, z):
    #M_vir in solar mass, return virial radius 
    #use Crit200 definition
    #return R_vir in Mpc
    rho_halo = 200 * Omega_m* rho_crit_z0_kgm3 * (1+z)**3
    R_vir = (3*M_vir_in_Msun*Msun/(4*np.pi*rho_halo))**(1/3)
    R_vir_Mpc = R_vir/Mpc
    return R_vir_Mpc

def Vel_Virial_analytic(M_vir_in_Msun, z):
    #M_vir in solar mass, return virial velocity in m/s
    #use Crit200 definition
    #return V_vir in m/s

    rho_halo = 200 * Omega_m* rho_crit_z0_kgm3 * (1+z)**3
    R_vir = (3*M_vir_in_Msun*Msun/(4*np.pi*rho_halo))**(1/3)
    V_vir = np.sqrt(G_grav*M_vir_in_Msun*Msun/R_vir)
    return V_vir

def inversefunc_Vel_Virial_analytic(V_vir, z):
    #V_vir in m/s, return M_vir in solar mass
    #use Crit200 definition
    #return M_vir in solar mass
    rho_halo = 200 * Omega_m* rho_crit_z0_kgm3 * (1+z)**3
    R_vir = np.sqrt(V_vir**2/(G_grav*4.0/3.0*np.pi*rho_halo))
    M_vir = 4.0/3.0*np.pi*R_vir**3*rho_halo/Msun
    return M_vir

def Vel_Virial_numerical(M_vir_in_Msun, R_vir_Mpc):
    #M_vir in solar mass, R_vir in Mpc
    V_vir = np.sqrt(G_grav*M_vir_in_Msun*Msun/(R_vir_Mpc*Mpc))
    return V_vir #m/s    

def Temperature_Virial_analytic(M_vir_in_Msun,z,mean_molecular_weight=None):  #van den Bosch Lecture 15
    if mean_molecular_weight is None:
        mean_molecular_weight = mu
    halo_profile_factor = 3.0/2.0
    V_vir = Vel_Virial_analytic(M_vir_in_Msun, z)
    T_vir = halo_profile_factor* mean_molecular_weight*mp*V_vir**2 /kB/3
    return T_vir

def inversefunc_Temperature_Virial_analytic(T_vir, z, mean_molecular_weight=None):  #van den Bosch Lecture 15
    #T_vir in K, return M_vir in solar mass
    #use Crit200 definition
    if mean_molecular_weight is None:
        mean_molecular_weight = mu
    halo_profile_factor = 3.0/2.0
    V_vir = np.sqrt(3*T_vir*kB/(halo_profile_factor*mean_molecular_weight*mp))
    M_vir = inversefunc_Vel_Virial_analytic(V_vir, z)
    return M_vir

def Temperature_Virial_analytic_oldversion(M_vir_in_Msun,z):  #van den Bosch Lecture 15
    halo_profile_factor = 3.0/2.0
    V_vir = Vel_Virial_analytic_oldversion(M_vir_in_Msun, z)
    T_vir = halo_profile_factor* mu*mp*V_vir**2 /kB/3
    return T_vir
    
#only valid for ionized gas
# def Temperature_Virial(M,z): #van den Bosch Lecture 15
#     #M in solar mass, return virial temperature in K
#     return 3.6e5 * (Vel_Virial(M,z)/1e3/100)**2

def Temperature_Virial_numerical(M_vir_in_Msun, R_vir_Mpc, mean_molecular_weight=None):
    #M_vir in solar mass, R_vir in Mpc
    if mean_molecular_weight is None:
        mean_molecular_weight = mu
    halo_profile_factor = 3.0/2.0
    V_vir = Vel_Virial_numerical(M_vir_in_Msun, R_vir_Mpc)
    T_vir = halo_profile_factor* mean_molecular_weight*mp*V_vir**2 /kB/3
    return T_vir

def get_A_number(mPert,Cs,rSoft):
    A = G_grav*mPert/(Cs**2*rSoft)
    return A

def crossing_time(Cs,rSoft):
    tCross = rSoft/Cs
    return tCross

def get_gas_lognH_analytic(z):
    #assume 200 times critical density
    #return lognH in cm^-3
    rho  = 200 * rho_b0*(1+z)**3 *Msun/Mpc**3
    nH = rho*hydrogen_mass_fraction/mp #now nH is just for hydrogen, not for all particles
    # nH = rho/(mu*mp)   
    nH_cm3 = nH/1.0e6
    lognH = np.log10(nH_cm3)
    return lognH

def get_mass_density_analytic(z):
    #assume 200 times critical density, unit kg/m^3
    rho = 200 * rho_m0*(1+z)**3 *Msun/Mpc**3
    return rho

def get_gas_lognH_numerical(M_vir_in_Msun, R_vir_Mpc):
    #M_vir in solar mass, R_vir in Mpc
    #return lognH in cm^-3
    R_vir_m = R_vir_Mpc * Mpc
    Mgas = Omega_b/Omega_m * M_vir_in_Msun
    rho = Mgas*Msun/(4/3*np.pi*R_vir_m**3)
    nH = rho*hydrogen_mass_fraction/mp #now nH is just for hydrogen, not for all particles
    # nH = rho/(mu*mp)  
    nH_cm3 = nH/1.0e6
    lognH = np.log10(nH_cm3)
    return lognH


if __name__ == "__main__":
    # Test the functions
    z = 8
    lognH = get_gas_lognH_analytic(z)
    nH = 10**lognH
    mass_density = get_mass_density_analytic(z)
    print(f"z = {z}, lognH = {lognH}, nH = {nH} cm^-3, mass_density = {mass_density} kg/m^3")
    Mhalo = 1e9
    Tvir = Temperature_Virial_analytic(Mhalo, z, mean_molecular_weight=0.6)
    print(f"Virial Temperature for Mhalo = {Mhalo} Msun at z = {z} is {Tvir} K")