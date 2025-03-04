
from physical_constants import *

def Vel_Virial_analytic(M_vir_in_Msun, z):  #van den Bosch Lecture 11
    #M_vir in solar mass, return virial velocity in m/s    
    #Delta_vir = Overdensity_Virial(z)
    Delta_vir = 200
    V_vir = 163*1e3 * (M_vir_in_Msun/1e12*h_Hubble)**(1/3) * (Delta_vir/200)**(1/6) * Omega_m**(1/6) *(1+z)**(1/2)
    return V_vir #m/s

def Vel_Virial_numerical(M_vir_in_Msun, R_vir_Mpc):
    #M_vir in solar mass, R_vir in Mpc
    V_vir = np.sqrt(G_grav*M_vir_in_Msun*Msun/(R_vir_Mpc*Mpc))
    return V_vir #m/s    

def Temperature_Virial_analytic(M_vir_in_Msun,z):  #van den Bosch Lecture 15
    halo_profile_factor = 3.0/2.0
    V_vir = Vel_Virial_analytic(M_vir_in_Msun, z)
    T_vir = halo_profile_factor* mu*mp*V_vir**2 /kB/3
    return T_vir
    
#only valid for ionized gas
# def Temperature_Virial(M,z): #van den Bosch Lecture 15
#     #M in solar mass, return virial temperature in K
#     return 3.6e5 * (Vel_Virial(M,z)/1e3/100)**2

def Temperature_Virial_numerical(M_vir_in_Msun, R_vir_Mpc):
    #M_vir in solar mass, R_vir in Mpc
    halo_profile_factor = 3.0/2.0
    V_vir = Vel_Virial_numerical(M_vir_in_Msun, R_vir_Mpc)
    T_vir = halo_profile_factor* mu*mp*V_vir**2 /kB/3
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
    nH = rho/mp
    nH_cm3 = nH/1.0e6
    lognH = np.log10(nH_cm3)
    return lognH

def get_gas_lognH_numerical(M_vir_in_Msun, R_vir_Mpc):
    #M_vir in solar mass, R_vir in Mpc
    #return lognH in cm^-3
    R_vir_m = R_vir_Mpc * Mpc
    Mgas = Omega_b/Omega_m * M_vir_in_Msun
    rho = Mgas*Msun/(4/3*np.pi*R_vir_m**3)
    nH = rho/mp
    nH_cm3 = nH/1.0e6
    lognH = np.log10(nH_cm3)
    return lognH