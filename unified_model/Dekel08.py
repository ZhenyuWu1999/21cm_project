#Dekel 2008 analytic model on gravitational quenching in massive clusters

import numpy as np
import matplotlib.pyplot as plt
from physical_constants import *
import HaloProperties
from Analytic_halo_profile import f_NFW, f_core, get_concentration
import os
concentration_model = 'ludlow16'

def get_Dekel08_A(z):

    a = 1/(1+z)
    Omega_m_z = Omega_m*(1+z)**3/(Omega_lambda + Omega_m*(1+z)**3)
    Omega_lambda_z = 1 - Omega_m_z
    Delta_z = (18*np.pi**2 - 82*Omega_lambda_z - 39*Omega_lambda_z**2)/Omega_m_z

    A = (Delta_z/200 * Omega_m/0.3 * (h_Hubble/0.7)**2)**(-1/3)/(1+z)
    return A

def get_halo_mass_density(z):
    #halo mass density in g/cm^3
    A = get_Dekel08_A(z)
    return 5.52e-28 * A**(-3) 

def get_Velvirial(M, z):
    #M in Msun
    #return Virial velocity in km/s
    M13 = M/1e13
    A = get_Dekel08_A(z)
    V300 = (M13/1.64/A**(3/2))**(1/3)
    Vvir = V300 * 300
    return Vvir

def get_concentration_Bullock01(M , z):
    #M in Msun
    M13 = M/1e13
    return 9.0*M13**(-0.15)/(1+z)

def get_T6_approx(M, z):
    #Tvir in 10^6 K
    M13 = M/1e13
    A = get_Dekel08_A(z)
    T6 = 2.33*M13**(2/3)*A**(-1)
    return T6

def get_mean_metallicity_DB06(z):
    #Dekel & Birnboim (2006)
    #return Z in Zsun
    s = 0.17
    return 10**(-s*z) * 0.3


def compare_Dekel08():
    z = 0
    print('z = ', z)
    print('halo density = ', get_halo_mass_density(z), ' g/cm^3')
    print(HaloProperties.get_mass_density_analytic(z)/1e3)
    M = 1e13
    print('M = ', M)
    print('Vvir = ', get_Velvirial(M, z), ' km/s')
    print(HaloProperties.Vel_Virial_analytic(M, z)/1e3)

    lgM_list = np.arange(11, 15.1, 0.1)
    T6_list = np.array([get_T6_approx(10**lgM, z) for lgM in lgM_list])
    Tvir_list = np.array([HaloProperties.Temperature_Virial_analytic(10**lgM, z) for lgM in lgM_list])

    fig = plt.figure(figsize=(6, 6), facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(lgM_list, 1e6*T6_list, label='T6_approx')
    ax.plot(lgM_list, Tvir_list, label='Tvir_analytic')
    ax.set_xlabel('lgM [Msun]', fontsize=14)
    ax.set_ylabel('T [K]', fontsize=14)
    ax.set_yscale('log')
    ax.grid()
    ax.legend()
    plt.tight_layout()
    output_dir = '/home/zwu/21cm_project/unified_model/Analytic_results/Dekel08_comparison'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, 'Tvir_vs_T6_approx.png')
    plt.savefig(filename, dpi=300)






def get_heating_Dekel08(M, z, fc):
    '''
    parameters:
    M: halo mass in Msun
    z: redshift
    fc: fraction of gas clumps in accretion flow
    fg: fraction of hot gas in halo
    return: Edot_heating in erg/s
    '''
    Vvir_kms = get_Velvirial(M, z)
    # concentration = get_concentration_Bullock01(M, z)
    #debug
    concentration = get_concentration(M, z, concentration_model)

    r_Rvir = 0.1 #0.1Rvir
    x = r_Rvir / concentration
    phi_hat_r = concentration/f_NFW(concentration) * (np.log(1+x)/x - np.log(1+concentration)/concentration)
    M13 = M/1e13
    a = 1/(1+z)
    heating_rate = 1/2*(M*Msun)*(Vvir_kms*1e3)**2 * 0.11*phi_hat_r*fc*M13**0.15*a**(-2.25)/Gyr
    heating_rate_erg = heating_rate * 1e7 #convert to erg/s
    return heating_rate_erg

def get_cooling_Dekel08(M, z, fg):
    # concentration = get_concentration_Bullock01(M, z)
    concentration = get_concentration(M, z, concentration_model)
    C9 = concentration/9.0
    A0_C = f_core(concentration)
    M13 = M/1e13
    A = get_Dekel08_A(z)
    T6 = get_T6_approx(M, z)
    Z = get_mean_metallicity_DB06(z)
    Lambda23 = 6.0*(Z/0.3)**0.7 * T6**(-1) + 0.2*T6**(1/2)  #Lambda in 1e-23 erg/s*cm^3
    Vvir_kms = get_Velvirial(M, z)
    cooling_rate = 1/2*(fg*M*Msun)*(Vvir_kms*1e3)**2 \
                *0.061*C9**3/A0_C*(fg/0.05)*M13**(-2/3)*Lambda23/A**2/Gyr
    cooling_rate_erg = cooling_rate * 1e7 #convert to erg/s
    return cooling_rate_erg


def get_cooling_Dekel08_approx(M, z, fg):
    #return cooling rate in erg/s
    M13 = M/1e13
    A = get_Dekel08_A(z)
    T6 = get_T6_approx(M, z)
    Z = get_mean_metallicity_DB06(z)
    Lambda23 = 6.0*(Z/0.3)**0.7 * T6**(-1) + 0.2*T6**(1/2)  #Lambda in 1e-23 erg/s*cm^3

    cooling_rate = 6.2e41 * (fg/0.05)**2 * M13 * Lambda23 *A**(-3)
    return cooling_rate

def get_heating_cooling_ratio_Dekel08(M, z, fc, fg):

    # concentration = get_concentration_Bullock01(M, z)
    concentration = get_concentration(M, z, concentration_model)
    A0_C = f_core(concentration)
    C9 = concentration/9.0
    
    M13 = M/1e13
    A = get_Dekel08_A(z)

    r_Rvir = 0.1 #0.1Rvir
    x = r_Rvir / concentration
    phi_hat_r = concentration/f_NFW(concentration) * (np.log(1+x)/x - np.log(1+concentration)/concentration)

    T6 = get_T6_approx(M, z)
    Z = get_mean_metallicity_DB06(z)
    #debug
    Z = 0.3
    Lambda23 = 6.0*(Z/0.3)**0.7 * T6**(-1) + 0.2*T6**(1/2)  #Lambda in 1e-23 erg/s*cm^3
    a = 1/(1+z)

    heating_cooling_ratio = 1.9*phi_hat_r*A0_C/C9**3 *fc/fg/(fg/0.05) *M13**(0.816667) /Lambda23*A**2/a**2.25
    return heating_cooling_ratio





def plot_heating_cooling_ratio_Dekel08():
    
    lgM_list = np.arange(11, 15.1, 0.1)
    All_results = []
    All_results_2 = []

    #params: different combinations of z, fc, fg
    #z = 0, fc = 0.05, fg = 0.05
    #z = 2, fc = 0.05, fg = 0.05
    #z = 2, fc = 0.025, fg = 0.075
    params = [(0, 0.05, 0.05), (2, 0.05, 0.05), (0, 0.025, 0.075)]

    for param in params:
        z, fc, fg = param
        print('z = ', z)
        print('fc = ', fc)
        print('fg = ', fg)
        heating_rate = []
        cooling_rate = []
        heating_cooling_ratio = []
        for lgM in lgM_list:
            M = 10**lgM
            heating_rate.append(get_heating_Dekel08(M, z, fc))
            cooling_rate.append(get_cooling_Dekel08(M, z, fg))
            heating_cooling_ratio.append(get_heating_cooling_ratio_Dekel08(M, z, fc, fg))
        heating_rate = np.array(heating_rate)
        cooling_rate = np.array(cooling_rate)
        ratio = heating_rate/cooling_rate
        heating_cooling_ratio = np.array(heating_cooling_ratio)
        All_results.append(ratio)
        All_results_2.append(heating_cooling_ratio)
    
    colors = ['blue', 'green', 'red']
    linestyles = ['--', '--', '--']

    labels = ['z=0, fc=0.05, fg=0.05', 'z=2, fc=0.05, fg=0.05', 'z=0, fc=0.025, fg=0.075']
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    for i in range(len(All_results)):
        result = All_results[i]
        ax.plot(lgM_list, np.log10(result), color=colors[i], linestyle=linestyles[i], label=labels[i])
        result2 = All_results_2[i]
        ax.plot(lgM_list, np.log10(result2), color=colors[i], linestyle='-')
    ax.axhline(np.log10(1), color='black', linestyle='--')
    ax.set_xlabel('lgM [Msun]', fontsize=14)
    ax.set_ylabel('lg(heating/cooling)', fontsize=14)
    ax.set_ylim(-3, 3)
    ax.grid()
    ax.set_title('Dekel08 heating/cooling ratio', fontsize=16)
    ax.legend()
    plt.tight_layout()
    output_dir = '/home/zwu/21cm_project/unified_model/Analytic_results/Dekel08_comparison'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, 'Dekel08_heating_cooling_ratio.png')
    plt.savefig(filename, dpi=300)

    result0 = All_results[0]
    result1 = All_results[1]
    result2 = All_results[2]

    print(result0/result2)


    


if __name__ == "__main__":
    compare_Dekel08()
    # plot_heating_cooling_ratio_Dekel08()
   
