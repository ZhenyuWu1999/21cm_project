
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from colossus.cosmology import cosmology
cosmology.setCosmology('planck15')
from colossus.halo import concentration

# Convert RuntimeWarnings to exceptions
warnings.filterwarnings('error', category=RuntimeWarning)


from physical_constants import *
from linear_evolution import D_z

def generalized_NFW_profile(x, rho_s, alpha):
    '''
    parameters:
    x: r/r_s  (concentration: Rvir/r_s = c)
    rho_s: characteristic density
    alpha: slope
    '''
    return rho_s / x**alpha / (1 + x)**(3 - alpha)

def f_NFW(x):
    return np.log(1+x) - x/(1+x)

def f_core(x):
    return np.log(1+x) -x*(3*x+2)/2/(x+1)**2

def get_concentration(M_in_Msun, z, model_name):
    '''
    parameters:
    M: halo mass in Msun
    z: redshift
    model: concentration model, see colossus tutorial, e.g. 'bullock01', 'ludlow16', 'child18','diemer19','ishiyama21'
    '''
    if model_name != 'bullock01_Dekel':
        M = M_in_Msun * h_Hubble # in Msun/h
        c = concentration.concentration(M, '200c', z, model = model_name, range_return = False)
    elif model_name == 'bullock01_Dekel':
        M13 = M_in_Msun/1e13
        c = 9.0*M13**(-0.15)/(1+z)

    return c

    
def density_NFW_profile(x, M, z, concentration_model):
    '''
    parameters:
    x: r/r_s  (concentration: Rvir/r_s = c)
    M: halo mass in Msun
    z: redshift
    concentration_model: concentration model name for colossus
    '''
    #assume 200 times critical density, unit kg/m^3
    rho_vir = 200 * rho_m0*(1+z)**3 *Msun/Mpc**3
    concentration = get_concentration(M, z, concentration_model)
    rho_s = concentration**3 / f_NFW(concentration) / 3.0 #in unit of rho_vir
    alpha = 1.0
    return generalized_NFW_profile(x, rho_s, alpha) #in unit of rho_vir

def gasdensity_NFW_profile(x, M, z, concentration_model):
    '''
    parameters:
    x: r/r_s  (concentration: Rvir/r_s = c)
    M: halo mass in Msun
    z: redshift
    concentration_model: concentration model name for colossus
    '''
    concentration = get_concentration(M, z, concentration_model)
    rho_s = concentration**3 / f_NFW(concentration) / 3.0
    alpha = 1.0
    f_gas = Omega_b/Omega_m
    return f_gas * generalized_NFW_profile(x, rho_s, alpha) #in unit of rho_vir

def gasdensity_core_profile(x, M, z, concentration_model):
    '''
    parameters:
    x: r/r_s  (concentration: Rvir/r_s = c)
    M: halo mass in Msun
    z: redshift
    concentration_model: concentration model name for colossus
    '''
    concentration = get_concentration(M, z, concentration_model)
    rho_s = concentration**3 / f_core(concentration) / 3.0  
    alpha = 0.0
    f_gas = Omega_b/Omega_m
    return f_gas * generalized_NFW_profile(x, rho_s, alpha) #in unit of rho_vir

def Velvir_NFW_profile(x, M, z, concentration_model):  #in virial velocity unit
    '''
    parameters:
    x: r/r_s  (concentration: Rvir/r_s = c)
    M: halo mass in Msun
    z: redshift
    concentration_model: concentration model name for colossus
    '''
    concentration = get_concentration(M, z, concentration_model)
    r_Rvir = x / concentration
    return np.sqrt(f_NFW(x)/f_NFW(concentration)/r_Rvir) #in virial velocity unit


def compare_concentration_model():
    output_dir = '/home/zwu/21cm_project/unified_model/Analytic_results/halo_profile'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for model_name in concentration.models:
        print(model_name)

    M = 10**np.arange(5.0, 15.4, 0.1)
    z_list = [0, 3, 6, 8, 10, 12, 15]
    models_to_plot = ['bullock01','ludlow16', 'child18','diemer19','ishiyama21']
    for z in z_list:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        for model_name in models_to_plot:
            c, mask = concentration.concentration(M, '200c', z, model = model_name, range_return = True)
            plt.plot(M[mask], c[mask], label = model_name.replace('_', '\\_'))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('M200c(Msun/h)',fontsize=14)
        plt.ylabel('Concentration',fontsize=14)
        plt.legend()
        plt.tight_layout()
        filename = os.path.join(output_dir, f'concentration_model_z{z}.png')
        plt.savefig(filename, dpi=300)

def plot_density_and_velocity_profile(concentration_model):
    
    output_dir = '/home/zwu/21cm_project/unified_model/Analytic_results/halo_profile'
    #plot total mass density NFW profile, gas NFW profile, gas core profile
    filename = os.path.join(output_dir, f'density_NFW_profile_{concentration_model}.png')
    z_list = [12, 6, 0]
    M_list = [1.0e7, 1.0e10, 1.0e13]
    
    for iz, z in enumerate(z_list):
        fig, ax = plt.subplots(1, len(M_list), figsize=(24, 6), facecolor='white')
        for iM, M in enumerate(M_list):
            concentration = get_concentration(M, z, concentration_model)
            r_Rvir_list = np.logspace(-2, 0, 100)
            x_list = r_Rvir_list * concentration

            rho_tot = density_NFW_profile(x_list, M, z, concentration_model)
            rho_gas_NFW = gasdensity_NFW_profile(x_list, M, z, concentration_model)
            rho_gas_core = gasdensity_core_profile(x_list, M, z, concentration_model)
            ax[iM].plot(r_Rvir_list, rho_tot, label='total', color='blue', linestyle='--')
            ax[iM].plot(r_Rvir_list, rho_gas_NFW, label='gas NFW', color='red', linestyle='-')
            ax[iM].plot(r_Rvir_list, rho_gas_core, label='gas core', color='green', linestyle='-')
            ax[iM].set_xscale('log')
            ax[iM].set_yscale('log')
            ax[iM].set_xlabel('r/Rvir', fontsize=14)
            ax[iM].set_ylabel('rho/rho_vir', fontsize=14)
            ax[iM].set_title(f'M={M:.1e}Msun, z={z}', fontsize=14)
            ax[iM].legend()
        plt.savefig(filename.replace('.png', f'_z{z}.png'))
    
    #plot velocity profile
    filename = os.path.join(output_dir, f'velocity_NFW_profile_{concentration_model}.png')
    z_list = [12, 6, 0]
    M_list = [1.0e7, 1.0e10, 1.0e13]

    for iz, z in enumerate(z_list):
        fig, ax = plt.subplots(1, len(M_list), figsize=(24, 6), facecolor='white')
        for iM, M in enumerate(M_list):
            concentration = get_concentration(M, z, concentration_model)
            r_Rvir_list = np.logspace(-2, 0, 100)
            x_list = r_Rvir_list * concentration

            Velvir = Velvir_NFW_profile(x_list, M, z, concentration_model)
            ax[iM].plot(r_Rvir_list, Velvir, label='velocity', color='blue', linestyle='--')
            ax[iM].set_xscale('log')
            ax[iM].set_yscale('log')
            ax[iM].set_xlabel('r/Rvir', fontsize=14)
            ax[iM].set_ylabel('velocity/virial velocity', fontsize=14)
            ax[iM].set_title(f'M={M:.1e}Msun, z={z}', fontsize=14)
            ax[iM].legend()
        plt.savefig(filename.replace('.png', f'_z{z}.png'))



"""
def compare_cooling_heating_profile(output_dir, Mvir, z, lgx_min, lgx_max, SHMF_model, concentration_model):
    '''
    Parameters:
    output_dir: output directory
    Mvir: halo mass in Msun/h
    z: redshift
    lgx_min: minimum subhalo mass ratio
    lgx_max: maximum subhalo mass ratio
    SHMF_model: SHMF model name
    concentration_model: concentration model name
    '''
    Mvir_in_Msun = Mvir/h_Hubble
    rho_vir = get_mass_density_analytic(z)
    R_vir = (3*Mvir_in_Msun*Msun/(4*np.pi*rho_vir))**(1/3)
    R_vir_cm = R_vir * (1e2)
    T_vir = Temperature_Virial_analytic(Mvir_in_Msun, z)

    r_Rvir_list = np.logspace(-2, 0, 50)
  
    concentration = get_concentration(Mvir_in_Msun, z, concentration_model) 
    x_list = r_Rvir_list * concentration 

    #assume isothermal temperature profile

    #cooling
    param_sets = [
        {'gas_metallicity': 1.0e-2, 'f_H2': 0.0},
        {'gas_metallicity': 1.0e-6, 'f_H2': 0.0}
    ]

    tot_cooling_rate_list = get_EqCooling_for_single_host(Mvir, z, param_sets)

    print("Total cooling rate Z=1e-2: ",tot_cooling_rate_list[0], "erg/s")
    print("Total cooling rate Z=1e-6: ",tot_cooling_rate_list[1], "erg/s")

    avg_cooling_density_Z2 = tot_cooling_rate_list[0]/ (4/3*np.pi*R_vir_cm**3)
    avg_cooling_density_Z6 = tot_cooling_rate_list[1] / (4/3*np.pi*R_vir_cm**3)

    print("Average cooling density Z=1e-2: ",avg_cooling_density_Z2, "erg/s/cm^3")
    print("Average cooling density Z=1e-6: ",avg_cooling_density_Z6, "erg/s/cm^3")


    cooling_NFW_Z2_list = []
    cooling_NFW_Z6_list = []
    cooling_core_Z2_list = []
    cooling_core_Z6_list = []
    all_cooling_Eq_NFW_results = all_cooling_Eq_core_results = []
    for r_Rvir in r_Rvir_list:
        all_cooling_Eq_NFW_results, all_cooling_Eq_core_results = get_EqCoolingDensity(r_Rvir, Mvir, z, concentration_model, param_sets)
        cooling_NFW_Z2_list.append(all_cooling_Eq_NFW_results[0])
        cooling_NFW_Z6_list.append(all_cooling_Eq_NFW_results[1])
        cooling_core_Z2_list.append(all_cooling_Eq_core_results[0])
        cooling_core_Z6_list.append(all_cooling_Eq_core_results[1])


    cooling_core_Z2_list = np.array(cooling_core_Z2_list).flatten()
    cooling_core_Z6_list = np.array(cooling_core_Z6_list).flatten()
    cooling_NFW_Z2_list = np.array(cooling_NFW_Z2_list).flatten()
    cooling_NFW_Z6_list = np.array(cooling_NFW_Z6_list).flatten()
    

    #heating
    lgM = np.log10(Mvir) #lgM in Msun/h
    singlehost_heating = integrate_SHMF_heating_for_single_host(z, lgx_min, lgx_max, lgM, SHMF_model)
    singlehost_heating_erg = singlehost_heating * 1e7
    singlehost_heating_density = singlehost_heating_erg / (4/3*np.pi*R_vir_cm**3) #erg/s/cm^3

    #debug
    # local_velocity = Velvir_NFW_profile(x_list, Mvir_in_Msun, z, concentration_model) 
    local_velocity = 1.0
    local_mass_density = density_NFW_profile(x_list, Mvir_in_Msun, z, concentration_model)
    local_gas_NFW_density = gasdensity_NFW_profile(x_list, Mvir_in_Msun, z, concentration_model)
    local_gas_core_density = gasdensity_core_profile(x_list, Mvir_in_Msun, z, concentration_model)

    heating_NFW_list = singlehost_heating_density/(Omega_b/Omega_m)*local_gas_NFW_density/local_velocity
    heating_core_list = singlehost_heating_density/(Omega_b/Omega_m)*local_gas_core_density/local_velocity
    
    heating_NFW_modified_list = heating_NFW_list*local_mass_density
    heating_core_modified_list = heating_core_list*local_mass_density

    #plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fig, ax1 = plt.subplots(figsize=(8, 6), facecolor='white')
    #cooling
    ax1.plot(r_Rvir_list,-cooling_NFW_Z2_list,'g--',label='Cooling NFW Z=1e-2')
    ax1.plot(r_Rvir_list,-cooling_NFW_Z6_list,'b--',label='Cooling NFW Z=1e-6')
    ax1.plot(r_Rvir_list,-cooling_core_Z2_list,'g-',label='Cooling core Z=1e-2')
    ax1.plot(r_Rvir_list,-cooling_core_Z6_list,'b-',label='Cooling core Z=1e-6')
    #heating
    ax1.plot(r_Rvir_list,heating_NFW_list,'r--',label='Heating NFW, const n_sub(r)')
    ax1.plot(r_Rvir_list,heating_core_list,'r-',label='Heating core, const n_sub(r)')
    ax1.plot(r_Rvir_list,heating_NFW_modified_list,'m--',label='Heating NFW, n_sub(r) ~ rho(r)')
    ax1.plot(r_Rvir_list,heating_core_modified_list,'m-',label='Heating core, n_sub(r) ~ rho(r)')
    #average cooling and heating density
    ax1.axhline(avg_cooling_density_Z2, color='g', linestyle='-',alpha=0.5,label='Avg Cooling Z=1e-2')
    ax1.axhline(avg_cooling_density_Z6, color='b', linestyle='-',alpha=0.5,label='Avg Cooling Z=1e-6')
    ax1.axhline(singlehost_heating_density, color='r', linestyle='-',alpha=0.5,label='Avg Heating')

    ax1.legend()
    ax1.set_xlim([min(r_Rvir_list),max(r_Rvir_list)])
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylabel(r'Cooling and Heating [erg/s/cm$^3$]',fontsize=14)
    ax1.set_xlabel(r'r/R$_{vir}$',fontsize=14)
    ax1.tick_params(axis='both', direction='in')

    #debug: no velocity correction
    filename = os.path.join(output_dir,f"cooling_heating_profile_lgM{lgM:.2f}_z{z:.2f}_muionized_novelcorrection.png")
    plt.savefig(filename,dpi=300)
"""

"""
def plot_cooling_heating_at_r_Rvir(min_lgM, max_lgM, r_Rvir, redshift, SHMF_model, concentration_model):
    #plot cooling and heating at r/Rvir = 1
    #min_lgM, max_lgM in Msun/h
    #redshift: redshift
    #SHMF_model: SHMF model name
    #concentration_model: concentration model name
    output_dir = '/home/zwu/21cm_project/unified_model/Analytic_profile_results/cooling_heating_ratio'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    z = redshift
    lgM_list = np.linspace(min_lgM, max_lgM, 20)
    Mvir_list = 10**lgM_list
    Mvir_in_Msun_list = Mvir_list/h_Hubble
    rho_vir = get_mass_density_analytic(redshift)
    R_vir_list = (3*Mvir_in_Msun_list*Msun/(4*np.pi*rho_vir))**(1/3)
    R_vir_cm_list = R_vir_list * (1e2)
    T_vir = Temperature_Virial_analytic(Mvir_in_Msun_list, redshift)

    concentration_list = np.array([get_concentration(Mvir_in_Msun, redshift, concentration_model) for Mvir_in_Msun in Mvir_in_Msun_list])
    
    x_list = r_Rvir * concentration_list

    ratio_NFW_Z2_list = []
    ratio_NFW_Z6_list = []
    ratio_core_Z2_list = []
    ratio_core_Z6_list = []

    param_sets = [
        {'gas_metallicity': 1.0e-2, 'f_H2': 0.0},
        {'gas_metallicity': 1.0e-6, 'f_H2': 0.0}
    ]
    for i, lgM in enumerate(lgM_list):
        Mvir = 10**lgM
        Mvir_in_Msun = Mvir/h_Hubble
        all_cooling_Eq_NFW_results, all_cooling_Eq_core_results = get_EqCoolingDensity(r_Rvir, Mvir, redshift, concentration_model, param_sets)
        cooling_NFW_Z2 = all_cooling_Eq_NFW_results[0]
        cooling_NFW_Z6 = all_cooling_Eq_NFW_results[1]
        cooling_core_Z2 = all_cooling_Eq_core_results[0]
        cooling_core_Z6 = all_cooling_Eq_core_results[1]

        singlehost_heating = integrate_SHMF_heating_for_single_host(z, -3, -1, lgM, SHMF_model)
        singlehost_heating_erg = singlehost_heating * 1e7
        singlehost_heating_density = singlehost_heating_erg / (4/3*np.pi*R_vir_cm_list[i]**3) #erg/s/cm^3

        local_velocity = 1.0
        local_mass_density = density_NFW_profile(x_list[i], Mvir_in_Msun, z, concentration_model)
        local_gas_NFW_density = gasdensity_NFW_profile(x_list[i], Mvir_in_Msun, z, concentration_model)
        local_gas_core_density = gasdensity_core_profile(x_list[i], Mvir_in_Msun, z, concentration_model)

        heating_NFW = singlehost_heating_density/(Omega_b/Omega_m)*local_gas_NFW_density/local_velocity
        heating_core = singlehost_heating_density/(Omega_b/Omega_m)*local_gas_core_density/local_velocity
        
        heating_NFW_modified = heating_NFW*local_mass_density
        heating_core_modified = heating_core*local_mass_density

        ratio_NFW_Z2 = -heating_NFW_modified/cooling_NFW_Z2
        ratio_NFW_Z6 = -heating_NFW_modified/cooling_NFW_Z6
        ratio_core_Z2 = -heating_core_modified/cooling_core_Z2
        ratio_core_Z6 = -heating_core_modified/cooling_core_Z6

        ratio_NFW_Z2_list.append(ratio_NFW_Z2)
        ratio_NFW_Z6_list.append(ratio_NFW_Z6)
        ratio_core_Z2_list.append(ratio_core_Z2)
        ratio_core_Z6_list.append(ratio_core_Z6)

            
    #plot ratio
    fig, ax1 = plt.subplots(figsize=(8, 6), facecolor='white')

    ax1.plot(lgM_list, ratio_NFW_Z2_list, 'g--', label='Cooling NFW Z=1e-2')
    ax1.plot(lgM_list, ratio_NFW_Z6_list, 'b--', label='Cooling NFW Z=1e-6')
    ax1.plot(lgM_list, ratio_core_Z2_list, 'g-', label='Cooling core Z=1e-2')
    ax1.plot(lgM_list, ratio_core_Z6_list, 'b-', label='Cooling core Z=1e-6')
    ax1.set_xlabel(r'log$_{10}$(M$_{vir}$/M$_{\odot}$)', fontsize=14)
    ax1.set_ylabel(r'Heating/Cooling Ratio', fontsize=14)
    ax1.set_yscale('log')
    ax1.legend()
    filename = os.path.join(output_dir,f"cooling_heating_ratio_rRvir{r_Rvir:.2f}_z{redshift:.2f}.png")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
"""

if __name__ == "__main__":


    # compare_concentration_model()
    # plot_density_and_velocity_profile('bullock01')
    # plot_density_and_velocity_profile('ludlow16')
    plot_density_and_velocity_profile('diemer19')




