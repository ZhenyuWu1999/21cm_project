
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

def get_concentration(M_in_Msun, z, model_name):
    '''
    parameters:
    M: halo mass in Msun
    z: redshift
    model: concentration model, see colossus tutorial, e.g. 'bullock01', 'ludlow16', 'child18','diemer19','ishiyama21'
    '''
    M = M_in_Msun * h_Hubble # in Msun/h
    c = concentration.concentration(M, '200c', z, model = model_name, range_return = False)
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
    rho_s = concentration**3 / f_NFW(concentration) / 3.0
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
    #z_list = [0, 3, 6, 8, 10, 12, 15]
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



if __name__ == "__main__":


    # compare_concentration_model()
    # plot_density_and_velocity_profile('bullock01')
    # plot_density_and_velocity_profile('ludlow16')
    plot_density_and_velocity_profile('diemer19')




