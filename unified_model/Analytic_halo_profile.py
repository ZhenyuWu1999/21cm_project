
import numpy as np
import matplotlib.pyplot as plt
import os

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

def get_concentration_Bullock01(M, z):
    '''
    parameters:
    M: halo mass in Msun
    z: redshift
    '''
    return 9.0 * (M/1.0e13)**(-0.15) * (1+z)**(-1)

def sigma_M_z_approx(M, z):
    #M in Msun
    ksi = (M/1.0e10*h_Hubble)**(-1)
    return D_z(z) * 22.26*ksi**(0.292)/(1 + 1.53*ksi**(0.275) + 3.36*ksi**(0.198))

def get_concentration_Ludlow16(M, z):
    '''
    parameters:
    M: halo mass in Msun
    z: redshift
    '''
    sigma_M_z = sigma_M_z_approx(M, z)
    delta_sc = 1.686
    nu = delta_sc / sigma_M_z
    a = (1+z)**(-1)
    nu_0 = (4.135 - 0.564/a - 0.210/a**2 + 0.0557/a**3 - 0.00348/a**4) / D_z(z)

    c_0 = 3.395*(1+z)**(-0.215)
    beta = 0.307*(1+z)**(0.540)
    gamma_1 = 0.628*(1+z)**(-0.047)
    gamma_2 = 0.317*(1+z)**(-0.893)
    c = c_0 * (nu/nu_0)**(-gamma_1) * (1 + (nu/nu_0)**(1/beta))**(-beta*(gamma_2 - gamma_1))
    return c

def density_NFW_profile(x, M, z):
    '''
    parameters:
    x: r/r_s  (concentration: Rvir/r_s = c)
    M: halo mass in Msun
    z: redshift
    '''
    #assume 200 times critical density, unit kg/m^3
    rho_vir = 200 * rho_m0*(1+z)**3 *Msun/Mpc**3
    concentration = get_concentration_Bullock01(M, z)
    rho_s = concentration**3 / f_NFW(concentration) / 3.0 #in unit of rho_vir
    alpha = 1.0
    return generalized_NFW_profile(x, rho_s, alpha) #in unit of rho_vir

def gasdensity_NFW_profile(x, M, z):
    '''
    parameters:
    x: r/r_s  (concentration: Rvir/r_s = c)
    M: halo mass in Msun
    z: redshift
    '''
    concentration = get_concentration_Bullock01(M, z)
    rho_s = concentration**3 / f_NFW(concentration) / 3.0
    alpha = 1.0
    f_gas = Omega_b/Omega_m
    return f_gas * generalized_NFW_profile(x, rho_s, alpha) #in unit of rho_vir

def gasdensity_core_profile(x, M, z):
    '''
    parameters:
    x: r/r_s  (concentration: Rvir/r_s = c)
    M: halo mass in Msun
    z: redshift
    '''
    concentration = get_concentration_Bullock01(M, z)
    rho_s = concentration**3 / f_NFW(concentration) / 3.0
    alpha = 0.0
    f_gas = Omega_b/Omega_m
    return f_gas * generalized_NFW_profile(x, rho_s, alpha) #in unit of rho_vir

def Velvir_NFW_profile(x, M, z):  #in virial velocity unit
    '''
    parameters:
    x: r/r_s  (concentration: Rvir/r_s = c)
    M: halo mass in Msun
    z: redshift
    '''
    concentration = get_concentration_Bullock01(M, z)
    r_Rvir = x / concentration
    return np.sqrt(f_NFW(x)/f_NFW(concentration)/r_Rvir) #in virial velocity unit


if __name__ == "__main__":

    #plot Bullock01 concentration

    output_dir = '/home/zwu/21cm_project/unified_model/Analytic_results/halo_profile'
    filename = os.path.join(output_dir, 'concentration_Bullock01.png')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    z_list = [15, 12, 10, 6, 0]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(z_list)))
    M_list = np.logspace(5, 15, 100)
    fig, ax = plt.subplots(figsize=(8, 6),facecolor='white')
    for index, z in enumerate(z_list):
        c_list_Bullock01 = get_concentration_Bullock01(M_list, z)
        c_list_Ludlow16 = get_concentration_Ludlow16(M_list, z)
        ax.plot(M_list, c_list_Bullock01, label=f'z={z}, Bullock01', color=colors[index], linestyle='--')
        ax.plot(M_list, c_list_Ludlow16, label=f'z={z}, Ludlow16', color=colors[index], linestyle='-')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('M [Msun]', fontsize=14)
    ax.set_ylabel('concentration', fontsize=14)
    ax.legend()
    plt.savefig(filename)


    #plot total mass density NFW profile, gas NFW profile, gas core profile
    filename = os.path.join(output_dir, 'density_NFW_profile.png')
    z_list = [12, 6, 0]
    M_list = [1.0e7, 1.0e10, 1.0e13]
    
    for iz, z in enumerate(z_list):
        fig, ax = plt.subplots(1, len(M_list), figsize=(24, 6), facecolor='white')
        for iM, M in enumerate(M_list):
            concentration = get_concentration_Bullock01(M, z)
            r_Rvir_list = np.logspace(-2, 0, 100)
            x_list = r_Rvir_list * concentration

            rho_tot = density_NFW_profile(x_list, M, z)
            rho_gas_NFW = gasdensity_NFW_profile(x_list, M, z)
            rho_gas_core = gasdensity_core_profile(x_list, M, z)
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
    filename = os.path.join(output_dir, 'velocity_NFW_profile.png')
    z_list = [12, 6, 0]
    M_list = [1.0e7, 1.0e10, 1.0e13]

    for iz, z in enumerate(z_list):
        fig, ax = plt.subplots(1, len(M_list), figsize=(24, 6), facecolor='white')
        for iM, M in enumerate(M_list):
            concentration = get_concentration_Bullock01(M, z)
            r_Rvir_list = np.logspace(-2, 0, 100)
            x_list = r_Rvir_list * concentration

            Velvir = Velvir_NFW_profile(x_list, M, z)
            ax[iM].plot(r_Rvir_list, Velvir, label='velocity', color='blue', linestyle='--')
            ax[iM].set_xscale('log')
            ax[iM].set_yscale('log')
            ax[iM].set_xlabel('r/Rvir', fontsize=14)
            ax[iM].set_ylabel('velocity/virial velocity', fontsize=14)
            ax[iM].set_title(f'M={M:.1e}Msun, z={z}', fontsize=14)
            ax[iM].legend()
        plt.savefig(filename.replace('.png', f'_z{z}.png'))