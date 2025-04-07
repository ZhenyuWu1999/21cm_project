
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import quad
from matplotlib.ticker import LogLocator

from HaloMassFunction import get_M_Jeans, SHMF_BestFit_dN_dlgx  , HMF_2Dbestfit, integrand_oldversion 
from physical_constants import *
from HaloProperties import Vel_Virial_analytic, Temperature_Virial_analytic, get_gas_lognH_analytic, \
get_mass_density_analytic, inversefunc_Temperature_Virial_analytic
from TNGDataHandler import get_simulation_resolution
from Grackle_cooling import run_constdensity_model
from pygrackle.utilities.physical_constants import sec_per_Myr
from Analytic_halo_profile import *

def lgM_to_Tvir(lgM, z):
    #lgM in Msun/h
    Tvir = Temperature_Virial_analytic(10**lgM/h_Hubble, z)
    return Tvir

def Tvir_to_lgM(Tvir, z):
    Mvir = inversefunc_Temperature_Virial_analytic(Tvir, z) #Mvir in Msun
    lgM = np.log10(Mvir * h_Hubble)  # convert to lgM [M_sun/h]
    return lgM

def get_DF_heating_useVelVirial(M, m, redshft):
    #M, m in Msun/h
    #return DF heating in J/s
    rho_g = 200 * rho_b0*(1+redshft)**3 *Msun/Mpc**3
    I_DF = 1.0 #do not consider I_DF here
    DF_heating = I_DF* 4 * np.pi * (G_grav * m *Msun/h_Hubble) ** 2 / Vel_Virial_analytic(M/h_Hubble, redshft) *rho_g
    return DF_heating

def get_DF_heating_useCs(M, m, redshft):
    #M, m in Msun/h
    #return DF heating in J/s
    rho_g = 200 * rho_b0*(1+redshft)**3 *Msun/Mpc**3
    I_DF = 1.0 #do not consider I_DF here
    Tvir = Temperature_Virial_analytic(M/h_Hubble, redshft)
    Cs = np.sqrt(5.0/3.0 * kB * Tvir / (mu*mp))
    DF_heating = I_DF* 4 * np.pi * (G_grav * m *Msun/h_Hubble) ** 2 / Cs *rho_g
    return DF_heating

def integrate_SHMF_heating_for_single_host(redshift, lgx_min, lgx_max, lgM, SHMF_model):
    lg_x_bin_edges = np.linspace(lgx_min, lgx_max, 50)
    lg_x_bin_centers = 0.5*(lg_x_bin_edges[1:]+lg_x_bin_edges[:-1])
    lg_x_bin_width = lg_x_bin_edges[1] - lg_x_bin_edges[0]
    dN_dlgx = SHMF_BestFit_dN_dlgx(lg_x_bin_centers, redshift, SHMF_model)
    N_subs_per_bin = dN_dlgx * lg_x_bin_width
    Mhost = 10**lgM
    m_subs = Mhost * 10**lg_x_bin_centers

    #debug: useVelVirial or useCs
    heating_per_sub = np.array([get_DF_heating_useCs(Mhost, m, redshift) for m in m_subs])
    heating_per_bin = heating_per_sub * N_subs_per_bin
    SHMF_heating = np.sum(heating_per_bin)
    return SHMF_heating

def get_heating_per_lgM(lgM_list, lgx_min_list, lgx_max_list, redshift, SHMF_model):
    '''
    return:
    data_dict = {'lgM_list':lgM_list, 
                 'Heating_singlehost':Heating_singlehost [J/s], 
                 'Heating_perlgM':Heating_perlgM [J/s (Mpc/h)$^{-3}$ dex$^{-1}$]}
    '''
    Heating_singlehost = []
    Heating_perlgM = []
    for index, lgM in enumerate(lgM_list):
        lgx_min = lgx_min_list[index]
        lgx_max = lgx_max_list[index]
        heating = integrate_SHMF_heating_for_single_host(redshift, lgx_min, lgx_max, lgM, SHMF_model)
        dN_dlgM = HMF_2Dbestfit(lgM, redshift)
        Heating_singlehost.append(heating)
        Heating_perlgM.append(heating*dN_dlgM)

    Heating_singlehost = np.array(Heating_singlehost)
    Heating_perlgM = np.array(Heating_perlgM)

    data_dict = {'lgM_list':lgM_list, 
                 'Heating_singlehost':Heating_singlehost, 
                 'Heating_perlgM':Heating_perlgM}
    return data_dict


def get_EqCooling_for_single_host(Mvir, redshift):
    #Mvir in Msun/h
    #return cooling rate in erg/s

    UVB_flag = False
    Compton_Xray_flag = False
    dynamic_final_flag = True
    
    mass_density = get_mass_density_analytic(redshift)
    volume_vir = Mvir*Msun/h_Hubble/mass_density
    volume_vir_cm3 = volume_vir * (1e6)
    lognH = get_gas_lognH_analytic(redshift)
    nH = 10**lognH
    specific_heating_rate = 0.0
    volumetric_heating_rate = 0.0
    temperature = Temperature_Virial_analytic(Mvir/h_Hubble, redshift)
    gas_metallicity_2 = 1.0e-2
    gas_metallicity_6 = 1.0e-6
    cooling_Eq_Z2 = run_constdensity_model(False,redshift,lognH,specific_heating_rate, volumetric_heating_rate, temperature, gas_metallicity_2, 
                            UVB_flag=UVB_flag, Compton_Xray_flag=Compton_Xray_flag, dynamic_final_flag=dynamic_final_flag)
    cooling_Eq_Z6 = run_constdensity_model(False,redshift,lognH,specific_heating_rate, volumetric_heating_rate, temperature, gas_metallicity_6,
                            UVB_flag=UVB_flag, Compton_Xray_flag=Compton_Xray_flag, dynamic_final_flag=dynamic_final_flag)

    # print("Equilibrium cooling rate:")
    # print(cooling_Eq_Z2["cooling_rate"])
    # print(cooling_Eq_Z6["cooling_rate"])
    # print("Equilibrium cooling time:")
    # print(cooling_Eq_Z2["cooling_time"])

    normalized_cooling_Z2 = cooling_Eq_Z2["cooling_rate"].v
    cooling_rate_Z2 = normalized_cooling_Z2 * nH**2
    tot_cooling_rate_Z2 = cooling_rate_Z2 * volume_vir_cm3
    normalized_cooling_Z6 = cooling_Eq_Z6["cooling_rate"].v
    cooling_rate_Z6 = normalized_cooling_Z6 * nH**2
    tot_cooling_rate_Z6 = cooling_rate_Z6 * volume_vir_cm3

    #debug
    # print("get_EqCooling_for_single_host, Tvir: ",temperature)
    # print("nH: ",nH)
    # print("normalized_cooling_Z2: ",normalized_cooling_Z2,"erg/s cm^3")
    # print("tot_cooling_rate_Z2: ",tot_cooling_rate_Z2)
    


    return -tot_cooling_rate_Z2, -tot_cooling_rate_Z6

def get_EqCoolingDensity(r_Rvir, Mvir, redshift, concentration_model):
    #r_Rir: ratio of r/Rvir
    #Mvir in Msun/h
    #return cooling rate in erg/s/cm^3

    UVB_flag = False
    Compton_Xray_flag = False
    dynamic_final_flag = True

    Mvir_in_Msun = Mvir/h_Hubble
    concentration = get_concentration(Mvir_in_Msun, redshift, concentration_model)
    x = r_Rvir * concentration
    vir_mass_density = get_mass_density_analytic(redshift)
    local_mass_density = density_NFW_profile(x, Mvir_in_Msun, redshift, concentration_model) * vir_mass_density
    local_gas_NFW_density = gasdensity_NFW_profile(x, Mvir_in_Msun, redshift, concentration_model) * vir_mass_density
    local_gas_core_density = gasdensity_core_profile(x, Mvir_in_Msun, redshift, concentration_model) * vir_mass_density

    local_nH_NFW_cm3 = local_gas_NFW_density/(mu*mp)/1.0e6
    local_nH_core_cm3 = local_gas_core_density/(mu*mp)/1.0e6
    local_lognH_NFW = np.log10(local_nH_NFW_cm3)
    local_lognH_core = np.log10(local_nH_core_cm3)

    specific_heating_rate = 0.0
    volumetric_heating_rate = 0.0
    temperature = Temperature_Virial_analytic(Mvir_in_Msun, redshift) #assume isothermal
    gas_metallicity_2 = 1.0e-2
    gas_metallicity_6 = 1.0e-6
    cooling_Eq_NFW_Z2 = run_constdensity_model(False,redshift,local_lognH_NFW,specific_heating_rate, volumetric_heating_rate, temperature, gas_metallicity_2,
                            UVB_flag=UVB_flag, Compton_Xray_flag=Compton_Xray_flag, dynamic_final_flag=dynamic_final_flag)
    cooling_Eq_NFW_Z6 = run_constdensity_model(False,redshift,local_lognH_NFW,specific_heating_rate, volumetric_heating_rate, temperature, gas_metallicity_6,
                            UVB_flag=UVB_flag, Compton_Xray_flag=Compton_Xray_flag, dynamic_final_flag=dynamic_final_flag)
    
    cooling_Eq_core_Z2 = run_constdensity_model(False,redshift,local_lognH_core,specific_heating_rate, volumetric_heating_rate, temperature, gas_metallicity_2,
                            UVB_flag=UVB_flag, Compton_Xray_flag=Compton_Xray_flag, dynamic_final_flag=dynamic_final_flag)
    cooling_Eq_core_Z6 = run_constdensity_model(False,redshift,local_lognH_core,specific_heating_rate, volumetric_heating_rate, temperature, gas_metallicity_6,
                            UVB_flag=UVB_flag, Compton_Xray_flag=Compton_Xray_flag, dynamic_final_flag=dynamic_final_flag)

    normalized_cooling_NFW_Z2 = cooling_Eq_NFW_Z2["cooling_rate"].v
    cooling_rate_NFW_Z2 = normalized_cooling_NFW_Z2 * local_nH_NFW_cm3**2
    normalized_cooling_NFW_Z6 = cooling_Eq_NFW_Z6["cooling_rate"].v
    cooling_rate_NFW_Z6 = normalized_cooling_NFW_Z6 * local_nH_NFW_cm3**2
    normalized_cooling_core_Z2 = cooling_Eq_core_Z2["cooling_rate"].v
    cooling_rate_core_Z2 = normalized_cooling_core_Z2 * local_nH_core_cm3**2
    normalized_cooling_core_Z6 = cooling_Eq_core_Z6["cooling_rate"].v
    cooling_rate_core_Z6 = normalized_cooling_core_Z6 * local_nH_core_cm3**2


    #debug
    # print("get_EqCoolingDensity, debug: ")
    # print("temperature: ",temperature)
    # print("r_Rvir: ",r_Rvir)
    # print("local nH, NFW: ",local_nH_NFW_cm3)
    # print("local nH, core: ",local_nH_core_cm3)
    # print("normalized_cooling_NFW_Z2: ",normalized_cooling_NFW_Z2,"erg/s cm^3")
    # print("cooling_rate_NFW_Z2: ",cooling_rate_NFW_Z2,"erg/s/cm^3")
    # print("normalized_cooling_core_Z2: ",normalized_cooling_core_Z2,"erg/s cm^3")
    # print("cooling_rate_core_Z2: ",cooling_rate_core_Z2,"erg/s/cm^3")

    return -cooling_rate_NFW_Z2, -cooling_rate_NFW_Z6, -cooling_rate_core_Z2, -cooling_rate_core_Z6            
                                    

    

def get_NonEqCooling_for_single_host(Mvir, redshift, heating_singlehost):
    #Mvir in Msun/h
    #heating_singlehost in J/s
    #return cooling rate in erg/s

    UVB_flag = False
    Compton_Xray_flag = False
    dynamic_final_flag = True
    
    mass_density = get_mass_density_analytic(redshift)
    volume_vir = Mvir*Msun/h_Hubble/mass_density
    volume_vir_cm3 = volume_vir * (1e6)
    lognH = get_gas_lognH_analytic(redshift)
    nH = 10**lognH
    t_ff = freefall_factor / np.sqrt(G_grav * mass_density) #unit s

    specific_heating_rate = 0.0
    heating_singlehost_erg = heating_singlehost * 1e7
    volumetric_heating_const = heating_singlehost_erg / volume_vir_cm3 #erg/s/cm^3
    #now create the volumetric heating rate array as a function of time for interpolation
    #(t = 0 - t_ff: volumetric_heating_const, t_ff - 2t_ff: 0, time in unit Myr)
    time_array = np.linspace(0, 3*t_ff/sec_per_Myr, 100)
    volumetric_heating_rate_array = np.array([volumetric_heating_const if time < t_ff else 0.0 for time in time_array])
    volumetric_heating_rate = (time_array, volumetric_heating_rate_array)
    print("volumetric_heating_rate: ",volumetric_heating_rate)
    print("t_ff: ",t_ff)
    final_time = t_ff

    temperature = Temperature_Virial_analytic(Mvir/h_Hubble, redshift)
    gas_metallicity_2 = 1.0e-2
    gas_metallicity_6 = 1.0e-6
    cooling_NonEq_Z2 = run_constdensity_model(True,redshift,lognH,specific_heating_rate, 0.0, temperature, gas_metallicity_2, 
                            UVB_flag=UVB_flag, Compton_Xray_flag=Compton_Xray_flag, dynamic_final_flag=dynamic_final_flag, final_time=final_time)
    heating_NonEq_Z2 = run_constdensity_model(True,redshift,lognH,specific_heating_rate, volumetric_heating_rate, temperature, gas_metallicity_2,
                            UVB_flag=UVB_flag, Compton_Xray_flag=Compton_Xray_flag, dynamic_final_flag=dynamic_final_flag, final_time=final_time)
    
    cooling_NonEq_Z6 = run_constdensity_model(True,redshift,lognH,specific_heating_rate, 0.0, temperature, gas_metallicity_6,
                            UVB_flag=UVB_flag, Compton_Xray_flag=Compton_Xray_flag, dynamic_final_flag=dynamic_final_flag, final_time=final_time)
    heating_NonEq_Z6 = run_constdensity_model(True,redshift,lognH,specific_heating_rate, volumetric_heating_rate, temperature, gas_metallicity_6,
                            UVB_flag=UVB_flag, Compton_Xray_flag=Compton_Xray_flag, dynamic_final_flag=dynamic_final_flag, final_time=final_time)
    print("Non-equilibrium cooling rate:")
    print(cooling_NonEq_Z6["time"])
    print(cooling_NonEq_Z6["temperature"])
    print(cooling_NonEq_Z6["cooling_rate"])

    print("-----------------------------")
    print("Non-equilibrium cooling+heating rate:")
    print(heating_NonEq_Z6["time"])
    print(heating_NonEq_Z6["temperature"])
    print(heating_NonEq_Z6["cooling_rate"])


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

    #assume isotropic temperature profile

    #cooling
    tot_cooling_rate_Z2, tot_cooling_rate_Z6 = get_EqCooling_for_single_host(Mvir, z)

    print("Total cooling rate Z=1e-2: ",tot_cooling_rate_Z2, "erg/s")
    print("Total cooling rate Z=1e-6: ",tot_cooling_rate_Z6, "erg/s")

    avg_cooling_density_Z2 = tot_cooling_rate_Z2 / (4/3*np.pi*R_vir_cm**3)
    avg_cooling_density_Z6 = tot_cooling_rate_Z6 / (4/3*np.pi*R_vir_cm**3)

    print("Average cooling density Z=1e-2: ",avg_cooling_density_Z2, "erg/s/cm^3")
    print("Average cooling density Z=1e-6: ",avg_cooling_density_Z6, "erg/s/cm^3")


    cooling_NFW_Z2_list = []
    cooling_NFW_Z6_list = []
    cooling_core_Z2_list = []
    cooling_core_Z6_list = []
    for r_Rvir in r_Rvir_list:
        cooling_NFW_Z2, cooling_NFW_Z6, cooling_core_Z2, cooling_core_Z6 = get_EqCoolingDensity(r_Rvir, Mvir, z, concentration_model)
        cooling_NFW_Z2_list.append(cooling_NFW_Z2)
        cooling_NFW_Z6_list.append(cooling_NFW_Z6)
        cooling_core_Z2_list.append(cooling_core_Z2)
        cooling_core_Z6_list.append(cooling_core_Z6)


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
    ax1.plot(r_Rvir_list,cooling_NFW_Z2_list,'g--',label='Cooling NFW Z=1e-2')
    ax1.plot(r_Rvir_list,cooling_NFW_Z6_list,'b--',label='Cooling NFW Z=1e-6')
    ax1.plot(r_Rvir_list,cooling_core_Z2_list,'g-',label='Cooling core Z=1e-2')
    ax1.plot(r_Rvir_list,cooling_core_Z6_list,'b-',label='Cooling core Z=1e-6')
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


    



def Analytic_model(redshift):
    #check contribution to heating (analytical result)
    M_Jeans = get_M_Jeans(redshift)
    print("Jeans mass: ",M_Jeans)
    lgM_limits = [4, 14]  # Limits for log10(M [Msun/h])

    lgM_list = np.linspace(lgM_limits[0], lgM_limits[1],50)
    #x = m/M
    #set Jeans mass as min subhalo mass (and test other values)
    lgx_min_MJeans_list = np.array([np.log10(M_Jeans/10**lgM_list[j]) for j in range(len(lgM_list))])
    lgx_min_3_list = np.array([np.log10(1e-3) for j in range(len(lgM_list))])
    lgx_min_5_list = np.array([np.log10(1e-5) for j in range(len(lgM_list))])
    
    #require max subhalo ratio to be 0.1 to avoid major mergers (and test other values)
    lgx_max_0_list = np.array([np.log10(1.0) for j in range(len(lgM_list))])
    lgx_max_half_list = np.array([np.log10(0.5) for j in range(len(lgM_list))])
    lgx_max_1_list = np.array([np.log10(1.0e-1) for j in range(len(lgM_list))])
    lgx_max_2_list = np.array([np.log10(1.0e-2) for j in range(len(lgM_list))])

    #cooling and heating
    cooling_Z2_list = []
    cooling_Z6_list = []
    for lgM in lgM_list:
        cooling_Z2, cooling_Z6 = get_EqCooling_for_single_host(10**lgM, redshift)
        cooling_Z2_list.append(cooling_Z2)
        cooling_Z6_list.append(cooling_Z6)
    cooling_Z2_list = np.array(cooling_Z2_list)
    cooling_Z6_list = np.array(cooling_Z6_list)

    data_minMJeans_max1 = get_heating_per_lgM(lgM_list, lgx_min_MJeans_list, lgx_max_1_list, redshift, 'BestFit_z')
    data_min5_max1 = get_heating_per_lgM(lgM_list, lgx_min_5_list, lgx_max_1_list, redshift, 'BestFit_z')
    data_min3_max1 = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_1_list, redshift, 'BestFit_z')
    data_min3_max0 = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_0_list, redshift, 'BestFit_z')
    data_min3_max2 = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_2_list, redshift, 'BestFit_z')
    data_min3_max1_Bosch16evolved = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_1_list, redshift, 'Bosch16evolved')
    data_min3_max1_Bosch16unevolved = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_1_list, redshift, 'Bosch16unevolved')

    #heating per logM (old version)
    ln_m_over_M_limits = [np.log(1e-3), np.log(1.0)]
    DF_heating_perlogM_old = []
    for logM in lgM_list:
        result, error = quad(integrand_oldversion, ln_m_over_M_limits[0], ln_m_over_M_limits[1], args=(logM, redshift, 'Bosch2016'))

        if (abs(error) > 0.01 * abs(result)):
            print(f"Warning: error in integration is large: {error} at z={redshift}, logM={logM}")
        DF_heating_perlogM_old.append(result)
    DF_heating_perlogM_old = np.array(DF_heating_perlogM_old)
    label_old = r'$m/M \in [10^{-3},1]$, Bosch16evolved'


    data_plots = [data_minMJeans_max1, data_min5_max1, data_min3_max1,data_min3_max0,data_min3_max2,
                  data_min3_max1_Bosch16evolved, data_min3_max1_Bosch16unevolved]
    colors = ['g','b','r','r','r',
              'grey','grey']
    labels = [r'$[m_J/M,10^{-1}]$ BestFit',r'$[10^{-5},10^{-1}]$ BestFit',r'$[10^{-3},10^{-1}]$ BestFit',r'$[10^{-3},1]$ BestFit',r'$[10^{-3},10^{-2}]$ BestFit',
                r'$[10^{-3},10^{-1}]$ Bosch16evolved',r'$[10^{-3},10^{-1}]$ Bosch16unevolved']
    linestyle = ['-', '-.', '--', ':', ':',
                 '-','--']
    linewidth = [1,1,1,2,1,
                 1,1]
    #plot heating for M
    output_dir = '/home/zwu/21cm_project/unified_model/Analytic_results'
    filename = os.path.join(output_dir,f"DF_heating_singlehost_z{redshift:.2f}_muionized.png")

    
    fig, ax1 = plt.subplots(figsize=(8, 6), facecolor='white')
    for i in range(len(data_plots)):
        data = data_plots[i]
        ax1.plot(data['lgM_list'],1e7*data['Heating_singlehost'],colors[i],linestyle=linestyle[i],linewidth=linewidth[i],label=labels[i])
    ax1.plot(lgM_list,cooling_Z2_list,'k-',label='Cooling Z=1e-2')
    ax1.plot(lgM_list,cooling_Z6_list,'k--',label='Cooling Z=1e-6')
    ax1.legend()
    ax1.set_xlim([min(lgM_list),max(lgM_list)])
    ax1.set_yscale('log')
    ax1.set_ylabel(r'Cooling and Heating [erg/s]',fontsize=14)
    ax1.set_xlabel(r'lgM [M$_{\odot}$/h]',fontsize=14)
    ax1.tick_params(axis='both', direction='in')
    

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())

    # Define clean Tvir ticks (integer powers of 10)
    Tvir_min = lgM_to_Tvir(min(lgM_list), redshift)
    Tvir_max = lgM_to_Tvir(max(lgM_list), redshift)
    Tvir_locator = LogLocator(base=10)
    Tvir_ticks = Tvir_locator.tick_values(Tvir_min, Tvir_max)
    lgM_ticks_top = [Tvir_to_lgM(Tvir, redshift) for Tvir in Tvir_ticks]

    # Filter valid ticks within the plot limits
    valid_ticks = [(lgM, Tvir) for lgM, Tvir in zip(lgM_ticks_top, Tvir_ticks) if min(lgM_list) <= lgM <= max(lgM_list)]
    lgM_ticks_top, Tvir_ticks = zip(*valid_ticks)

    # Set the ticks and labels
    ax2.set_xticks(lgM_ticks_top)
    ax2.set_xticklabels([f"$10^{int(np.log10(Tvir))}$" for Tvir in Tvir_ticks])
    ax2.set_xlabel(r'Virial Temperature [K]', fontsize=14)
    ax2.tick_params(axis='x', direction='in')

    plt.tight_layout()
    plt.savefig(filename,dpi=300)



    #plot heating per logM
    filename = os.path.join(output_dir,f"DF_heating_perlogM_z{redshift:.2f}.png")
    fig = plt.figure(facecolor='white')
    for i in range(len(data_plots)): 
        data = data_plots[i]
        plt.plot(data['lgM_list'],1e7*data['Heating_perlgM'],colors[i],linestyle=linestyle[i],linewidth=linewidth[i],label=labels[i])
    #also compare with old version
    plt.plot(lgM_list,1e7*DF_heating_perlogM_old,'k-',label=label_old)
    gas_resolution, dark_matter_resolution = get_simulation_resolution('TNG50-1')
    plt.axvline(np.log10(50*dark_matter_resolution), color='black', linestyle='--',label='50 particles')
    plt.legend()
    plt.xlim([min(lgM_list),max(lgM_list)])
    plt.ylim([1e35,1e43])
    plt.yscale('log')
    plt.ylabel(r'DF heating per lgM [erg/s (Mpc/h)$^{-3}$ dex$^{-1}$]',fontsize=14)
    plt.xlabel(r'lgM [M$_{\odot}$/h]',fontsize=14)
    plt.savefig(filename,dpi=300)




if __name__ == "__main__":

    # Analytic_model(0)

    # for z in [15, 12, 10, 8, 6]:
    #     Analytic_model(z)
    # z = 12
    # lgM = 7.0
    # heating_singlehost = integrate_SHMF_heating_for_single_host(z, -3, -1, lgM, 'BestFit_z')
    # get_NonEqCooling_for_single_host(10**lgM, z, heating_singlehost)


    # cooling and heating profile
    concentration_model = 'ludlow16'
    output_dir = f'/home/zwu/21cm_project/unified_model/Analytic_results/cooling_heating_profile/{concentration_model}'

    '''
    for z in [0]:
        for M in [1e7, 1e10, 1e13, 1e14, 1e15]:
            print(f"z={z}, M={M:.2e}")
            compare_cooling_heating_profile(output_dir, M, z, -3, -1, 'Bosch16evolved', 'ludlow16')
    '''
    '''
    for z in [6]:
        output_dir = os.path.join(output_dir, f'z6')
        for M in [1e7, 1e10, 1e12]:
            print(f"z={z}, M={M:.2e}")
            compare_cooling_heating_profile(output_dir, M, z, -3, -1, 'BestFit_z', 'ludlow16')
    '''
    '''
    for z in [10]:
        output_dir = os.path.join(output_dir, f'z10')
        for M in [1e7, 1e10, 1e11]:
            print(f"z={z}, M={M:.2e}")
            compare_cooling_heating_profile(output_dir, M, z, -3, -1, 'BestFit_z', 'ludlow16')
    '''
    '''
    for z in [12]:
        output_dir = os.path.join(output_dir, f'z12')
        for M in [1e7, 1e10, 1e11]:
            print(f"z={z}, M={M:.2e}")
            compare_cooling_heating_profile(output_dir, M, z, -3, -1, 'BestFit_z', 'ludlow16')
    '''
    for z in [15]:
        output_dir = os.path.join(output_dir, f'z15')
        for M in [1e7, 1e10]:
            print(f"z={z}, M={M:.2e}")
            compare_cooling_heating_profile(output_dir, M, z, -3, -1, 'BestFit_z', 'ludlow16')
    