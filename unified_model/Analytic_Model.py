
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
from scipy.integrate import quad
from matplotlib.ticker import LogLocator
import copy

from HaloMassFunction import get_M_Jeans, SHMF_BestFit_dN_dlgx, HMF_2Dbestfit, integrand_oldversion, \
get_cumulativeSHMF_sigma_correction, get_normalized_SHMF_Cumulative, onetime_sample_SHMF_for_Ntot
from physical_constants import *
from HaloProperties import Vel_Virial_analytic, Temperature_Virial_analytic, get_gas_lognH_analytic, \
get_mass_density_analytic, inversefunc_Temperature_Virial_analytic
from TNGDataHandler import get_simulation_resolution
from Grackle_cooling import run_constdensity_model
from pygrackle.utilities.physical_constants import sec_per_Myr
from Analytic_halo_profile import *
from Dekel08 import get_heating_Dekel08, get_cooling_Dekel08, get_cooling_Dekel08_Eq25
from TNGDataHandler import load_processed_data
from Config import simulation_set, Kim2005_result
from DF_Ostriker99_wake_structure import Idf_Ostriker99_nosingularity_Vtrmin

def lgM_to_Tvir(lgM, z, mean_molecular_weight=mu):
    #lgM in Msun/h
    Tvir = Temperature_Virial_analytic(10**lgM/h_Hubble, z, mean_molecular_weight)  # Tvir in K
    return Tvir

def Tvir_to_lgM(Tvir, z, mean_molecular_weight=mu):
    Mvir = inversefunc_Temperature_Virial_analytic(Tvir, z, mean_molecular_weight) #Mvir in Msun
    lgM = np.log10(Mvir * h_Hubble)  # convert to lgM [M_sun/h]
    return lgM

def get_DF_heating_useVelVirial(M, m, redshft):
    #M, m in Msun/h
    #return DF heating in J/s
    rho_g = 200 * rho_b0*(1+redshft)**3 *Msun/Mpc**3
    I_DF = 1.0 #do not consider I_DF here
    DF_heating = I_DF* 4 * np.pi * (G_grav * m *Msun/h_Hubble) ** 2 / Vel_Virial_analytic(M/h_Hubble, redshft) *rho_g
    return DF_heating

def get_DF_heating_useCs(M, m, redshft, mean_molecular_weight=mu):
    #M, m in Msun/h
    #return DF heating in J/s
    rho_g = 200 * rho_b0*(1+redshft)**3 *Msun/Mpc**3
    I_DF = 1.0 #do not consider I_DF here
    Tvir = Temperature_Virial_analytic(M/h_Hubble, redshft)
    Cs = np.sqrt(5.0/3.0 * kB * Tvir / (mean_molecular_weight*mp))
    DF_heating = I_DF* 4 * np.pi * (G_grav * m *Msun/h_Hubble) ** 2 / Cs *rho_g
    return DF_heating

def integrate_SHMF_heating_for_single_host(redshift, lgx_min, lgx_max, lgM, SHMF_model, mean_molecular_weight=mu):
    lg_x_bin_edges = np.linspace(lgx_min, lgx_max, 50)
    lg_x_bin_centers = 0.5*(lg_x_bin_edges[1:]+lg_x_bin_edges[:-1])
    lg_x_bin_width = lg_x_bin_edges[1] - lg_x_bin_edges[0]
    dN_dlgx = SHMF_BestFit_dN_dlgx(lg_x_bin_centers, redshift, SHMF_model)
    N_subs_per_bin = dN_dlgx * lg_x_bin_width
    Mhost = 10**lgM
    m_subs = Mhost * 10**lg_x_bin_centers

    #debug: useVelVirial or useCs
    heating_per_sub = np.array([get_DF_heating_useCs(Mhost, m, redshift, mean_molecular_weight) for m in m_subs])
    heating_per_bin = heating_per_sub * N_subs_per_bin
    SHMF_heating = np.sum(heating_per_bin)
    return SHMF_heating

def integrate_SHMF_heating_for_single_host_with_variance(redshift, lgx_min, lgx_max, lgM, SHMF_model, 
                                                        variance_factor_list, correction_model, mean_molecular_weight=mu):
    #variance_factor_list: list of factors for variance, e.g., [1, 2, 3] means 1, 2, and 3 sigma levels
    #correction_model: 'superPoisson' or 'supersubPoisson', or 'None'
    lg_x_bin_edges = np.linspace(lgx_min, lgx_max, 50)
    lg_x_bin_centers = 0.5*(lg_x_bin_edges[1:] + lg_x_bin_edges[:-1])
    lg_x_bin_width = lg_x_bin_edges[1] - lg_x_bin_edges[0]
    dN_dlgx_mean = SHMF_BestFit_dN_dlgx(lg_x_bin_centers, redshift, SHMF_model)
    N_subs_per_bin_mean = dN_dlgx_mean * lg_x_bin_width
    
    #N_cumulative: N(>m/M)
    N_cumulative_mean = np.cumsum(N_subs_per_bin_mean[::-1])[::-1]
    
    # calculate variance
    sigma_Poisson = np.sqrt(N_cumulative_mean)
    Poisson_corr = get_cumulativeSHMF_sigma_correction(N_cumulative_mean, correction_model)
    sigma_Poissoncorr = sigma_Poisson * Poisson_corr

    # calculate average heating rate
    Mhost = 10**lgM
    m_subs = Mhost * 10**lg_x_bin_centers
    heating_per_sub = np.array([get_DF_heating_useCs(Mhost, m, redshift, mean_molecular_weight) for m in m_subs])
    heating_per_bin_mean = heating_per_sub * N_subs_per_bin_mean
    heating_mean = np.sum(heating_per_bin_mean)
    
    # get boundaries of different sigma levels
    heating_upper_list = []
    heating_lower_list = []
    
    for variance_factor in variance_factor_list:
        # Upper boundary: +n sigma
        N_cumulative_upper = N_cumulative_mean + variance_factor * sigma_Poissoncorr
        # Lower boundary: -n sigma
        N_cumulative_lower = N_cumulative_mean - variance_factor * sigma_Poissoncorr
        N_cumulative_lower = np.maximum(N_cumulative_lower, 0.0)  
        
        # convert back to differential SHMF
        def cumulative_to_differential(N_cumulative):
            N_subs_per_bin = np.zeros_like(N_cumulative)
            N_subs_per_bin[:-1] = N_cumulative[:-1] - N_cumulative[1:]
            N_subs_per_bin[-1] = N_cumulative[-1]
            return N_subs_per_bin
        
        N_subs_per_bin_upper = cumulative_to_differential(N_cumulative_upper)
        N_subs_per_bin_lower = cumulative_to_differential(N_cumulative_lower)
        
        #heating rates
        heating_per_bin_upper = heating_per_sub * N_subs_per_bin_upper
        heating_per_bin_lower = heating_per_sub * N_subs_per_bin_lower
        
        heating_upper = np.sum(heating_per_bin_upper)
        heating_lower = np.sum(heating_per_bin_lower)
        
        heating_upper_list.append(heating_upper)
        heating_lower_list.append(heating_lower)
    
    return heating_upper_list, heating_lower_list, heating_mean


def integrate_SHMF_heating_for_single_host_PoissonSampling(redshift, lgx_min, lgx_max, lgM, SHMF_model, n_samples, mean_molecular_weight=mu, verbose=True):
    
    lg_x_vals, F_vals, N_mean = get_normalized_SHMF_Cumulative(lgx_min, lgx_max, redshift, SHMF_model)
    if verbose:
        print(f"Mean total number of subhalos: {N_mean:.2f}")
        print(f"F_vals range: [{F_vals[0]:.3f}, {F_vals[-1]:.3f}]")

    #plot histogram of the generated lg_psi and compare with BestFit_z model
    lg_x_bin_edges = np.linspace(lgx_min, lgx_max, 50)
    lg_x_bin_centers = 0.5*(lg_x_bin_edges[1:] + lg_x_bin_edges[:-1])
    lg_x_bin_width = lg_x_bin_edges[1] - lg_x_bin_edges[0]
    n_bins = len(lg_x_bin_centers)

    dN_dlgx_mean = SHMF_BestFit_dN_dlgx(lg_x_bin_centers, redshift, SHMF_model)
    theoretical_counts = dN_dlgx_mean * lg_x_bin_width

    SHMF_heating_for_host_samples = []
    for i in range(n_samples):
        #Ntot_sample = round(N_mean)  # Use mean as test case, without Poisson fluctuations
        Ntot_sample = np.random.poisson(N_mean) #with Poisson fluctuations
        if verbose:
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{n_samples} samples")
            
        # Bin the sample
        if Ntot_sample == 0:
            SHMF_heating_for_host_samples.append(0.0)
        else:
            # Sample individual subhalo masses (psi values)
            sampled_lg_psi = onetime_sample_SHMF_for_Ntot(lg_x_vals, F_vals, Ntot_sample)
            #directly sum the heating of all subhalos without binning
            Mhost = 10**lgM
            m_subs = Mhost * 10**sampled_lg_psi
            heating_of_subs = np.array([get_DF_heating_useCs(Mhost, m, redshift, mean_molecular_weight) for m in m_subs])
            heating_sum = np.sum(heating_of_subs)
            SHMF_heating_for_host_samples.append(heating_sum)
    SHMF_heating_for_host_samples = np.array(SHMF_heating_for_host_samples)
    return SHMF_heating_for_host_samples

def get_heating_per_lgM(lgM_list, lgx_min_list, lgx_max_list, redshift, SHMF_model, mean_molecular_weight=mu):
    '''
    return:
    data_dict = {'lgM_list':lgM_list, 
                 'Heating_singlehost':Heating_singlehost [J/s], 
                 'Heating_perlgM':Heating_perlgM [J/s (Mpc/h)$^{-3}$ dex$^{-1}$]}
    '''
    Heating_singlehost = []
    Heating_perlgM = []
    Heating_perlgM_totHMF = []
    for index, lgM in enumerate(lgM_list):
        lgx_min = lgx_min_list[index]
        lgx_max = lgx_max_list[index]
        heating = integrate_SHMF_heating_for_single_host(redshift, lgx_min, lgx_max, lgM, SHMF_model, mean_molecular_weight)
        dN_dlgM = HMF_2Dbestfit(lgM, redshift, include_selection_factor=True) 
        dN_dlgM_totHMF = HMF_2Dbestfit(lgM, redshift, include_selection_factor=False)
        Heating_singlehost.append(heating)
        Heating_perlgM.append(heating*dN_dlgM)
        Heating_perlgM_totHMF.append(heating*dN_dlgM_totHMF)

    Heating_singlehost = np.array(Heating_singlehost)
    Heating_perlgM = np.array(Heating_perlgM)
    Heating_perlgM_totHMF = np.array(Heating_perlgM_totHMF)

    data_dict = {'lgM_list':lgM_list, 
                 'Heating_singlehost':Heating_singlehost, 
                 'Heating_perlgM':Heating_perlgM,
                 'Heating_perlgM_totHMF':Heating_perlgM_totHMF}
    return data_dict


def get_EqCooling_for_single_host(Mvir, redshift, param_sets, mean_molecular_weight=mu):
    """
    Calculate cooling rates for multiple parameter sets
    Parameters:
    -----------
    Mvir : float
        Virial mass in Msun/h
    redshift : float
        Redshift value
    param_sets : list of dict
        List of parameter dictionaries, each containing:
        - 'gas_metallicity': Gas metallicity in Zsun
        - 'f_H2': H2 fraction
    
    Returns:
    --------
    list : List of dictionaries, each containing the input parameters and resulting cooling rate
    """
    results = []
    
    UVB_flag = False
    Compton_Xray_flag = False
    dynamic_final_flag = False
    
    # Pre-calculate common values
    mass_density = get_mass_density_analytic(redshift)
    volume_vir = Mvir*Msun/h_Hubble/mass_density
    volume_vir_cm3 = volume_vir * (1e6)
    lognH = get_gas_lognH_analytic(redshift)
    nH = 10**lognH
    specific_heating_rate = 0.0
    volumetric_heating_rate = 0.0
    temperature = Temperature_Virial_analytic(Mvir/h_Hubble, redshift, mean_molecular_weight)
    
    # Iterate through all parameter sets
    for params in param_sets:
        print("parameters: ", params)
        gas_metallicity = params['gas_metallicity']
        f_H2 = params['f_H2']

        params_for_constdensity = {
            "evolve_cooling": False,
            "redshift": redshift,
            "lognH": lognH,
            "specific_heating_rate": specific_heating_rate,
            "volumetric_heating_rate": volumetric_heating_rate,
            "temperature": temperature,
            "gas_metallicity": gas_metallicity,
            "f_H2": f_H2,
        }
        cooling_Eq = run_constdensity_model(
            params_for_constdensity, UVB_flag=UVB_flag, 
            Compton_Xray_flag=Compton_Xray_flag, dynamic_final_flag=dynamic_final_flag,
            converge_when_setup=False,
        )

        normalized_cooling = cooling_Eq["cooling_rate"].v
        cooling_rate = normalized_cooling * nH**2
        tot_cooling_rate = cooling_rate * volume_vir_cm3

        #debug
        # if(debug_print):
        #     print(f"Mvir: {Mvir:.3e} Msun/h")
        #     print("nH: ", nH)
        #     print("mass_density: ", mass_density)
        #     print("temperature: ", temperature)
        #     print("Cooling rate: ", cooling_rate)
        #     print("Cooling rate (normalized): ", normalized_cooling)
        #     print("Volume virial: ", volume_vir_cm3)
        #     print("Total cooling rate: ", tot_cooling_rate)
        
        # Create result dictionary with all input parameters and the cooling rate
        result = -tot_cooling_rate  # Add the cooling rate
        
        results.append(result)
    
    return results


def get_EqCoolingDensity(r_Rvir, Mvir, redshift, concentration_model, param_sets):
    #r_Rir: ratio of r/Rvir
    #Mvir in Msun/h
    #param_sets: list of dictionaries with keys 'gas_metallicity' and 'f_H2'
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
 
    all_cooling_Eq_NFW_results = []
    all_cooling_Eq_core_results = []
    for params in param_sets:
        gas_metallicity = params['gas_metallicity']
        f_H2 = params['f_H2']

        params_for_constdensity = {
            "evolve_cooling": False,
            "redshift": redshift,
            "lognH": local_lognH_NFW,
            "specific_heating_rate": specific_heating_rate,
            "volumetric_heating_rate": volumetric_heating_rate,
            "temperature": temperature,
            "gas_metallicity": gas_metallicity,
            "f_H2": f_H2,
        }
        
        cooling_Eq_NFW = run_constdensity_model(
            params_for_constdensity, UVB_flag=UVB_flag, 
            Compton_Xray_flag=Compton_Xray_flag, dynamic_final_flag=dynamic_final_flag,
            converge_when_setup=True,
        )

        params_for_constdensity["lognH"] = local_lognH_core

        cooling_Eq_core = run_constdensity_model(
            params_for_constdensity, UVB_flag=UVB_flag, 
            Compton_Xray_flag=Compton_Xray_flag, dynamic_final_flag=dynamic_final_flag,
            converge_when_setup=True,
        )
        
        all_cooling_Eq_NFW_results.append(cooling_Eq_NFW["cooling_rate"].v*local_nH_NFW_cm3**2)
        all_cooling_Eq_core_results.append(cooling_Eq_core["cooling_rate"].v*local_nH_core_cm3**2)

    return all_cooling_Eq_NFW_results, all_cooling_Eq_core_results
                                    

    
"""
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
"""


def get_peak_cosmic_DFheating(redshift):
    lgM_limits = [4, 14]  # Limits for log10(M [Msun/h])
    if (redshift < 6.0):
        lgM_limits = [4, 16]

    lgM_list = np.linspace(lgM_limits[0], lgM_limits[1],50)
    bin_centers = lgM_list
    bin_width = bin_centers[1] - bin_centers[0]  # Assuming uniform bin spacing
    bin_edges = np.zeros(len(bin_centers) + 1)
    bin_edges[:-1] = bin_centers - bin_width/2
    bin_edges[-1] = bin_centers[-1] + bin_width/2
    bin_widths = np.diff(bin_edges)
    lgx_min_2_list = np.array([np.log10(1e-2) for j in range(len(lgM_list))])
    lgx_max_0_list = np.array([np.log10(1.0) for j in range(len(lgM_list))])
    data_min2_max0 = get_heating_per_lgM(lgM_list, lgx_min_2_list, lgx_max_0_list, redshift, 'BestFit_z')

    peak_index = np.argmax(data_min2_max0['Heating_perlgM'])
    peak_lgM = data_min2_max0['lgM_list'][peak_index]
    print(f"Peak heating at lgM = {peak_lgM:.2f} for minM=1e-2, maxM=1.0 at z={redshift:.2f}")
    return peak_lgM

def plot_peak_lgM_cosmic_DFheating():
    output_dir = '/home/zwu/21cm_project/unified_model/Analytic_results/cosmic_DFheating'
    z_list = np.linspace(15, 0, 50)
    peak_lgM_list = []
    for z in z_list:
        peak_lgM = get_peak_cosmic_DFheating(z)
        peak_lgM_list.append(peak_lgM)
    peak_lgM_list = np.array(peak_lgM_list)
    fig, ax1 = plt.subplots(figsize=(8, 6), facecolor='white')
    ax1.plot(z_list, peak_lgM_list, 'r-', label='Peak lgM')
    ax1.xaxis.set_inverted(True) 
    ax1.set_xlabel('Redshift', fontsize=14)
    ax1.set_ylabel('Peak lgM [Msun/h]', fontsize=14)
    ax1.set_yscale('linear')
    ax1.legend()
    filename = os.path.join(output_dir,f"peak_lgM_cosmic_DFheating.png")
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_cosmic_DFheating(redshift, snapNum = None):
    #check contribution to heating
    # M_Jeans = get_M_Jeans(redshift)
    # print("Jeans mass: ",M_Jeans)
    print(f"plotting cosmic DF heating at z = {redshift:.2f} ...")

    lgM_limits = [6, 14]  # Limits for log10(M [Msun/h])
    if (redshift < 6.0):
        lgM_limits = [6, 15.5]

    lgM_list = np.linspace(lgM_limits[0], lgM_limits[1], 41)
    bin_centers = lgM_list
    bin_width = bin_centers[1] - bin_centers[0]  # Assuming uniform bin spacing
    bin_edges = np.zeros(len(bin_centers) + 1)
    bin_edges[:-1] = bin_centers - bin_width/2
    bin_edges[-1] = bin_centers[-1] + bin_width/2
    bin_widths = np.diff(bin_edges)

    #x = m/M
    #set Jeans mass as min subhalo mass (and test other values)
    # lgx_min_MJeans_list = np.array([np.log10(M_Jeans/10**lgM_list[j]) for j in range(len(lgM_list))])
    lgx_min_3_list = np.array([np.log10(1e-3) for j in range(len(lgM_list))])
    lgx_min_2_list = np.array([np.log10(1e-2) for j in range(len(lgM_list))])
    
    #require max subhalo ratio to be 0.1 to avoid major mergers (and test other values)
    lgx_max_0_list = np.array([np.log10(1.0) for j in range(len(lgM_list))])
    lgx_max_half_list = np.array([np.log10(0.5) for j in range(len(lgM_list))])
    lgx_max_1_list = np.array([np.log10(1.0e-1) for j in range(len(lgM_list))])
    lgx_max_2_list = np.array([np.log10(1.0e-2) for j in range(len(lgM_list))])

    # data_minMJeans_max1 = get_heating_per_lgM(lgM_list, lgx_min_MJeans_list, lgx_max_1_list, redshift, 'BestFit_z')
    data_min2_max1 = get_heating_per_lgM(lgM_list, lgx_min_2_list, lgx_max_1_list, redshift, 'BestFit_z')
    data_min3_max1 = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_1_list, redshift, 'BestFit_z')
    data_min2_max0 = get_heating_per_lgM(lgM_list, lgx_min_2_list, lgx_max_0_list, redshift, 'BestFit_z')
    data_min3_max0 = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_0_list, redshift, 'BestFit_z')
    data_min3_max0_Bosch16evolved = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_0_list, redshift, 'Bosch16evolved')
    data_min3_max0_Bosch16unevolved = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_0_list, redshift, 'Bosch16unevolved')

    #heating per logM (old version)
    '''
    ln_m_over_M_limits = [np.log(1e-3), np.log(1.0)]
    DF_heating_perlogM_old = []
    for logM in lgM_list:
        result, error = quad(integrand_oldversion, ln_m_over_M_limits[0], ln_m_over_M_limits[1], args=(logM, redshift, 'Bosch2016'))

        if (abs(error) > 0.01 * abs(result)):
            print(f"Warning: error in integration is large: {error} at z={redshift}, logM={logM}")
        DF_heating_perlogM_old.append(result)
    DF_heating_perlogM_old = np.array(DF_heating_perlogM_old)
    label_old = r'$m/M \in [10^{-3},1]$, Bosch16evolved'
    '''
    if snapNum is not None:
        print(f"also compare with TNG snap {snapNum} ...")
        base_dir = '/home/zwu/21cm_project/unified_model/TNG_results/'
        processed_file = os.path.join(base_dir, simulation_set, f'snap_{snapNum}', 
                                    f'processed_halos_snap_{snapNum}.h5')
        data = load_processed_data(processed_file)
        header = data.header
        print("box size: ", header['BoxSize']," ckpc/h")
        scale_factor = 1 / (1 + redshift)  
        boxsize = header['BoxSize'] / 1e3  # in cMpc/h
        box_volume = boxsize**3

        host_indices = data.subhalo_data['host_index'].value
        host_mass = data.halo_data['GroupMass'].value[host_indices]
        host_M200 = data.halo_data['Group_M_Crit200'].value[host_indices]
        host_M = host_mass
        sub_DFheating_fid = data.subhalo_data['DF_heating_fid'].value #J/s
        sub_mach = data.subhalo_data['mach_number'].value
        #set mach > 5 to be 5
        sub_mach_cut = np.clip(sub_mach, None, 5.0)
        Vt_rmin = 40  #debug: Vt/rmin = ?
        mach_DF_correction = np.array([Idf_Ostriker99_nosingularity_Vtrmin(mach, Vt_rmin) for mach in sub_mach_cut])
        sub_DFheating_with_mach = sub_DFheating_fid * mach_DF_correction

        #calculate heating per lgM for comparison later
        log_host_M = np.log10(host_M)
        heating_per_bin = np.zeros(len(bin_edges) - 1)
        count_per_bin = np.zeros(len(bin_edges) - 1, dtype=int)

        for i in range(len(host_M)):
            if bin_edges[0] <= log_host_M[i] < bin_edges[-1]:
                bin_idx = np.digitize(log_host_M[i], bin_edges) - 1
                heating_per_bin[bin_idx] += sub_DFheating_with_mach[i]
                count_per_bin[bin_idx] += 1
        heating_rate_per_lgM_TNG = heating_per_bin / bin_widths / box_volume  # J/s/(cMpc/h)^3/dex
        print(f"Total number of subhalos processed: {len(host_M)}")
        print(f"Subhalos assigned to bins: {count_per_bin.sum()}")
        print(f"Total DF heating: {heating_per_bin.sum():.2e} J/s")


    plot_datasets = [
        # {
        #     "data": data_min2_max1,
        #     "quantity_to_plot": 'Heating_perlgM',
        #     "color": 'orange',
        #     "label": r'$[10^{-2},10^{-1}]$ BestFit',
        #     "linestyle": '-',
        #     "linewidth": 2
        # },
        # {
        #     "data": data_min3_max1,
        #     "quantity_to_plot": 'Heating_perlgM',
        #     "color": 'r',
        #     "label": r'$[10^{-3},10^{-1}]$ BestFit',
        #     "linestyle": ':',
        #     "linewidth": 2
        # },
        {
            "data": data_min2_max0,
            "quantity_to_plot": 'Heating_perlgM_totHMF',
            "color": 'orange',
            "label": r'$[10^{-2},1]$ BestFit',
            "linestyle": '-',
            "linewidth": 4.5
        },

        {
            "data": data_min3_max0,
            "quantity_to_plot": 'Heating_perlgM',
            "color": 'r',
            "label": r'$[10^{-3},1]$ BestFit (reduced HMF)',
            "linestyle": ':',
            "linewidth": 4.5
        },
        {
            "data": data_min3_max0,
            "quantity_to_plot": 'Heating_perlgM_totHMF',
            "color": 'r',
            "label": r'$[10^{-3},1]$ BestFit',
            "linestyle": '--',
            "linewidth": 4.5
        },
        {
            "data": data_min3_max0_Bosch16evolved,
            "quantity_to_plot": 'Heating_perlgM_totHMF',
            "color": 'grey',
            "label": r'$[10^{-3},1]$ Bosch16evolved',
            "linestyle": '-',
            "linewidth": 1.5
        },
        {
            "data": data_min3_max0_Bosch16unevolved,
            "quantity_to_plot": 'Heating_perlgM_totHMF',
            "color": 'grey',
            "label": r'$[10^{-3},1]$ Bosch16unevolved',
            "linestyle": '--',
            "linewidth": 1.5
        }
    ]

    # selected_heating_datasets_index = [0, 1, 2, 3, 4, 5, 6]
    # heating_datasets = [plot_datasets[i] for i in selected_heating_datasets_index]
    heating_datasets = plot_datasets

    output_dir = '/home/zwu/21cm_project/unified_model/Analytic_results/cosmic_DFheating'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir,f"DF_heating_perlogM_z{redshift:.2f}_totHMF.png")
    fig = plt.figure(facecolor='white')
    ax = fig.gca()
    for dataset in heating_datasets:
        quantity_to_plot = dataset["quantity_to_plot"]
        plt.plot(dataset["data"]["lgM_list"], 1e7*dataset["data"][quantity_to_plot]*scale_factor**3, 
                color=dataset["color"], 
                label=dataset["label"],
                linestyle=dataset["linestyle"], 
                linewidth=dataset["linewidth"])
    #also compare with TNG if snapNum is not None
    if snapNum is not None:
        plt.bar(bin_centers, 1e7*heating_rate_per_lgM_TNG, width=bin_width, color='greenyellow', alpha=0.7, 
        label='TNG50-1', edgecolor='lime', align='center')


    #also compare with old version
    # plt.plot(lgM_list,1e7*DF_heating_perlogM_old,'k-',label=label_old)
    gas_resolution, dark_matter_resolution = get_simulation_resolution('TNG50-1')
    ax.tick_params(axis='both', which = 'both', direction='in')
    plt.axvline(np.log10(100*dark_matter_resolution), color='k', linestyle='--',label=r'100 m$_{\mathrm{DM}}$')
    plt.legend()
    plt.xlim([min(lgM_list),max(lgM_list)])
    plt.ylim([1e33,1e41])
    plt.yscale('log')
    plt.ylabel(r'DF heating per lgM [erg/s (cMpc/h)$^{-3}$ dex$^{-1}$]',fontsize=14)
    plt.xlabel(r'lgM [M$_{\odot}$/h]',fontsize=14)
    plt.savefig(filename,dpi=300)

    print("Figure saved to: ", filename)
    plt.close()

#compare cooling and DF heating for massive halos at low-z
def plot_global_heating_cooling_singlehost(redshift, min_lgM, max_lgM):

    print(f"plotting DF heating and cooling in a single host halo at z = {redshift:.2f} ...")
    lgM_limits = [min_lgM, max_lgM]  # Limits for log10(M [Msun/h])
    lgM_list = np.linspace(lgM_limits[0], lgM_limits[1],50)
    #x = m/M
    #set Jeans mass as min subhalo mass (and test other values)
    # lgx_min_MJeans_list = np.array([np.log10(M_Jeans/10**lgM_list[j]) for j in range(len(lgM_list))])
    # lgx_min_2_list = np.array([np.log10(1e-2) for j in range(len(lgM_list))])
    lgx_min_3_list = np.array([np.log10(1e-3) for j in range(len(lgM_list))])
    
    #require max subhalo ratio to be 0.1 to avoid major mergers (and test other values)
    lgx_max_0_list = np.array([np.log10(1.0) for j in range(len(lgM_list))])
    lgx_max_1_list = np.array([np.log10(1.0e-1) for j in range(len(lgM_list))])

    print("Calculating DF heating for host halo ...")
    # data_minMJeans_max1 = get_heating_per_lgM(lgM_list, lgx_min_MJeans_list, lgx_max_1_list, redshift, 'BestFit_z')
    data_min3_max1 = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_1_list, redshift, 'BestFit_z')
    data_min3_max0 = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_0_list, redshift, 'BestFit_z')
    data_min3_max0_Bosch16evolved = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_0_list, redshift, 'Bosch16evolved')
    data_min3_max0_Bosch16unevolved = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_0_list, redshift, 'Bosch16unevolved')

    # same as data_min3_max0, but with fg = 0.05
    data_min3_max0_fg005 = copy.deepcopy(data_min3_max0)
    data_min3_max0_fg005['Heating_perlgM'] *= (0.05/(Omega_b/Omega_m))
    data_min3_max0_fg005['Heating_singlehost'] *= (0.05/(Omega_b/Omega_m))

    plot_heating_datasets = [
        {
            "data": data_min3_max1,
            "color": 'r',
            "label": r'$[10^{-3},10^{-1}]$ BestFit, f$_g = \Omega_b/\Omega_m$',
            "linestyle": ':',
            "linewidth": 2
        },
        {
            "data": data_min3_max0,
            "color": 'r',
            "label": r'$[10^{-3},1]$ BestFit, f$_g = \Omega_b/\Omega_m$',
            "linestyle": ':',
            "linewidth": 4
        },
        {
            "data": data_min3_max0_fg005,
            "color": 'orange',
            "label": r'$[10^{-3},1]$ BestFit, f$_g$=0.05',
            "linestyle": ':',
            "linewidth": 4
        },
        {
            "data": data_min3_max0_Bosch16evolved,
            "color": 'grey',
            "label": r'$[10^{-3},1]$ Bosch16evolved, f$_g = \Omega_b/\Omega_m$',
            "linestyle": '-',
            "linewidth": 2
        },
        {
            "data": data_min3_max0_Bosch16unevolved,
            "color": 'grey',
            "label": r'$[10^{-3},1]$ Bosch16unevolved, f$_g = \Omega_b/\Omega_m$',
            "linestyle": '--',
            "linewidth": 2
        }
    ]
    
    #then calculate cooling rates
    print("calculating cooling rates for host halo ...")
    Z_Dekel = 0.3*10**(-0.17*redshift)
    cooling_param_sets = [
        {"gas_metallicity": 1.0e-6, "f_H2": 0.0},
        {"gas_metallicity": 1.0e-2, "f_H2": 0.0},
        {"gas_metallicity": Z_Dekel, "f_H2": 0.0},
        {"gas_metallicity": Z_Dekel, "f_H2": 0.0},

        # {"gas_metallicity": 1.0e-6, "f_H2": 1.0e-5},
        # {"gas_metallicity": 1.0e-6, "f_H2": 1.0e-4},
        # {"gas_metallicity": 1.0e-6, "f_H2": 1.0e-3},
        # {"gas_metallicity": 1.0e-2, "f_H2": 1.0e-4},
    ]
    colors = ['blue','blue','blue','cyan','deepskyblue','cyan','lime','cyan']
    markers = ['o','s','^','^','o','o','o','s']
    markersizes = [20, 20, 20, 20, 10, 10, 10, 10]
    fg_cooling = [0.05, 0.05, 0.05, (Omega_b/Omega_m), 0.05, 0.05, 0.05, 0.05]
    fg_correction = [fg/(Omega_b/Omega_m) for fg in fg_cooling]
    fg_correction_sq = [fg**2 for fg in fg_correction]

    cooling_results = []
    
    for lgM in lgM_list:
        Mvir = 10**lgM
        c = get_concentration(Mvir/h_Hubble, redshift, 'ludlow16')
        profile_correction_for_cooling = c**3*(c**2+5*c+10)/30/(c+1)**5 * c**3/3/f_core(c)**2
        profile_correction_Dekel08 = c**3/(90*f_core(c))
        # print("lgM: ",lgM, "c: ",c, "profile_correction_for_cooling: ",profile_correction_for_cooling)
        # print("profile_correction_Dekel08_approx: ",profile_correction_Dekel08)
        
        cooling_result = get_EqCooling_for_single_host(Mvir, redshift, cooling_param_sets)

        #debug: the effect of cooling density integral
        #use Dekel08 profile correction for now
        cooling_result = np.array(cooling_result) * profile_correction_for_cooling 
        
        cooling_results.append(cooling_result)
    cooling_results = np.array(cooling_results) #shape: (len(lgM_list), len(cooling_param_sets))

    #fg correction
    for i in range(len(cooling_param_sets)):
        cooling_results[:, i] *= fg_correction_sq[i]


    def get_cooling_label(metallicity, f_H2, fg):
        if fg == Omega_b/Omega_m:
            fg_label = r'f$_g=\Omega_b/\Omega_m$'
        else:
            fg_label = rf'f$_g={fg:.2f}$'

        if f_H2 == 0.0:
            return f'Z={metallicity:.1e} Zsun' + ', ' + fg_label
        else:
            return f'Z={metallicity:.1e} Zsun, f_H2={f_H2:.1e}' + ', ' + fg_label

    plot_cooling_datasets = [
        {
            "data": cooling_results[:, i],
            "color": colors[i],
            "label": get_cooling_label(cooling_param_sets[i]["gas_metallicity"], cooling_param_sets[i]["f_H2"], fg_cooling[i]),
            "marker": markers[i],
            "markersize": markersizes[i],
        } for i in range(len(cooling_param_sets))
    ]
    

    #if z <= 2: also compare with Dekel08 heating and cooling rates
    if redshift <= 8.0:
        fg = 0.05
        fc = 0.05
        massive_lgM_list = np.linspace(11, 15, 50)
        heating_Dekel08_fid = np.array([get_heating_Dekel08(10**lgM/h_Hubble, redshift, fc) for lgM in massive_lgM_list])
        cooling_Dekel08_fid = np.array([get_cooling_Dekel08(10**lgM/h_Hubble, redshift, fg) for lgM in massive_lgM_list])
        print("cooling_Dekel08_fid:", cooling_Dekel08_fid)


    selected_heating_datasets_index = [0, 1, 2, 3, 4]
    heating_datasets = [plot_heating_datasets[i] for i in selected_heating_datasets_index]
    output_dir = '/home/zwu/21cm_project/unified_model/Analytic_results/singlehost'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir,f"DF_heating_singlehost_z{redshift:.2f}_profilefgcorr.png")

    fig, ax1 = plt.subplots(figsize=(8, 6), facecolor='white')
    for dataset in heating_datasets:
        ax1.plot(dataset["data"]["lgM_list"], 1e7*dataset["data"]["Heating_singlehost"], 
                color=dataset["color"], 
                label=dataset["label"],
                linestyle=dataset["linestyle"], 
                linewidth=dataset["linewidth"])

    for dataset in plot_cooling_datasets:
        ax1.scatter(lgM_list, dataset["data"], 
                    edgecolors=dataset["color"],
                    facecolors='none',
                    label=dataset["label"],
                    marker=dataset["marker"],
                    s=dataset["markersize"],
                    )
        
    if redshift <= 8.0:
        ax1.plot(massive_lgM_list, heating_Dekel08_fid, color='crimson', linestyle='-', label=r'Heating Dekel08 (f$_c$ = 0.05)')    
        ax1.plot(massive_lgM_list, cooling_Dekel08_fid, 'b-', label=rf'Cooling Dekel08 (Z = {Z_Dekel:.1e} Zsun, f$_g$ = 0.05)')
    
    if redshift == 0.0:
        M_Kim2005 = Kim2005_result[0]*h_Hubble #Msun/h
        heating_Kim2005 = Kim2005_result[1] #erg/s
        ax1.scatter(np.log10(M_Kim2005), heating_Kim2005, color='purple', marker='*', s=100, label='DF heating Kim05')


    handles, labels = ax1.get_legend_handles_labels()
    heating_keywords = ['Heating', 'BestFit', 'Bosch', 'Kim']
    heating_handles_labels = [(h, l) for h, l in zip(handles, labels) if any(k in l for k in heating_keywords)]
    cooling_handles_labels = [(h, l) for h, l in zip(handles, labels) if not any(k in l for k in heating_keywords)]

    heating_handles, heating_labels = zip(*heating_handles_labels)
    cooling_handles, cooling_labels = zip(*cooling_handles_labels)
    legend1 = ax1.legend(heating_handles, heating_labels, loc='upper left', title='Heating')
    legend2 = ax1.legend(cooling_handles, cooling_labels, loc='lower right', title='Cooling')
    # Add back the first legend manually so it doesn't get overwritten
    ax1.add_artist(legend1)


    ax1.set_xlim([min(lgM_list),max(lgM_list)])
    ax1.set_yscale('log')
    ax1.set_ylabel(r'Cooling and Heating [erg/s]',fontsize=14)
    ax1.set_xlabel(r'lgM [M$_{\odot}$/h]',fontsize=14)
    ax1.tick_params(axis='both', direction='in')
    ax1.grid(alpha = 0.3)

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
    # ax2.set_xlabel(r'Virial Temperature [K] ($\mu$ ='+f'{mu})', fontsize=14)
    ax2.set_xlabel(r'Virial Temperature [K]', fontsize=14)
    ax2.tick_params(axis='x', direction='in')

    plt.tight_layout()
    plt.savefig(filename,dpi=300)




    #also save the heating and cooling results to a text file
    lgM_list = np.array(lgM_list)
    txt_filename = os.path.join(output_dir,f"singlehost_z{redshift:.2f}.txt")
    with open(txt_filename, 'w') as f:
        f.write("lgM, Heating_singlehost[erg/s](data_min3_max1, data_min3_max0, data_min3_max0_fg005), Cooling_singlehost[erg/s](Z=1e-6, Z=1e-2, Z=Z_Dekel; fg=0.05)\n")
        for i in range(len(lgM_list)):
            f.write(f"{float(lgM_list[i]):.2f}, "
                    f"{float(1e7 * data_min3_max1['Heating_singlehost'][i]):.2e}, "
                    f"{float(1e7 * data_min3_max0['Heating_singlehost'][i]):.2e}, "
                    f"{float(1e7 * data_min3_max0_fg005['Heating_singlehost'][i]):.2e}, "
                    f"{plot_cooling_datasets[0]['data'][i].item():.2e}, "
                    f"{plot_cooling_datasets[1]['data'][i].item():.2e}, "
                    f"{plot_cooling_datasets[2]['data'][i].item():.2e}\n")

                    

def run_heating_cooling_singlehost():
    # plot_global_heating_cooling_singlehost(0, 10, 15)
    # plot_global_heating_cooling_singlehost(1, 10, 15)
    # plot_global_heating_cooling_singlehost(2, 10, 15)
    # plot_global_heating_cooling_singlehost(3, 10, 14)
    # plot_global_heating_cooling_singlehost(4, 10, 14)
    # plot_global_heating_cooling_singlehost(5, 9, 13)
    # plot_global_heating_cooling_singlehost(6, 9, 13)
    # plot_global_heating_cooling_singlehost(7, 9, 12)
    plot_global_heating_cooling_singlehost(8, 9, 12)

def plot_heating_cooling_ratio_singlehost():
    z_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    base_dir = '/home/zwu/21cm_project/unified_model/Analytic_results/singlehost'
    All_results = []
    for z in z_list:
        txt_filename = os.path.join(base_dir,f"singlehost_z{z:.2f}.txt")
        #lgM, Heating_singlehost[erg/s](data_min3_max1, data_min3_max0, data_min3_max0_fg005), Cooling_singlehost[erg/s](Z=1e-6, Z=1e-2, Z=Z_Dekel; fg=0.05)

        data = np.loadtxt(txt_filename, skiprows=1, delimiter=',')

        lgM = data[:, 0]
        heating_min3_max0_fg005 = data[:, 3]
        cooling_Z_Dekel = data[:, 6]
        ratio_z = heating_min3_max0_fg005 / cooling_Z_Dekel
        All_results.append({
            "z": z,
            "lgM": lgM,
            "ratio": ratio_z
        })
    #plot the ratio
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    colors = plt.cm.rainbow(np.linspace(1, 0, len(z_list)))
    for result in All_results:
        z = result["z"]
        lgM = result["lgM"]
        ratio = result["ratio"]
        heating_Dekel08_fid = np.array([get_heating_Dekel08(10**x/h_Hubble, z, 0.05) for x in lgM])
        cooling_Dekel08_fid = np.array([get_cooling_Dekel08(10**x/h_Hubble, z, 0.05) for x in lgM])
        ratio_Dekel08 = heating_Dekel08_fid / cooling_Dekel08_fid

        ax.plot(lgM[lgM>11], ratio[lgM>11], color=colors[z], label=f'z={z:.2f}')
        ax.plot(lgM[lgM>11], ratio_Dekel08[lgM>11], color=colors[z], linestyle='--')
    solid_line = mlines.Line2D([], [], color='black', linestyle='-', label=r'This work (SHMF: [10$^{-3}$,1] BestFit, f$_g$=0.05)')
    dashed_line = mlines.Line2D([], [], color='black', linestyle='--', label=r"Dekel08 (f$_c$ = f$_g$ = 0.05)")
    #add Heating/Cooling = 1 line
    ax.axhline(y=1, color='grey', linestyle='-')
    ax.set_yscale('log')
    ax.set_xlabel(r'lgM [M$_{\odot}$/h]', fontsize=14)
    ax.set_ylabel(r'Heating/Cooling Ratio', fontsize=14)
    legend1 = ax.legend(loc='lower right', title='z', fontsize=13)  
    ax.add_artist(legend1)  
    legend2 = ax.legend(handles=[solid_line, dashed_line], loc='upper left', fontsize=13)

    ax.tick_params(axis='both', which='both', direction='in')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'heating_cooling_ratio.png'), dpi=300)



#compare cooling and DF heating for minihalos at high-z
def plot_global_heating_cooling_singlehost_minihalo(redshift):

    print(f"plotting DF heating and cooling in a minihalo at z = {redshift:.2f} ...")
    min_lgM = 6.0
    max_lgM = 7.5
    lgM_limits = [min_lgM, max_lgM]  # Limits for log10(M [Msun/h])
    lgM_list = np.linspace(lgM_limits[0], lgM_limits[1],50)
    #x = m/M
    #set Jeans mass as min subhalo mass (and test other values)
    lgx_min_3_list = np.array([np.log10(1e-3) for j in range(len(lgM_list))])
    
    #require max subhalo ratio to be 0.1 to avoid major mergers (and test other values)
    lgx_max_0_list = np.array([np.log10(1.0) for j in range(len(lgM_list))])
    lgx_max_1_list = np.array([np.log10(1.0e-1) for j in range(len(lgM_list))])

    data_min3_max1 = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_1_list, redshift, 'BestFit_z', mean_molecular_weight=mu_minihalo)
    data_min3_max0 = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_0_list, redshift, 'BestFit_z', mean_molecular_weight=mu_minihalo)

 
    plot_heating_datasets = [
        {
            "data": data_min3_max1,
            "color": 'r',
            "label": r'$[10^{-3},10^{-1}]$ BestFit, f$_g = \Omega_b/\Omega_m$',
            "linestyle": ':',
            "linewidth": 2
        },
        {
            "data": data_min3_max0,
            "color": 'r',
            "label": r'$[10^{-3},1]$ BestFit, f$_g = \Omega_b/\Omega_m$',
            "linestyle": ':',
            "linewidth": 4
        },
    ]
    

    #then calculate the cooling rates
    cooling_param_sets = [
        {"gas_metallicity": 0.0, "f_H2": 0.0, 
         "color": "lime", "marker": "o", "markersize":10,
         "label":"H2 = 0"},
        {"gas_metallicity": 0.0, "f_H2": 1.0e-6,
         "color": "cyan", "marker": "o", "markersize":10,
         "label": "H2 = 1e-6"},
        {"gas_metallicity": 0.0, "f_H2": 1.0e-5,
         "color": "deepskyblue", "marker": "o", "markersize":10,
         "label": "H2 = 1e-5"},
        {"gas_metallicity": 0.0, "f_H2": 1.0e-4,
         "color": "blue", "marker": "o", "markersize":10,
         "label": "H2 = 1e-4"},
    ]

    fg_cooling = (Omega_b/Omega_m)*np.ones(len(cooling_param_sets))
    fg_correction = fg_cooling/(Omega_b/Omega_m) 
    fg_correction_sq = fg_correction**2

    cooling_results = []
    
    for lgM in lgM_list:
        Mvir = 10**lgM
        c = get_concentration(Mvir/h_Hubble, redshift, 'ludlow16')
        profile_correction_for_cooling = c**3*(c**2+5*c+10)/30/(c+1)**5 * c**3/3/f_core(c)**2
        profile_correction_Dekel08 = c**3/(90*f_core(c))
        print("lgM: ",lgM, "c: ",c, "profile_correction_for_cooling: ",profile_correction_for_cooling)
        
        cooling_result = get_EqCooling_for_single_host(Mvir, redshift, cooling_param_sets, mean_molecular_weight=mu_minihalo)

        #debug: the effect of cooling density integral
        #use Dekel08 profile correction for now
        cooling_result = np.array(cooling_result) * profile_correction_for_cooling 
        
        cooling_results.append(cooling_result)
    cooling_results = np.array(cooling_results) #shape: (len(lgM_list), len(cooling_param_sets))

    #fg correction
    for i in range(len(cooling_param_sets)):
        cooling_results[:, i] *= fg_correction_sq[i]
    

    selected_heating_datasets_index = [0, 1]
    heating_datasets = [plot_heating_datasets[i] for i in selected_heating_datasets_index]
    output_dir = '/home/zwu/21cm_project/unified_model/Analytic_results/singlehost_minihalo'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir,f"DF_heating_singlehost_z{redshift:.2f}_profilefgcorr.png")



    fig, ax1 = plt.subplots(figsize=(8, 6), facecolor='white')
    #plot heating
    for dataset in heating_datasets:
        ax1.plot(dataset["data"]["lgM_list"], 1e7*dataset["data"]["Heating_singlehost"], 
                color=dataset["color"], 
                label=dataset["label"],
                linestyle=dataset["linestyle"], 
                linewidth=dataset["linewidth"])
    #plot cooling
    for i, params in enumerate(cooling_param_sets):
        ax1.scatter(
            lgM_list,                      # x
            cooling_results[:, i],         # y
            edgecolors=params.get("color", "C0"),
            facecolors="none",
            label=params.get("label") ,
            marker=params.get("marker", "o"),
            s=params.get("markersize", 8),  
        )

    handles, labels = ax1.get_legend_handles_labels()
    heating_keywords = ['Heating', 'BestFit', 'Bosch', 'Kim']
    heating_handles_labels = [(h, l) for h, l in zip(handles, labels) if any(k in l for k in heating_keywords)]
    cooling_handles_labels = [(h, l) for h, l in zip(handles, labels) if not any(k in l for k in heating_keywords)]

    heating_handles, heating_labels = zip(*heating_handles_labels)
    cooling_handles, cooling_labels = zip(*cooling_handles_labels)
    legend1 = ax1.legend(heating_handles, heating_labels, loc='upper left', title='Heating')
    legend2 = ax1.legend(cooling_handles, cooling_labels, loc='lower right', title='Cooling')
    # Add back the first legend manually so it doesn't get overwritten
    ax1.add_artist(legend1)


    ax1.set_xlim([min(lgM_list),max(lgM_list)])
    ax1.set_yscale('log')
    ax1.set_ylabel(r'Cooling and Heating [erg/s]',fontsize=14)
    ax1.set_xlabel(r'lgM [M$_{\odot}$/h]',fontsize=14)
    ax1.tick_params(axis='both', direction='in')
    ax1.grid(alpha = 0.3)

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())

    # Define clean Tvir ticks (integer powers of 10)
    Tvir_min = lgM_to_Tvir(min(lgM_list), redshift, mean_molecular_weight=mu_minihalo)
    Tvir_max = lgM_to_Tvir(max(lgM_list), redshift, mean_molecular_weight=mu_minihalo)
    Tvir_locator = LogLocator(base=10)
    Tvir_ticks = Tvir_locator.tick_values(Tvir_min, Tvir_max)
    lgM_ticks_top = [Tvir_to_lgM(Tvir, redshift, mean_molecular_weight=mu_minihalo) for Tvir in Tvir_ticks]

    # Filter valid ticks within the plot limits
    valid_ticks = [(lgM, Tvir) for lgM, Tvir in zip(lgM_ticks_top, Tvir_ticks) if min(lgM_list) <= lgM <= max(lgM_list)]
    lgM_ticks_top, Tvir_ticks = zip(*valid_ticks)

    # Set the ticks and labels
    ax2.set_xticks(lgM_ticks_top)
    ax2.set_xticklabels([f"$10^{int(np.log10(Tvir))}$" for Tvir in Tvir_ticks])
    # ax2.set_xlabel(r'Virial Temperature [K] ($\mu$ ='+f'{mu})', fontsize=14)
    ax2.set_xlabel(r'Virial Temperature [K]', fontsize=14)
    ax2.tick_params(axis='x', direction='in')

    plt.tight_layout()
    plt.savefig(filename,dpi=300)



    """
    #also save the heating and cooling results to a text file
    lgM_list = np.array(lgM_list)
    txt_filename = os.path.join(output_dir,f"singlehost_z{redshift:.2f}.txt")
    with open(txt_filename, 'w') as f:
        f.write("lgM, Heating_singlehost[erg/s](data_min3_max1, data_min3_max0, data_min3_max0_fg005), Cooling_singlehost[erg/s](Z=1e-6, Z=1e-2, Z=Z_Dekel; fg=0.05)\n")
        for i in range(len(lgM_list)):
            f.write(f"{float(lgM_list[i]):.2f}, "
                    f"{float(1e7 * data_min3_max1['Heating_singlehost'][i]):.2e}, "
                    f"{float(1e7 * data_min3_max0['Heating_singlehost'][i]):.2e}, "
                    f"{float(1e7 * data_min3_max0_fg005['Heating_singlehost'][i]):.2e}, "
                    f"{plot_cooling_datasets[0]['data'][i].item():.2e}, "
                    f"{plot_cooling_datasets[1]['data'][i].item():.2e}, "
                    f"{plot_cooling_datasets[2]['data'][i].item():.2e}\n")

                    
    """

if __name__ == "__main__":
    
    #1. cosmic DF heating (integrate over HMF)
    z_list = [15, 12, 10, 8, 6, 3, 0]
    snapNum_list = [1, 2, 4, 8, 13, 25, 99]
   
    for z, snapNum in zip(z_list, snapNum_list):
        plot_cosmic_DFheating(z, snapNum)
    # plot_peak_lgM_cosmic_DFheating()

    #2. cimpare heating and cooling for a single host halo
    # run_heating_cooling_singlehost()
    # plot_heating_cooling_ratio_singlehost()

    # plot_global_heating_cooling_singlehost_minihalo(redshift = 15)

