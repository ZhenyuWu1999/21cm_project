
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
from scipy.integrate import quad
from matplotlib.ticker import LogLocator, LogFormatter, MultipleLocator
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

def get_heating_per_lgM(lgM_list, lgx_min_list, lgx_max_list, redshift, SHMF_model, mean_molecular_weight=mu, PoissonSamplingFlag = False):
    '''
    return:
    data_dict = {'lgM_list':lgM_list,
                 'Heating_singlehost':Heating_singlehost [J/s],
                 'Heating_perlgM':Heating_perlgM [J/s (Mpc/h)$^{-3}$ dex$^{-1}$]}
    '''
    Heating_singlehost = []
    Heating_perlgM = []
    Heating_perlgM_totHMF = []
    if PoissonSamplingFlag:
        # Percentiles corresponding to ±1, ±2, ±3 sigma ranges
        percentiles = [16, 84, 2.5, 97.5, 0.15, 99.85]
        hs_perc = {p: [] for p in percentiles}
        hs_mean = []  # Store mean heating rate from Poisson samples
        hs_median = []


    for index, lgM in enumerate(lgM_list):
        lgx_min = lgx_min_list[index]
        lgx_max = lgx_max_list[index]
        heating = integrate_SHMF_heating_for_single_host(redshift, lgx_min, lgx_max, lgM, SHMF_model, mean_molecular_weight)

        dN_dlgM = HMF_2Dbestfit(lgM, redshift, include_selection_factor=True)
        dN_dlgM_totHMF = HMF_2Dbestfit(lgM, redshift, include_selection_factor=False)
        Heating_singlehost.append(heating)
        Heating_perlgM.append(heating*dN_dlgM)
        Heating_perlgM_totHMF.append(heating*dN_dlgM_totHMF)

        if PoissonSamplingFlag:  #use Poisson sampling of SHMF to see the variance of heating
            n_samples = 500
            heating_of_samples = integrate_SHMF_heating_for_single_host_PoissonSampling(redshift,
                                lgx_min, lgx_max, lgM, SHMF_model, n_samples, mean_molecular_weight=mean_molecular_weight, verbose=True)


            heating_of_samples = np.asarray(heating_of_samples).reshape(-1)
            hs_mean.append(np.mean(heating_of_samples))
            hs_median.append(np.median(heating_of_samples))
            # Compute and store required percentiles
            pct_vals = np.percentile(heating_of_samples, percentiles)
            for p, v in zip(percentiles, pct_vals):
                hs_perc[p].append(v)



    Heating_singlehost = np.array(Heating_singlehost)
    Heating_perlgM = np.array(Heating_perlgM)
    Heating_perlgM_totHMF = np.array(Heating_perlgM_totHMF)

    data_dict = {'lgM_list':lgM_list,
                 'Heating_singlehost':Heating_singlehost,
                 'Heating_perlgM':Heating_perlgM,
                 'Heating_perlgM_totHMF':Heating_perlgM_totHMF}

    if PoissonSamplingFlag:
        # Add Poisson statistics results
        data_dict['Heating_singlehost_mean'] = np.array(hs_mean)
        data_dict['Heating_singlehost_median'] = np.array(hs_median)

        # Add ±1, ±2, ±3 sigma percentile values
        data_dict['Heating_singlehost_p16'] = np.array(hs_perc[16])
        data_dict['Heating_singlehost_p84'] = np.array(hs_perc[84])
        data_dict['Heating_singlehost_p2p5'] = np.array(hs_perc[2.5])
        data_dict['Heating_singlehost_p97p5'] = np.array(hs_perc[97.5])
        data_dict['Heating_singlehost_p0p15'] = np.array(hs_perc[0.15])
        data_dict['Heating_singlehost_p99p85'] = np.array(hs_perc[99.85])


    return data_dict

def create_heating_data_dict_with_fgas(data_dict, f_gas):
    ratio = f_gas / (Omega_b / Omega_m)
    heating_data_dict_with_fgas = copy.deepcopy(data_dict)
    for key in ['Heating_singlehost', 'Heating_perlgM', 'Heating_perlgM_totHMF']:
        heating_data_dict_with_fgas[key] = data_dict[key] * ratio
    #if PoissonSamplingFlag also scale those
    if 'Heating_singlehost_mean' in data_dict:
        additional_keys = ['Heating_singlehost_mean', 'Heating_singlehost_median',
        'Heating_singlehost_p16', 'Heating_singlehost_p84',
        'Heating_singlehost_p2p5', 'Heating_singlehost_p97p5',
        'Heating_singlehost_p0p15', 'Heating_singlehost_p99p85']
        for key in additional_keys:
            heating_data_dict_with_fgas[key] = data_dict[key] * ratio
    return heating_data_dict_with_fgas


def get_EqCooling_for_single_host(Mvir, redshift, param_sets, mean_molecular_weight=mu, converge_when_setup=True):
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
    print("lognH: ", lognH)
    print("nH: ", nH)

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
            converge_when_setup=converge_when_setup,
        )
        print("temperature:", temperature)
        print("initial H2 fraction:", f_H2)
        print("final H2 fraction:", cooling_Eq["H2I_density"].v/cooling_Eq["density"].v)

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
        result = float(-tot_cooling_rate)  # Add the cooling rate

        results.append(result)

    return results


def get_EqCooling_envelope_for_single_host_minihalo(
    Mvir,
    redshift,
    param_sets,
    mean_molecular_weight=mu,
    return_components=False,
):
    """
    Return the larger cooling rate from fixed initial species and equilibrium
    species setup. This keeps the molecular-cooling branch tied to the chosen
    initial H2 fraction while recovering atomic cooling above Tvir ~ 1e4 K.
    """
    cooling_converged = np.asarray(
        get_EqCooling_for_single_host(
            Mvir,
            redshift,
            param_sets,
            mean_molecular_weight=mean_molecular_weight,
            converge_when_setup=True,
        ),
        dtype=float,
    )
    cooling_fixed_species = np.asarray(
        get_EqCooling_for_single_host(
            Mvir,
            redshift,
            param_sets,
            mean_molecular_weight=mean_molecular_weight,
            converge_when_setup=False,
        ),
        dtype=float,
    )
    cooling_envelope = np.maximum(cooling_converged, cooling_fixed_species)

    if return_components:
        return cooling_envelope, cooling_converged, cooling_fixed_species
    return cooling_envelope


def get_cumulative_cooling_and_heating_withinradius_singlehost(
    Mvir,
    redshift,
    param_sets,
    radii_Rvir,
    alpha,
    concentration_model='ludlow16',
    f_gas=Omega_b/Omega_m,
    mean_molecular_weight=mu,
    converge_when_setup=True,
):
    """
    Return cumulative cooling within selected radii for one host halo.

    For now this function only computes cooling and returns the radii sorted
    in ascending order together with the cumulative cooling fraction
    C(<r)/C(<Rvir) and cumulative cooling rate.
    """
    radii_sorted = np.sort(np.atleast_1d(np.asarray(radii_Rvir, dtype=float)))
    if np.any(radii_sorted < 0):
        raise ValueError("radii_Rvir must be non-negative.")

    concentration_value = get_concentration(Mvir / h_Hubble, redshift, concentration_model)
    total_cooling_baseline = np.asarray(
        get_EqCooling_for_single_host(
            Mvir,
            redshift,
            param_sets,
            mean_molecular_weight=mean_molecular_weight,
            converge_when_setup=converge_when_setup,
        ),
        dtype=float,
    )

    cooling_fraction = np.asarray(
        get_cumulative_cooling_fraction(radii_sorted, concentration_value, alpha),
        dtype=float,
    )

    f_gas_array = np.asarray(f_gas, dtype=float)
    if f_gas_array.ndim == 0:
        f_gas_array = np.full(len(param_sets), float(f_gas_array))
    elif f_gas_array.shape != (len(param_sets),):
        raise ValueError("f_gas must be scalar or have the same length as param_sets.")
    fg_correction_sq = (f_gas_array / (Omega_b / Omega_m)) ** 2

    cumulative_cooling = np.outer(cooling_fraction, total_cooling_baseline * fg_correction_sq)

    return radii_sorted, cooling_fraction, cumulative_cooling


"""
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
    ax1.plot(z_list, peak_lgM_list, 'r-')
    ax1.xaxis.set_inverted(True)
    ax1.set_xlabel('Redshift', fontsize=14)
    ax1.set_ylabel(r'$\log_{10}(M_{peak}) [M_{\odot}/h]$', fontsize=14)
    ax1.set_yscale('linear')
    ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12)
    # ax1.legend()
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    filename = os.path.join(output_dir,f"peak_lgM_cosmic_DFheating.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print("Saved figure: ", filename)


def plot_cosmic_DFheating(redshift, snapNum = None, ax=None, show_legend=True, show_title=False, show_ylabel = True, save_fig=True):
    #check contribution to heating
    # M_Jeans = get_M_Jeans(redshift)
    # print("Jeans mass: ",M_Jeans)
    print(f"plotting cosmic DF heating at z = {redshift:.2f} ...")

    lgM_limits = [6, 15.5]  # Limits for log10(M [Msun/h])
    # if (redshift >= 6.0):
    #     lgM_limits = [6, 14]

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
    lgx_min_1_list = np.array([np.log10(0.1) for j in range(len(lgM_list))])

    #require max subhalo ratio to be 0.1 to avoid major mergers (and test other values)
    lgx_max_0_list = np.array([np.log10(1.0) for j in range(len(lgM_list))])
    lgx_max_half_list = np.array([np.log10(0.5) for j in range(len(lgM_list))])
    lgx_max_1_list = np.array([np.log10(1.0e-1) for j in range(len(lgM_list))])
    lgx_max_2_list = np.array([np.log10(1.0e-2) for j in range(len(lgM_list))])

    # data_minMJeans_max1 = get_heating_per_lgM(lgM_list, lgx_min_MJeans_list, lgx_max_1_list, redshift, 'BestFit_z')
    data_min2_max1 = get_heating_per_lgM(lgM_list, lgx_min_2_list, lgx_max_1_list, redshift, 'BestFit_z')
    data_min3_max1 = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_1_list, redshift, 'BestFit_z')
    data_min1_max0 = get_heating_per_lgM(lgM_list, lgx_min_1_list, lgx_max_0_list, redshift, 'BestFit_z')
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
            "data": data_min1_max0,
            "quantity_to_plot": 'Heating_perlgM_totHMF',
            "color": 'royalblue',
            "label": r'$[10^{-1},1]$ BestFit',
            "linestyle": '-',
            "linewidth": 2.2
        },

        {
            "data": data_min2_max0,
            "quantity_to_plot": 'Heating_perlgM_totHMF',
            "color": 'orange',
            "label": r'$[10^{-2},1]$ BestFit',
            "linestyle": '-',
            "linewidth": 2.2
        },

        {
            "data": data_min3_max0,
            "quantity_to_plot": 'Heating_perlgM',
            "color": 'r',
            "label": r'$[10^{-3},1]$ BestFit (reduced HMF)',
            "linestyle": ':',
            "linewidth": 2.2
        },
        {
            "data": data_min3_max0,
            "quantity_to_plot": 'Heating_perlgM_totHMF',
            "color": 'r',
            "label": r'$[10^{-3},1]$ BestFit',
            "linestyle": '--',
            "linewidth": 2.2
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
    fig = None
    if ax is None:
        fig = plt.figure(facecolor='white')
        ax = fig.gca()

    for dataset in heating_datasets:
        quantity_to_plot = dataset["quantity_to_plot"]
        ax.plot(dataset["data"]["lgM_list"], 1e7*dataset["data"][quantity_to_plot]*scale_factor**3,
                color=dataset["color"],
                label=dataset["label"],
                linestyle=dataset["linestyle"],
                linewidth=dataset["linewidth"])
    #also compare with TNG if snapNum is not None
    if snapNum is not None:
        # ax.bar(bin_centers, 1e7*heating_rate_per_lgM_TNG, width=bin_width, color='greenyellow', alpha=0.7, label='TNG50-1', edgecolor='lime', align='center')
        ax.bar(bin_centers, 1e7*heating_rate_per_lgM_TNG, width=bin_width, color='#A8DADC', edgecolor='#4C9A9A',alpha=0.7, label='TNG50-1', align='center')



    #also compare with old version
    # plt.plot(lgM_list,1e7*DF_heating_perlogM_old,'k-',label=label_old)
    gas_resolution, dark_matter_resolution = get_simulation_resolution('TNG50-1')
    ax.tick_params(axis='both', which = 'both', direction='in')
    ax.axvline(np.log10(100*dark_matter_resolution), color='k', linestyle='--',label=r'100 m$_{\mathrm{DM}}$')

    if show_legend:
        ax.legend()
    ax.set_xlim([min(lgM_list),max(lgM_list)])
    ax.set_ylim([1e33,1e41])
    ax.set_yscale('log')
    if show_ylabel:
        ax.set_ylabel(r'DF heating per lgM [erg/s (cMpc/h)$^{-3}$ dex$^{-1}$]',fontsize=14)
    else:
        ax.set_ylabel("")

    ax.set_xlabel(r'lgM [M$_{\odot}$/h]',fontsize=14)
    if show_title:
        ax.set_title(f"z = {redshift}")

    if ax is None:
        plt.tight_layout()
        if save_fig:
            plt.savefig(filename, dpi=300)
            print("Figure saved to: ", filename)
        plt.close()

def plot_cosmic_DFheating_multi_z(
    redshifts=(15, 12, 10, 8, 6, 3, 0),
    snapNums=(1, 2, 4, 8, 13, 25, 99),
    layout=None,
    sharex=False,
    sharey=True,
    output_dir='/home/zwu/21cm_project/unified_model/Analytic_results/cosmic_DFheating',
    filename_prefix='DF_heating_perlogM_multiZ'
):
    if layout is None:
        n = len(redshifts)
        if n <= 3:
            layout = (n, 1)
        elif n <= 4:
            layout = (2, 2)
        elif n <= 6:
            layout = (3, 2)
        else:
            layout = (4, 2)

    nrows, ncols = layout
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 5.0 * nrows),
        sharex=sharex,
        sharey=sharey,
        facecolor='white'
    )
    axes = np.atleast_1d(axes).flatten()

    for i, (z, snap) in enumerate(zip(redshifts, snapNums)):
        if i >= len(axes):
            break
        ax = axes[i]

        plot_cosmic_DFheating(
            z, snap,
            ax=ax,
            show_legend=(i == 0),
            show_title=False,
            show_ylabel=False,
            save_fig=False
        )

        ax.text(
            0.95, 0.08, f"z = {z}",
            transform=ax.transAxes,
            fontsize=12,
            va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='0.7')
        )

        ax.set_xticks(np.arange(6, 16, 1))


    for j in range(len(redshifts), len(axes)):
        axes[j].axis('off')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"{filename_prefix}_z" + "_".join(map(str, redshifts)) + ".png")

    fig.supylabel(r'DF heating per lgM [erg/s (cMpc/h)$^{-3}$ dex$^{-1}$]', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35)
    plt.savefig(filename, dpi=300)
    print(f"multiZ: Plot saved to {filename}")
    plt.close()


#compare cooling and DF heating for massive halos at low-z
def plot_global_heating_cooling_singlehost(redshift, min_lgM, max_lgM,
                                            ax=None,
                                            show_legend_heating=True,
                                            show_legend_cooling=True,
                                            show_tvir_axis=True,
                                            save_fig=True,
                                            xlim=None):

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
    data_min3_max0 = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_0_list, redshift, 'BestFit_z', mean_molecular_weight=mu, PoissonSamplingFlag=True)
    # data_min3_max0_Bosch16evolved = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_0_list, redshift, 'Bosch16evolved')
    # data_min3_max0_Bosch16unevolved = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_0_list, redshift, 'Bosch16unevolved')

    # same as data_min3_max0, but with fg = 0.05
    data_min3_max0_fg005 = create_heating_data_dict_with_fgas(data_min3_max0, f_gas=0.05)

    plot_heating_datasets = [
        # {
        #     "data": data_min3_max1,
        #     "color": 'r',
        #     "label": r'$[10^{-3},10^{-1}]$ BestFit, f$_g = \Omega_b/\Omega_m$',
        #     "linestyle": ':',
        #     "linewidth": 2
        # },
        # {
        #     "data": data_min3_max0,
        #     "color": 'r',
        #     "label": r'$[10^{-3},1]$ BestFit, f$_g = \Omega_b/\Omega_m$',
        #     "linestyle": ':',
        #     "linewidth": 4
        # },
        {
            "data": data_min3_max0_fg005,
            "color": 'red',
            "label": r'$[10^{-3},1]$ BestFit, f$_g$=0.05',
            "linestyle": ':',
            "linewidth": 4
        },
        # {
        #     "data": data_min3_max0_Bosch16evolved,
        #     "color": 'grey',
        #     "label": r'$[10^{-3},1]$ Bosch16evolved, f$_g = \Omega_b/\Omega_m$',
        #     "linestyle": '-',
        #     "linewidth": 2
        # },
        # {
        #     "data": data_min3_max0_Bosch16unevolved,
        #     "color": 'grey',
        #     "label": r'$[10^{-3},1]$ Bosch16unevolved, f$_g = \Omega_b/\Omega_m$',
        #     "linestyle": '--',
        #     "linewidth": 2
        # }
    ]

    #then calculate cooling rates
    print("calculating cooling rates for host halo ...")
    Z_Dekel = 0.3*10**(-0.17*redshift)
    cooling_param_sets = [
        {"gas_metallicity": 1.0e-3, "f_H2": 0.0},
        {"gas_metallicity": 1.0, "f_H2": 0.0},
        {"gas_metallicity": Z_Dekel, "f_H2": 0.0},
    ]
    colors = ['blue','blue','blue']
    # markers = ['o','s','^','^','o','o','o','s']
    # markersizes = [20, 20, 20, 20, 10, 10, 10, 10]
    linestyles = ['-','-','-']
    fg_cooling = [0.05, 0.05, 0.05]
    fg_correction = [fg/(Omega_b/Omega_m) for fg in fg_cooling]
    fg_correction_sq = [fg**2 for fg in fg_correction]

    cooling_results = []
    profile_type = 'core' #'core' or 'NFW'
    concentration_model = 'ludlow16' #remember to change the concentration model in Dekel08.py

    for lgM in lgM_list:
        Mvir = 10**lgM
        c = get_concentration(Mvir/h_Hubble, redshift, concentration_model)
        profile_correction_for_cooling = get_profile_corr_for_cooling(profile_type, c)
        # profile_correction_Dekel08 = c**3/(90*f_core(c)) only for testing Dekel08 approx
        # print("lgM: ",lgM, "c: ",c, "profile_correction_for_cooling: ",profile_correction_for_cooling)
        # print("profile_correction_Dekel08_approx: ",profile_correction_Dekel08)

        cooling_result = get_EqCooling_for_single_host(Mvir, redshift, cooling_param_sets, converge_when_setup=True)
        cooling_result = np.array(cooling_result) * profile_correction_for_cooling

        cooling_results.append(cooling_result)
    cooling_results = np.array(cooling_results) #shape: (len(lgM_list), len(cooling_param_sets))

    print("cooling results shape: ", cooling_results.shape)



    #fg correction
    for i in range(len(cooling_param_sets)):
        cooling_results[:, i] *= fg_correction_sq[i]
    print("cooling results shape: ", cooling_results.shape)



    def get_cooling_label(metallicity, f_H2, fg):
        if fg == Omega_b/Omega_m:
            fg_label = r'f$_g=\Omega_b/\Omega_m$'
        else:
            fg_label = rf'f$_g={fg:.2f}$'

        if f_H2 == 0.0:
            return f'Z={metallicity:.2f}' + r' Z$_{\odot}$' + ', ' + fg_label
        else:
            return f'Z={metallicity:.2f}' + r' Z$_{\odot}$' + f', f_H2={f_H2:.2f}' + ', ' + fg_label

    plot_cooling_datasets = [
        {
            "data": cooling_results[:, i],
            "color": colors[i],
            "label": get_cooling_label(cooling_param_sets[i]["gas_metallicity"], cooling_param_sets[i]["f_H2"], fg_cooling[i]),
            # "marker": markers[i],
            # "markersize": markersizes[i],
            "linestyle": linestyles[i],
        } for i in range(len(cooling_param_sets))
    ]


    #if z <= 2: also compare with Dekel08 heating and cooling rates
    if redshift <= 8.0:
        fg = 0.05
        fc = 0.05
        massive_lgM_list = np.linspace(11, max_lgM, 50)
        #modify the concentration model in Dekel08!
        heating_Dekel08_fid = np.array([get_heating_Dekel08(10**lgM/h_Hubble, redshift, fc) for lgM in massive_lgM_list])
        cooling_Dekel08_fid = np.array([get_cooling_Dekel08(10**lgM/h_Hubble, redshift, fg, profile_type) for lgM in massive_lgM_list])
        print("cooling_Dekel08_fid:", cooling_Dekel08_fid)


    # selected_heating_datasets_index = [0, 1, 2]
    # heating_datasets = [plot_heating_datasets[i] for i in selected_heating_datasets_index]
    heating_datasets = plot_heating_datasets
    output_dir = '/home/zwu/21cm_project/unified_model/Analytic_results/singlehost'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir,f"DF_heating_singlehost_z{redshift:.2f}_{profile_type}_{concentration_model}.png")

    if ax is None:
        fig, ax1 = plt.subplots(figsize=(8, 6), facecolor='white')
    else:
        ax1 = ax

    # for dataset in heating_datasets:
    #     ax1.plot(dataset["data"]["lgM_list"], 1e7*dataset["data"]["Heating_singlehost"],
    #             color=dataset["color"],
    #             label=dataset["label"],
    #             linestyle=dataset["linestyle"],
    #             linewidth=dataset["linewidth"])

    def add_SHMF_heating_variation_to_plot(ax1, data_SHMFvariation, color, heating_label):
        ax1.plot(data_SHMFvariation["lgM_list"],1e7*data_SHMFvariation["Heating_singlehost_mean"], color='red', linestyle='-', linewidth=1.5, label=heating_label+' Mean')
        ax1.plot(data_SHMFvariation["lgM_list"],1e7*data_SHMFvariation["Heating_singlehost_median"], color='red', linestyle=':', linewidth=1.5, label='Median')

        ax1.fill_between(data_SHMFvariation["lgM_list"],
                        1e7*data_SHMFvariation["Heating_singlehost_p16"],
                        1e7*data_SHMFvariation["Heating_singlehost_p84"],
                        alpha=0.4, color = color, label='16/84 %')
        ax1.fill_between(data_SHMFvariation["lgM_list"],
                        1e7*data_SHMFvariation["Heating_singlehost_p2p5"],
                        1e7*data_SHMFvariation["Heating_singlehost_p97p5"],
                        alpha=0.2, color = color, label='2.5/97.5 %')
        ax1.fill_between(data_SHMFvariation["lgM_list"],
                        1e7*data_SHMFvariation["Heating_singlehost_p0p15"],
                        1e7*data_SHMFvariation["Heating_singlehost_p99p85"],
                        alpha=0.1, color = color, label='0.15/99.85 %')


    # add_SHMF_heating_variation_to_plot(ax1, data_min3_max0, 'red')
    heating_label = r'$[10^{-3},1]$ BestFit, f$_g$=0.05'
    add_SHMF_heating_variation_to_plot(ax1, data_min3_max0_fg005, 'orange', heating_label)

    #fit the slope of heating rate
    data_SHMFvariation = data_min3_max0_fg005
    x_fit = np.array(data_SHMFvariation["lgM_list"])
    y_fit = np.log10(data_SHMFvariation["Heating_singlehost_mean"])
    heating_slope, intercept = np.polyfit(x_fit, y_fit, 1)
    print(f"Fitted heating slope: {heating_slope:.3f} at z={redshift}")


    # for dataset in plot_cooling_datasets:
    #     ax1.scatter(lgM_list, dataset["data"],
    #                 edgecolors=dataset["color"],
    #                 facecolors='none',
    #                 label=dataset["label"],
    #                 marker=dataset["marker"],
    #                 s=dataset["markersize"],
    #                 )
    selected_plot_cooling_datasets_index = [2]
    for i in selected_plot_cooling_datasets_index:
        dataset = plot_cooling_datasets[i]
        ax1.plot(lgM_list, dataset["data"],
                color=dataset["color"],
                label=dataset["label"],
                linestyle=dataset["linestyle"],
                )
    #fill between the metallicity range
    minZ = cooling_param_sets[0]["gas_metallicity"]; maxZ = cooling_param_sets[1]["gas_metallicity"]
    cooling_minZ = cooling_results[:,0]; cooling_maxZ = cooling_results[:,1]
    # cooling_Zrange_label = f'Z={minZ:.1e}-{maxZ:.1f}'+r' Z$_{\odot}$, f$_g$=0.05'
    cooling_Zrange_label = f'Z=0.001-1'+r' Z$_{\odot}$, f$_g$=0.05'

    ax1.fill_between(lgM_list, cooling_minZ, cooling_maxZ, color='blue', alpha=0.3, label=cooling_Zrange_label)

    if redshift <= 8.0:
        ax1.plot(massive_lgM_list, heating_Dekel08_fid, color='crimson', linestyle='--', label=r'Gravitational heating Dekel07 (f$_c$ = 0.05)')
        ax1.plot(massive_lgM_list, cooling_Dekel08_fid, linestyle='--', color = "darkblue", label=f'Cooling Dekel07 (Z = {Z_Dekel:.2f}'+ r' Z$_{\odot}$, f$_g$ = 0.05)')

    if redshift == 0.0:
        M_Kim2005 = Kim2005_result[0]*h_Hubble #Msun/h
        heating_Kim2005 = Kim2005_result[1] #erg/s
        ax1.scatter(np.log10(M_Kim2005), heating_Kim2005, color='purple', marker='*', s=100, label='DF heating Kim05')

    handles, labels = ax1.get_legend_handles_labels()
    heating_keywords = ['Heating', 'heating', 'BestFit', 'Bosch', 'Kim', 'Mean', 'Median', '%']

    heating_handles_labels = [(h, l) for h, l in zip(handles, labels) if any(k in l for k in heating_keywords)]
    cooling_handles_labels = [(h, l) for h, l in zip(handles, labels) if not any(k in l for k in heating_keywords)]

    legend1 = None
    if show_legend_heating and heating_handles_labels:
        heating_handles, heating_labels = zip(*heating_handles_labels)
        legend1 = ax1.legend(heating_handles, heating_labels, loc='upper left', title='Heating')

    if show_legend_cooling and cooling_handles_labels:
        cooling_handles, cooling_labels = zip(*cooling_handles_labels)
        legend2 = ax1.legend(cooling_handles, cooling_labels, loc='lower right', title='Cooling')

    if legend1 is not None:
        ax1.add_artist(legend1)


    '''
    handles, labels = ax1.get_legend_handles_labels()
    heating_keywords = ['Heating', 'heating', 'BestFit', 'Bosch', 'Kim', 'Mean', 'Median', '%']
    heating_handles_labels = [(h, l) for h, l in zip(handles, labels) if any(k in l for k in heating_keywords)]
    cooling_handles_labels = [(h, l) for h, l in zip(handles, labels) if not any(k in l for k in heating_keywords)]

    heating_handles, heating_labels = zip(*heating_handles_labels)
    cooling_handles, cooling_labels = zip(*cooling_handles_labels)
    legend1 = ax1.legend(heating_handles, heating_labels, loc='upper left', title='Heating')
    legend2 = ax1.legend(cooling_handles, cooling_labels, loc='lower right', title='Cooling')
    # Add back the first legend manually so it doesn't get overwritten
    ax1.add_artist(legend1)
    '''
    #add annotation of heating slope
    if max_lgM >=14.5:
        starting_lgM_for_annotation = 13.0
        starting_heating_for_annotation = 1.0e37
        ending_lgM_for_annotation = 14.5
    else:
        starting_lgM_for_annotation = 11.5
        starting_heating_for_annotation = 1.0e44
        ending_lgM_for_annotation = 12.5

    ending_heating_for_annotation = starting_heating_for_annotation * 10**(heating_slope * (ending_lgM_for_annotation - starting_lgM_for_annotation))
    ax1.plot([starting_lgM_for_annotation, ending_lgM_for_annotation],
            [starting_heating_for_annotation, ending_heating_for_annotation],
            color='black', linestyle='-', linewidth=1)
    text_for_annotation = "Heating ∝ M$^{" + f"{heating_slope:.2f}" + "}$"
    ax1.text((starting_lgM_for_annotation + ending_lgM_for_annotation)/2,
            (starting_heating_for_annotation + ending_heating_for_annotation)*0.1,
            text_for_annotation, fontsize=12, ha='center')
    if xlim is not None:
        ax1.set_xlim(xlim)
    else:
        ax1.set_xlim([min(lgM_list),max(lgM_list)])
    ax1.set_yscale('log')
    ax1.set_ylabel(r'Cooling and Heating [erg/s]',fontsize=14)
    ax1.set_xlabel(r'log$_{10}$ M [M$_{\odot}$/h]',fontsize=14)
    ax1.tick_params(axis='both', direction='in')
    ax1.grid(alpha = 0.3)

    if show_tvir_axis:
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        xlim = ax1.get_xlim()
        Tvir_min = lgM_to_Tvir(xlim[0], redshift)
        Tvir_max = lgM_to_Tvir(xlim[1], redshift)
        # Tvir_locator = LogLocator(base=10)
        # Tvir_ticks = Tvir_locator.tick_values(Tvir_min, Tvir_max)
        # lgM_ticks_top = [Tvir_to_lgM(Tvir, redshift) for Tvir in Tvir_ticks]
        # valid_ticks = [(lgM, Tvir) for lgM, Tvir in zip(lgM_ticks_top, Tvir_ticks)
        #             if min(lgM_list) <= lgM <= max(lgM_list)]


        # only use log10 based Tvir ticks
        exp_min = int(np.floor(np.log10(Tvir_min)))
        exp_max = int(np.ceil(np.log10(Tvir_max)))
        Tvir_ticks = [10**e for e in range(exp_min, exp_max + 1)]

        lgM_ticks_top = [Tvir_to_lgM(Tvir, redshift) for Tvir in Tvir_ticks]

        xlim = ax1.get_xlim()
        valid_ticks = [(lgM, Tvir) for lgM, Tvir in zip(lgM_ticks_top, Tvir_ticks)
                    if xlim[0] <= lgM <= xlim[1]]


        if valid_ticks:
            lgM_ticks_top, Tvir_ticks = zip(*valid_ticks)
            ax2.set_xticks(lgM_ticks_top)
            ax2.set_xticklabels([f"$10^{int(np.log10(Tvir))}$" for Tvir in Tvir_ticks])
        ax2.set_xlabel(r'Virial Temperature [K]', fontsize=14)
        ax2.tick_params(axis='x', direction='in')

    if ax is None:
        plt.tight_layout()
        if save_fig:
            plt.savefig(filename, dpi=300)
            print(f"Plot saved to {filename}")
        plt.close()



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

def run_heating_cooling_singlehost():
    # plot_global_heating_cooling_singlehost(0, 10, 15)
    # plot_global_heating_cooling_singlehost(1, 10, 15)
    # plot_global_heating_cooling_singlehost(2, 10, 15)
    # plot_global_heating_cooling_singlehost(3, 10, 14)
    # plot_global_heating_cooling_singlehost(4, 10, 13)
    # plot_global_heating_cooling_singlehost(5, 9, 13)
    plot_global_heating_cooling_singlehost(6, 9, 13)
    # plot_global_heating_cooling_singlehost(7, 9, 12)
    # plot_global_heating_cooling_singlehost(8, 9, 12)


def plot_global_heating_cooling_multi_z(
    redshifts=(0, 2, 6),
    layout=None,
    sharex=False,
    sharey=True,
    show_tvir_axis=True,
    output_dir='/home/zwu/21cm_project/unified_model/Analytic_results/singlehost',
    filename_prefix='DF_heating_singlehost_multiZ'
):
    # redshift -> (min_lgM, max_lgM)
    default_mass_ranges = {
        0: (10, 15),
        1: (10, 15),
        2: (10, 15),
        3: (10, 14),
        4: (10, 13),
        5: (9, 13),
        6: (9, 13),
        7: (9, 12),
        8: (9, 12),
    }

    if layout is None:
        n = len(redshifts)
        if n == 3:
            layout = (3, 1)
        elif n == 4:
            layout = (2, 2)
        else:
            layout = (n, 1)

    nrows, ncols = layout
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 5 * nrows),
        sharex=sharex,
        sharey=sharey,
        facecolor='white'
    )
    axes = np.atleast_1d(axes).flatten()

    for i, z in enumerate(redshifts):
        if i >= len(axes):
            break

        min_lgM, max_lgM = default_mass_ranges[z]
        ax = axes[i]

        plot_global_heating_cooling_singlehost(
            z, min_lgM, max_lgM,
            ax=ax,
            show_legend_heating=(z == 0),
            show_legend_cooling=True,
            show_tvir_axis=show_tvir_axis,
            save_fig=False
        )
        # ax.set_title(f"z = {z}")
        ax.text(
            0.05, 0.08, f"z = {z}",
            transform=ax.transAxes,
            fontsize=12,
            va='bottom', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='0.7')
        )


    # Turn off unused axes if layout has extras
    for j in range(len(redshifts), len(axes)):
        axes[j].axis('off')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.join(
        output_dir,
        f"{filename_prefix}_z" + "_".join(map(str, redshifts)) + ".png"
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35)
    plt.savefig(filename, dpi=300)
    print(f"multiZ: Plot saved to {filename}")
    plt.close()


"""
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

"""

#compare cooling and DF heating for minihalos at high-z
def plot_global_heating_cooling_singlehost_minihalo(redshift):

    print(f"plotting DF heating and cooling in a minihalo at z = {redshift:.2f} ...")
    min_lgM = 6.0
    max_lgM = 8.0
    lgM_limits = [min_lgM, max_lgM]  # Limits for log10(M [Msun/h])
    lgM_list = np.linspace(lgM_limits[0], lgM_limits[1],50)
    #x = m/M
    #set Jeans mass as min subhalo mass (and test other values)
    lgx_min_3_list = np.array([np.log10(1e-3) for j in range(len(lgM_list))])

    #require max subhalo ratio to be 0.1 to avoid major mergers (and test other values)
    lgx_max_0_list = np.array([np.log10(1.0) for j in range(len(lgM_list))])
    # lgx_max_1_list = np.array([np.log10(1.0e-1) for j in range(len(lgM_list))])

    # data_min3_max1 = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_1_list, redshift, 'BestFit_z', mean_molecular_weight=mu_minihalo)
    data_min3_max0 = get_heating_per_lgM(lgM_list, lgx_min_3_list, lgx_max_0_list, redshift, 'BestFit_z', mean_molecular_weight=mu_minihalo, PoissonSamplingFlag=True)

    #print the keys


    plot_heating_datasets = [
        # {
        #     "data": data_min3_max1,
        #     "color": 'r',
        #     "label": r'$[10^{-3},10^{-1}]$ BestFit, f$_g = \Omega_b/\Omega_m$',
        #     "linestyle": ':',
        #     "linewidth": 2
        # },
        {
            "data": data_min3_max0,
            "color": 'r',
            "label": r'$[10^{-3},1]$ BestFit, f$_g = \Omega_b/\Omega_m$',
            "linestyle": '-',
            "linewidth": 4
        },
    ]


    #then calculate the cooling rates
    cooling_param_sets = [
        # {"gas_metallicity": 0.0, "f_H2": 0.0,
        #  "color": "grey", "marker": "o", "markersize":10,
        #  "label":r"initial f$_{\mathrm{H}_2} = 0$"},
        {"gas_metallicity": 0.0, "f_H2": 1.0e-6,
         "color": "cyan", "marker": "o", "markersize":10,
         "label": r"f$_{\mathrm{H}_2} = 1e-6$"},
        {"gas_metallicity": 0.0, "f_H2": 1.0e-5,
         "color": "deepskyblue", "marker": "o", "markersize":10,
         "label": r"f$_{\mathrm{H}_2} = 1e-5$"},
        {"gas_metallicity": 0.0, "f_H2": 1.0e-4,
         "color": "royalblue", "marker": "o", "markersize":10,
         "label": r"f$_{\mathrm{H}_2} = 1e-4$"},
        #  {"gas_metallicity": 0.0, "f_H2": 1.0e-3,
        #   "color": "blue", "marker": "o", "markersize":10,
        #   "label": r"initial f$_{\mathrm{H}_2} = 1e-3$"},
    ]

    fg_cooling = (Omega_b/Omega_m)*np.ones(len(cooling_param_sets))
    fg_correction = fg_cooling/(Omega_b/Omega_m)
    fg_correction_sq = fg_correction**2

    cooling_results = []
    profile_type = 'core' #core or NFW
    concentration_model = 'ludlow16'
    for lgM in lgM_list:
        Mvir = 10**lgM
        c = get_concentration(Mvir/h_Hubble, redshift, concentration_model)
        profile_correction_for_cooling = get_profile_corr_for_cooling(profile_type, c)
        profile_correction_Dekel08 = c**3/(90*f_core(c))
        print("lgM: ",lgM, "c: ",c, "profile_correction_for_cooling: ",profile_correction_for_cooling)

        cooling_result = get_EqCooling_envelope_for_single_host_minihalo(
            Mvir,
            redshift,
            cooling_param_sets,
            mean_molecular_weight=mu_minihalo,
        )
        cooling_result = np.asarray(cooling_result) * profile_correction_for_cooling


        cooling_results.append(cooling_result)
    cooling_results = np.array(cooling_results) #shape: (len(lgM_list), len(cooling_param_sets))

    #fg correction
    for i in range(len(cooling_param_sets)):
        cooling_results[:, i] *= fg_correction_sq[i]


    selected_heating_datasets_index = [0]
    heating_datasets = [plot_heating_datasets[i] for i in selected_heating_datasets_index]
    output_dir = '/home/zwu/21cm_project/unified_model/Analytic_results/singlehost_minihalo'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir,f"DF_heating_singlehost_z{redshift:.2f}_profilefgcorr.png")



    fig, ax1 = plt.subplots(figsize=(8, 6), facecolor='white')
    #plot heating
    # for dataset in heating_datasets:
    #     ax1.plot(dataset["data"]["lgM_list"], 1e7*dataset["data"]["Heating_singlehost"],
    #             color=dataset["color"],
    #             label=dataset["label"],
    #             linestyle=dataset["linestyle"],
    #             linewidth=dataset["linewidth"])
    #also fill between the variation due to SHMF sampling
    def add_SHMF_heating_variation_to_plot(ax1, data_SHMFvariation, color):
        ax1.plot(data_SHMFvariation["lgM_list"],1e7*data_SHMFvariation["Heating_singlehost_mean"], color='red', linestyle='-', linewidth=1.5, label='Mean')
        ax1.plot(data_SHMFvariation["lgM_list"],1e7*data_SHMFvariation["Heating_singlehost_median"], color='red', linestyle=':', linewidth=1.5, label='Median')
        ax1.fill_between(data_SHMFvariation["lgM_list"],
                        1e7*data_SHMFvariation["Heating_singlehost_p16"],
                        1e7*data_SHMFvariation["Heating_singlehost_p84"],
                        alpha=0.4, color = color, label='16/84 %')
        ax1.fill_between(data_SHMFvariation["lgM_list"],
                        1e7*data_SHMFvariation["Heating_singlehost_p2p5"],
                        1e7*data_SHMFvariation["Heating_singlehost_p97p5"],
                        alpha=0.2, color = color, label='2.5/97.5 %')
        ax1.fill_between(data_SHMFvariation["lgM_list"],
                        1e7*data_SHMFvariation["Heating_singlehost_p0p15"],
                        1e7*data_SHMFvariation["Heating_singlehost_p99p85"],
                        alpha=0.1, color = color, label='0.15/99.85 %')
    add_SHMF_heating_variation_to_plot(ax1, data_min3_max0, 'orange')

    #plot cooling
    for i, params in enumerate(cooling_param_sets):
        ax1.plot(
            lgM_list,
            cooling_results[:, i],
            color=params.get("color", "C0"),
            marker=params.get("marker", "o"),
            markersize=params.get("markersize", 8),
            markerfacecolor="none",
            linestyle=params.get("linestyle", "-"),
            linewidth=params.get("linewidth", 1.5),
            label=params.get("label"),
        )


    handles, labels = ax1.get_legend_handles_labels()
    heating_keywords = ['Heating', 'heating', 'BestFit', 'Bosch', 'Kim', 'Mean', 'Median', '%']

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

    # --- major ticks: 10^n ---
    major_locator = LogLocator(base=10, subs=(1.0,))
    Tvir_major = major_locator.tick_values(Tvir_min, Tvir_max)
    # --- minor ticks: 2..9 × 10^n (gives 7000/8000/9000 and 2e4/3e4 etc.) ---
    minor_locator = LogLocator(base=10, subs=np.arange(2, 10))
    Tvir_minor = minor_locator.tick_values(Tvir_min, Tvir_max)
    Tvir_ticks = np.unique(np.concatenate([Tvir_major, Tvir_minor]))

    # map Tvir -> lgM positions
    lgM_ticks_top = np.array([
        Tvir_to_lgM(Tvir, redshift, mean_molecular_weight=mu_minihalo)
        for Tvir in Tvir_ticks
    ])
    mask = (lgM_ticks_top >= min(lgM_list)) & (lgM_ticks_top <= max(lgM_list))
    lgM_ticks_top = lgM_ticks_top[mask]
    Tvir_ticks = Tvir_ticks[mask]
    def fmt_Tvir(T):
        # check if T is an integer power of 10
        p = np.log10(T)
        if np.isclose(p, np.round(p), rtol=0, atol=1e-10):
            return rf"$10^{{{int(np.round(p))}}}$"          # 10^n
        else:
            return ""
        # non-major ticks: use 7000 / 2e4 style
        # if T < 1e4:
        #     return f"{int(np.round(T))}"
        # return f"{T:.0e}".replace("e+0", "e").replace("e+","e")  # 2e4, 3e4...

    # Set the ticks and labels
    ax2.set_xticks(lgM_ticks_top)
    ax2.set_xticklabels([fmt_Tvir(T) for T in Tvir_ticks])

    ax2.set_xlabel(r'Tvir [K]', fontsize=14)
    ax2.tick_params(axis='x', direction='in')

    plt.tight_layout()
    plt.savefig(filename,dpi=300)
    print(f"Plot saved to {filename}")
    plt.close()



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



def plot_modelA_cumulative_heating_cooling_minihalo(
    redshift=12.0,
    min_lgM=6.0,
    max_lgM=8.0,
    n_lgM=50,
    alpha=0.0,
    f_gas=Omega_b / Omega_m,
    concentration_model='ludlow16',
    output_dir='/home/zwu/21cm_project/unified_model/Analytic_HC_results_within_radius',
    include_shmf_scatter=True,
):
    """
    Plot cumulative Model A DF heating and cooling within selected radii.

    Heating uses the existing global SHMF heating amplitude multiplied by the
    Model A gas-profile cumulative fraction. Cooling starts from the uniform
    virial-density baseline and is multiplied by the cumulative cooling profile
    correction I_cool(<r), so the Rvir curve matches the global profile-corrected
    cooling rate.
    """
    print(
        f"plotting cumulative Model A heating/cooling in high-z minihalos: "
        f"z={redshift:.2f}, alpha={alpha:.2f}, f_gas={f_gas:.4f}, "
        f"SHMF scatter={include_shmf_scatter}"
    )
    os.makedirs(output_dir, exist_ok=True)

    lgM_list = np.linspace(min_lgM, max_lgM, n_lgM)
    lgx_min_list = np.full_like(lgM_list, np.log10(1.0e-3), dtype=float)
    lgx_max_list = np.full_like(lgM_list, np.log10(1.0), dtype=float)

    heating_data = get_heating_per_lgM(
        lgM_list,
        lgx_min_list,
        lgx_max_list,
        redshift,
        'BestFit_z',
        mean_molecular_weight=mu_minihalo,
        PoissonSamplingFlag=include_shmf_scatter,
    )
    fg_corr = f_gas / (Omega_b / Omega_m)
    heating_global_erg_s = 1.0e7 * heating_data['Heating_singlehost'] * fg_corr
    heating_scatter_global_erg_s = {}
    if include_shmf_scatter:
        for key in [
            'Heating_singlehost_mean',
            'Heating_singlehost_median',
            'Heating_singlehost_p16',
            'Heating_singlehost_p84',
            'Heating_singlehost_p2p5',
            'Heating_singlehost_p97p5',
            'Heating_singlehost_p0p15',
            'Heating_singlehost_p99p85',
        ]:
            heating_scatter_global_erg_s[key] = 1.0e7 * heating_data[key] * fg_corr

    cooling_param_sets = [
        {
            'gas_metallicity': 0.0,
            'f_H2': 1.0e-6,
            'color': 'cyan',
            'label': r'$f_{\rm H_2}=10^{-6}$',
        },
        {
            'gas_metallicity': 0.0,
            'f_H2': 1.0e-5,
            'color': 'deepskyblue',
            'label': r'$f_{\rm H_2}=10^{-5}$',
        },
        {
            'gas_metallicity': 0.0,
            'f_H2': 1.0e-4,
            'color': 'royalblue',
            'label': r'$f_{\rm H_2}=10^{-4}$',
        },
    ]

    radius_specs = [
        ('rs', r'$<r_s$', None),
        ('0p5Rvir', r'$<0.5R_{\rm vir}$', 0.5),
        ('Rvir', r'$<R_{\rm vir}$', 1.0),
    ]
    n_radii = len(radius_specs)
    n_cooling = len(cooling_param_sets)
    n_mass = len(lgM_list)

    concentration_values = np.zeros(n_mass)
    radii_Rvir = np.zeros((n_radii, n_mass))
    heating_fraction = np.zeros((n_radii, n_mass))
    cooling_correction = np.zeros((n_radii, n_mass))
    cooling_uniform_erg_s = np.zeros((n_mass, n_cooling))
    cooling_uniform_converged_erg_s = np.zeros((n_mass, n_cooling))
    cooling_uniform_fixed_species_erg_s = np.zeros((n_mass, n_cooling))

    for i, lgM in enumerate(lgM_list):
        Mvir = 10.0 ** lgM
        concentration_value = get_concentration(Mvir / h_Hubble, redshift, concentration_model)
        concentration_values[i] = concentration_value
        radii_here = np.array([
            1.0 / concentration_value if radius_value is None else radius_value
            for _, _, radius_value in radius_specs
        ])
        radii_Rvir[:, i] = radii_here
        heating_fraction[:, i] = get_cumulative_heating_fraction_modelA(
            radii_here,
            concentration_value,
            alpha,
        )
        cooling_correction[:, i] = get_profile_corr_for_cooling_within_radius(
            radii_here,
            concentration_value,
            alpha,
        )

        (
            cooling_uniform_erg_s[i, :],
            cooling_uniform_converged_erg_s[i, :],
            cooling_uniform_fixed_species_erg_s[i, :],
        ) = get_EqCooling_envelope_for_single_host_minihalo(
            Mvir,
            redshift,
            cooling_param_sets,
            mean_molecular_weight=mu_minihalo,
            return_components=True,
        )

    heating_within_erg_s = heating_global_erg_s[None, :] * heating_fraction
    heating_scatter_within_erg_s = {
        key: values[None, :] * heating_fraction
        for key, values in heating_scatter_global_erg_s.items()
    }
    cooling_within_erg_s = (
        cooling_uniform_erg_s[None, :, :]
        * fg_corr**2
        * cooling_correction[:, :, None]
    )
    ratio_within = heating_within_erg_s[:, :, None] / cooling_within_erg_s

    alpha_tag = f"alpha{alpha:.1f}".replace('.', 'p').replace('-', 'm')
    fg_tag = 'fgcosmic' if np.isclose(f_gas, Omega_b / Omega_m) else f"fg{f_gas:.3f}".replace('.', 'p')
    z_tag = f"z{redshift:.2f}".replace('.', 'p')

    fig, axes = plt.subplots(1, n_radii, figsize=(17, 5.2), sharey=True, facecolor='white')
    for ir, (_, radius_label, _) in enumerate(radius_specs):
        ax = axes[ir]
        if include_shmf_scatter:
            ax.fill_between(
                lgM_list,
                heating_scatter_within_erg_s['Heating_singlehost_p0p15'][ir],
                heating_scatter_within_erg_s['Heating_singlehost_p99p85'][ir],
                color='orange',
                alpha=0.1,
                linewidth=0,
                label='0.15/99.85 %',
                zorder=1,
            )
            ax.fill_between(
                lgM_list,
                heating_scatter_within_erg_s['Heating_singlehost_p2p5'][ir],
                heating_scatter_within_erg_s['Heating_singlehost_p97p5'][ir],
                color='orange',
                alpha=0.2,
                linewidth=0,
                label='2.5/97.5 %',
                zorder=2,
            )
            ax.fill_between(
                lgM_list,
                heating_scatter_within_erg_s['Heating_singlehost_p16'][ir],
                heating_scatter_within_erg_s['Heating_singlehost_p84'][ir],
                color='orange',
                alpha=0.4,
                linewidth=0,
                label='16/84 %',
                zorder=3,
            )
            ax.plot(
                lgM_list,
                heating_scatter_within_erg_s['Heating_singlehost_median'][ir],
                color='red',
                linewidth=1.5,
                linestyle=':',
                label='Median',
                zorder=4,
            )
        ax.plot(
            lgM_list,
            heating_within_erg_s[ir],
            color='red',
            linewidth=1.5,
            label='Mean',
            zorder=5,
        )
        for ic, params in enumerate(cooling_param_sets):
            ax.plot(
                lgM_list,
                cooling_within_erg_s[ir, :, ic],
                color=params['color'],
                linewidth=1.8,
                linestyle='-',
                label=params['label'],
            )
        ax.set_title(radius_label, fontsize=13)
        ax.set_yscale('log')
        ax.set_xlabel(r'log$_{10}$ M [M$_\odot$/h]', fontsize=12)
        ax.tick_params(axis='both', direction='in')
        ax.grid(alpha=0.25)
        if ir == 0:
            ax.set_ylabel(r'Heating and Cooling [erg/s]', fontsize=12)
        ax.legend(fontsize=8)
    fig.suptitle(
        rf'Model A cumulative heating/cooling, z={redshift:.1f}, '
        rf'$\alpha={alpha:.1f}$, {concentration_model}, $f_g={f_gas:.3f}$',
        fontsize=13,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    scatter_tag = '_SHMFscatter' if include_shmf_scatter else ''
    hc_filename = os.path.join(
        output_dir,
        f'modelA_cumulative_HC_minihalo_{z_tag}_{alpha_tag}_{fg_tag}_{concentration_model}{scatter_tag}.png',
    )
    plt.savefig(hc_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved cumulative H/C plot: {hc_filename}")

    summary_path = os.path.join(
        output_dir,
        f'modelA_cumulative_HC_minihalo_{z_tag}_{alpha_tag}_{fg_tag}_{concentration_model}{scatter_tag}.npz',
    )
    save_data = dict(
        lgM_list=lgM_list,
        concentration=concentration_values,
        radii_Rvir=radii_Rvir,
        heating_global_erg_s=heating_global_erg_s,
        heating_fraction=heating_fraction,
        cooling_correction=cooling_correction,
        cooling_uniform_erg_s=cooling_uniform_erg_s,
        cooling_uniform_converged_erg_s=cooling_uniform_converged_erg_s,
        cooling_uniform_fixed_species_erg_s=cooling_uniform_fixed_species_erg_s,
        heating_within_erg_s=heating_within_erg_s,
        cooling_within_erg_s=cooling_within_erg_s,
        ratio_within=ratio_within,
        radius_names=np.array([name for name, _, _ in radius_specs]),
        f_H2_values=np.array([params['f_H2'] for params in cooling_param_sets]),
        alpha=alpha,
        redshift=redshift,
        f_gas=f_gas,
        include_shmf_scatter=include_shmf_scatter,
    )
    if include_shmf_scatter:
        for key, values in heating_scatter_global_erg_s.items():
            save_data[f'{key}_erg_s'] = values
        for key, values in heating_scatter_within_erg_s.items():
            save_data[f'{key}_within_erg_s'] = values
    np.savez(summary_path, **save_data)
    print(f"Saved cumulative H/C data: {summary_path}")

    return {
        'lgM_list': lgM_list,
        'concentration': concentration_values,
        'radii_Rvir': radii_Rvir,
        'heating_within_erg_s': heating_within_erg_s,
        'cooling_within_erg_s': cooling_within_erg_s,
        'ratio_within': ratio_within,
        'heating_scatter_within_erg_s': heating_scatter_within_erg_s,
        'hc_filename': hc_filename,
        'summary_path': summary_path,
    }


def plot_modelA_cumulative_heating_cooling_massivehalo(
    redshift,
    min_lgM=None,
    max_lgM=None,
    n_lgM=50,
    alpha=0.0,
    f_gas=0.05,
    concentration_model='ludlow16',
    output_dir='/home/zwu/21cm_project/unified_model/Analytic_HC_results_within_radius',
    include_shmf_scatter=True,
):
    """
    Plot low-z massive-halo Model A cumulative heating/cooling within selected radii.

    This mirrors the old global massive-halo comparison: atomic cooling with
    equilibrium species setup, f_g=0.05, Dekel metallicity plus a broad
    metallicity band, and SHMF Poisson scatter on the DF heating amplitude.
    """
    default_mass_ranges = {
        0: (10, 15),
        1: (10, 15),
        2: (10, 15),
        3: (10, 14),
        4: (10, 13),
        5: (9, 13),
        6: (9, 13),
        7: (9, 12),
        8: (9, 12),
    }
    redshift_key = int(redshift)
    if min_lgM is None or max_lgM is None:
        if redshift_key not in default_mass_ranges:
            raise ValueError("Provide min_lgM/max_lgM for redshifts outside the default grid.")
        default_min_lgM, default_max_lgM = default_mass_ranges[redshift_key]
        min_lgM = default_min_lgM if min_lgM is None else min_lgM
        max_lgM = default_max_lgM if max_lgM is None else max_lgM

    print(
        f"plotting low-z cumulative Model A heating/cooling: "
        f"z={redshift:.2f}, alpha={alpha:.2f}, f_gas={f_gas:.3f}, "
        f"concentration={concentration_model}, SHMF scatter={include_shmf_scatter}"
    )
    os.makedirs(output_dir, exist_ok=True)

    lgM_list = np.linspace(min_lgM, max_lgM, n_lgM)
    lgx_min_list = np.full_like(lgM_list, np.log10(1.0e-3), dtype=float)
    lgx_max_list = np.full_like(lgM_list, np.log10(1.0), dtype=float)

    heating_data_raw = get_heating_per_lgM(
        lgM_list,
        lgx_min_list,
        lgx_max_list,
        redshift,
        'BestFit_z',
        mean_molecular_weight=mu,
        PoissonSamplingFlag=include_shmf_scatter,
    )
    heating_data = create_heating_data_dict_with_fgas(heating_data_raw, f_gas=f_gas)
    heating_global_erg_s = 1.0e7 * heating_data['Heating_singlehost']
    heating_scatter_global_erg_s = {}
    if include_shmf_scatter:
        for key in [
            'Heating_singlehost_mean',
            'Heating_singlehost_median',
            'Heating_singlehost_p16',
            'Heating_singlehost_p84',
            'Heating_singlehost_p2p5',
            'Heating_singlehost_p97p5',
            'Heating_singlehost_p0p15',
            'Heating_singlehost_p99p85',
        ]:
            heating_scatter_global_erg_s[key] = 1.0e7 * heating_data[key]

    Z_Dekel = 0.3 * 10.0**(-0.17 * redshift)
    cooling_param_sets = [
        {'gas_metallicity': 1.0e-3, 'f_H2': 0.0, 'label': r'$Z=10^{-3}Z_\odot$'},
        {'gas_metallicity': 1.0, 'f_H2': 0.0, 'label': r'$Z=Z_\odot$'},
        {'gas_metallicity': Z_Dekel, 'f_H2': 0.0, 'label': rf'$Z_{{\rm Dekel}}={Z_Dekel:.2f}Z_\odot$'},
    ]

    radius_specs = [
        ('rs', r'$<r_s$', None),
        ('0p5Rvir', r'$<0.5R_{\rm vir}$', 0.5),
        ('Rvir', r'$<R_{\rm vir}$', 1.0),
    ]
    n_radii = len(radius_specs)
    n_cooling = len(cooling_param_sets)
    n_mass = len(lgM_list)

    concentration_values = np.zeros(n_mass)
    radii_Rvir = np.zeros((n_radii, n_mass))
    heating_fraction = np.zeros((n_radii, n_mass))
    cooling_correction = np.zeros((n_radii, n_mass))
    cooling_uniform_erg_s = np.zeros((n_mass, n_cooling))

    for i, lgM in enumerate(lgM_list):
        Mvir = 10.0 ** lgM
        concentration_value = get_concentration(Mvir / h_Hubble, redshift, concentration_model)
        concentration_values[i] = concentration_value
        radii_here = np.array([
            1.0 / concentration_value if radius_value is None else radius_value
            for _, _, radius_value in radius_specs
        ])
        radii_Rvir[:, i] = radii_here
        heating_fraction[:, i] = get_cumulative_heating_fraction_modelA(
            radii_here,
            concentration_value,
            alpha,
        )
        cooling_correction[:, i] = get_profile_corr_for_cooling_within_radius(
            radii_here,
            concentration_value,
            alpha,
        )
        cooling_uniform_erg_s[i, :] = np.asarray(
            get_EqCooling_for_single_host(
                Mvir,
                redshift,
                cooling_param_sets,
                mean_molecular_weight=mu,
                converge_when_setup=True,
            ),
            dtype=float,
        )

    fg_corr = f_gas / (Omega_b / Omega_m)
    heating_within_erg_s = heating_global_erg_s[None, :] * heating_fraction
    heating_scatter_within_erg_s = {
        key: values[None, :] * heating_fraction
        for key, values in heating_scatter_global_erg_s.items()
    }
    cooling_within_erg_s = (
        cooling_uniform_erg_s[None, :, :]
        * fg_corr**2
        * cooling_correction[:, :, None]
    )
    ratio_within = heating_within_erg_s[:, :, None] / cooling_within_erg_s

    alpha_tag = f"alpha{alpha:.1f}".replace('.', 'p').replace('-', 'm')
    fg_tag = f"fg{f_gas:.3f}".replace('.', 'p')
    z_tag = f"z{redshift:.2f}".replace('.', 'p')
    scatter_tag = '_SHMFscatter' if include_shmf_scatter else ''

    fig, axes = plt.subplots(1, n_radii, figsize=(17, 5.2), sharey=True, facecolor='white')
    for ir, (_, radius_label, _) in enumerate(radius_specs):
        ax = axes[ir]
        if include_shmf_scatter:
            ax.fill_between(
                lgM_list,
                heating_scatter_within_erg_s['Heating_singlehost_p0p15'][ir],
                heating_scatter_within_erg_s['Heating_singlehost_p99p85'][ir],
                color='orange',
                alpha=0.1,
                linewidth=0,
                label='0.15/99.85 %',
                zorder=1,
            )
            ax.fill_between(
                lgM_list,
                heating_scatter_within_erg_s['Heating_singlehost_p2p5'][ir],
                heating_scatter_within_erg_s['Heating_singlehost_p97p5'][ir],
                color='orange',
                alpha=0.2,
                linewidth=0,
                label='2.5/97.5 %',
                zorder=2,
            )
            ax.fill_between(
                lgM_list,
                heating_scatter_within_erg_s['Heating_singlehost_p16'][ir],
                heating_scatter_within_erg_s['Heating_singlehost_p84'][ir],
                color='orange',
                alpha=0.4,
                linewidth=0,
                label='16/84 %',
                zorder=3,
            )
            ax.plot(
                lgM_list,
                heating_scatter_within_erg_s['Heating_singlehost_median'][ir],
                color='red',
                linewidth=1.5,
                linestyle=':',
                label='Median',
                zorder=4,
            )
        ax.plot(
            lgM_list,
            heating_within_erg_s[ir],
            color='red',
            linewidth=1.5,
            label='Mean',
            zorder=5,
        )

        cooling_lower = np.minimum(cooling_within_erg_s[ir, :, 0], cooling_within_erg_s[ir, :, 1])
        cooling_upper = np.maximum(cooling_within_erg_s[ir, :, 0], cooling_within_erg_s[ir, :, 1])
        ax.fill_between(
            lgM_list,
            cooling_lower,
            cooling_upper,
            color='blue',
            alpha=0.3,
            linewidth=0,
            label=r'$Z=10^{-3}-1Z_\odot$',
            zorder=1,
        )
        ax.plot(
            lgM_list,
            cooling_within_erg_s[ir, :, 2],
            color='blue',
            linewidth=1.8,
            label=rf'$Z_{{\rm Dekel}}={Z_Dekel:.2f}Z_\odot$',
            zorder=4,
        )
        ax.set_title(radius_label, fontsize=13)
        ax.set_yscale('log')
        ax.set_xlabel(r'log$_{10}$ M [M$_\odot$/h]', fontsize=12)
        ax.tick_params(axis='both', direction='in')
        ax.grid(alpha=0.25)
        if ir == 0:
            ax.set_ylabel(r'Heating and Cooling [erg/s]', fontsize=12)
        ax.legend(fontsize=8)

    fig.suptitle(
        rf'Model A cumulative heating/cooling, z={redshift:.1f}, core profile, '
        rf'{concentration_model}, $f_g={f_gas:.2f}$',
        fontsize=13,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    hc_filename = os.path.join(
        output_dir,
        f'modelA_cumulative_HC_massivehalo_{z_tag}_{alpha_tag}_{fg_tag}_{concentration_model}{scatter_tag}.png',
    )
    plt.savefig(hc_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved low-z cumulative H/C plot: {hc_filename}")

    summary_path = os.path.join(
        output_dir,
        f'modelA_cumulative_HC_massivehalo_{z_tag}_{alpha_tag}_{fg_tag}_{concentration_model}{scatter_tag}.npz',
    )
    save_data = dict(
        lgM_list=lgM_list,
        concentration=concentration_values,
        radii_Rvir=radii_Rvir,
        heating_global_erg_s=heating_global_erg_s,
        heating_fraction=heating_fraction,
        cooling_correction=cooling_correction,
        cooling_uniform_erg_s=cooling_uniform_erg_s,
        heating_within_erg_s=heating_within_erg_s,
        cooling_within_erg_s=cooling_within_erg_s,
        ratio_within=ratio_within,
        radius_names=np.array([name for name, _, _ in radius_specs]),
        metallicity_values=np.array([params['gas_metallicity'] for params in cooling_param_sets]),
        alpha=alpha,
        redshift=redshift,
        f_gas=f_gas,
        Z_Dekel=Z_Dekel,
        include_shmf_scatter=include_shmf_scatter,
    )
    if include_shmf_scatter:
        for key, values in heating_scatter_global_erg_s.items():
            save_data[f'{key}_erg_s'] = values
        for key, values in heating_scatter_within_erg_s.items():
            save_data[f'{key}_within_erg_s'] = values
    np.savez(summary_path, **save_data)
    print(f"Saved low-z cumulative H/C data: {summary_path}")

    return {
        'lgM_list': lgM_list,
        'concentration': concentration_values,
        'radii_Rvir': radii_Rvir,
        'heating_within_erg_s': heating_within_erg_s,
        'cooling_within_erg_s': cooling_within_erg_s,
        'ratio_within': ratio_within,
        'heating_scatter_within_erg_s': heating_scatter_within_erg_s,
        'hc_filename': hc_filename,
        'summary_path': summary_path,
    }


def plot_modelA_cumulative_heating_cooling_massivehalo_redshift_set(
    redshifts=(0, 2, 6),
    output_dir='/home/zwu/21cm_project/unified_model/Analytic_HC_results_within_radius',
):
    results = {}
    for redshift in redshifts:
        results[redshift] = plot_modelA_cumulative_heating_cooling_massivehalo(
            redshift=redshift,
            output_dir=output_dir,
            include_shmf_scatter=True,
        )
    return results

"""
def plot_modelA_differential_cooling_heating_profile(
    concentration_model='ludlow16',
    output_dir='/home/zwu/21cm_project/unified_model/debug',
):

    # Plot differential normalized cooling/heating profile shapes for Model A.

    # The plotted quantities are derivatives of the cumulative profile fractions:
    # d[C(<r)/C(<Rvir)]/d(r/Rvir) and d[H_A(<r)/H_A(<Rvir)]/d(r/Rvir).
    # They isolate the radial profile correction before applying global
    # heating/cooling amplitudes or gas-fraction corrections.

    test_cases = [
        {
            'label': 'case 1 (z=0 halo)',
            'Mvir': 1.0e12,
            'redshift': 0.0,
        },
        {
            'label': 'case 2 (z=12 minihalo)',
            'Mvir': 1.0e6,
            'redshift': 12.0,
        },
    ]
    os.makedirs(output_dir, exist_ok=True)

    r_grid = np.logspace(-4, 0, 600)
    alpha_values = [0.0, 1.0]
    colors = {0.0: 'tab:blue', 1.0: 'tab:orange'}

    for case in test_cases:
        Mvir = case['Mvir']
        redshift = case['redshift']
        concentration_value = get_concentration(Mvir / h_Hubble, redshift, concentration_model)
        rs_over_rvir = 1.0 / concentration_value

        print(f"\nModel A differential profile: {case['label']}")
        print(f"Mvir={Mvir:.2e} Msun/h, z={redshift:.1f}")
        print(f"concentration = {concentration_value:.6f}, r_s/Rvir = {rs_over_rvir:.6f}")

        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        for alpha in alpha_values:
            cooling_dfdx = get_differential_cooling_fraction(
                r_grid,
                concentration_value,
                alpha,
            )
            heating_dfdx = get_differential_heating_fraction_modelA(
                r_grid,
                concentration_value,
                alpha,
            )
            cooling_integral = np.trapezoid(cooling_dfdx, r_grid)
            heating_integral = np.trapezoid(heating_dfdx, r_grid)
            print(
                f"alpha={alpha:.1f}: integral dC/dx={cooling_integral:.5f}, "
                f"integral dH_A/dx={heating_integral:.5f}"
            )

            ax.plot(
                r_grid,
                cooling_dfdx,
                color=colors[alpha],
                linewidth=2.0,
                linestyle='-',
                label=rf'Cooling, $\alpha={alpha:.1f}$',
            )
            ax.plot(
                r_grid,
                heating_dfdx,
                color=colors[alpha],
                linewidth=2.0,
                linestyle='--',
                label=rf'$H_{{\rm gas}}$, $\alpha={alpha:.1f}$',
            )

        ax.axvline(
            rs_over_rvir,
            color='black',
            linestyle='-.',
            linewidth=1.2,
            label=r'$r_s/R_{\rm vir}$',
        )
        ax.axvline(
            0.5,
            color='grey',
            linestyle=':',
            linewidth=1.2,
            label=r'$0.5\,R_{\rm vir}$',
        )
        ax.set_xscale('log')
        ax.set_xlim(1.0e-4, 1.0)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(r'$r/R_{\rm vir}$', fontsize=14)
        ax.set_ylabel(r'$dF/d(r/R_{\rm vir})$', fontsize=14)
        ax.text(
            0.97,
            0.06,
            rf'$z={redshift:.0f}$, $M_{{\rm host}}=10^{{{np.log10(Mvir):.0f}}}\,M_\odot/h$',
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.7', alpha=0.9),
        )
        ax.tick_params(axis='both', direction='in')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=10)
        plt.tight_layout()

        safe_label = case['label'].replace(' ', '_').replace('(', '').replace(')', '')
        filename = os.path.join(output_dir, f'modelA_differential_cooling_heating_profile_{safe_label}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"saved Model A differential profile plot: {filename}")


"""


def _format_subhalo_profile_label(config):
    radial_bias_model = config["radial_bias_model"].lower()
    if radial_bias_model == "jb17":
        return "subhalo profile: NFW*JvBIII"
    if radial_bias_model == "han16":
        return rf'subhalo profile: NFW*Han16, $\gamma={config.get("han16_gamma", 1.33):.2f}$'
    return "subhalo profile: NFW"


def _get_alpha_linestyle(alpha):
    if np.isclose(alpha, 0.0):
        return '-'
    if np.isclose(alpha, 1.0):
        return '--'
    return ':'


def _get_subhalo_profile_color(config):
    radial_bias_model = config["radial_bias_model"].lower()
    if radial_bias_model == "jb17":
        return 'tab:green'
    if radial_bias_model == "han16":
        return 'tab:purple'
    return '#666666'


def _get_xmax_marker(x_max):
    return 's' if np.isclose(x_max, 1.0) else 'D'



def plot_overlay_Hgas_Hsub_cumulative_heating_cooling(
    input_path=None,
    output_path=None,
    radius_names_to_plot=('rs', '0p5Rvir'),
    hsub_configs=None,
    hsub_color='tab:purple',
    cooling_color='blue',
    hgas_color='tab:red',
    shmf_color='orange',
    figsize=None,
):
    """
    Plot cumulative heating/cooling within selected radii and overlay H_sub.

    The input file is the npz summary produced by
    plot_modelA_cumulative_heating_cooling_massivehalo(). It contains the
    global heating/cooling amplitudes, the H_gas profile correction, the
    cooling profile correction, and SHMF scatter bands. This function keeps
    those data unchanged and computes H_sub curves only for the overlay.
    """
    if input_path is None:
        input_path = (
            '/home/zwu/21cm_project/unified_model/Analytic_HC_results_within_radius/'
            'modelA_cumulative_HC_massivehalo_z0p00_alpha0p0_fg0p050_ludlow16_SHMFscatter.npz'
        )
    if output_path is None:
        output_path = (
            '/home/zwu/21cm_project/unified_model/Analytic_HC_results_within_radius/'
            'overlay_Hgas_Hsub_mean_massivehalo_z0p00_alpha0p0_fg0p050_ludlow16_test.png'
        )
    if hsub_configs is None:
        hsub_configs = [
            {'x_max': 1.0, 'radial_bias_model': 'han16', 'han16_gamma': 1.33},
            {'x_max': 2.0, 'radial_bias_model': 'han16', 'han16_gamma': 1.33},
        ]

    data = np.load(input_path, allow_pickle=True)
    lgM_list = data['lgM_list']
    concentration_values = data['concentration']
    radii_Rvir = data['radii_Rvir']
    radius_names = [str(name) for name in data['radius_names']]
    alpha = float(data['alpha'])
    redshift = float(data['redshift'])
    f_gas = float(data['f_gas'])
    Z_Dekel = float(data['Z_Dekel']) if 'Z_Dekel' in data.files else 0.3 * 10.0**(-0.17 * redshift)

    radius_indices = []
    for radius_name in radius_names_to_plot:
        if radius_name not in radius_names:
            raise ValueError(f"Radius name {radius_name!r} not found in {radius_names}.")
        radius_indices.append(radius_names.index(radius_name))

    radius_labels = {
        'rs': r'$<r_s$',
        '0p5Rvir': r'$<0.5R_{\rm vir}$',
        'Rvir': r'$<R_{\rm vir}$',
    }
    n_panels = len(radius_indices)
    if figsize is None:
        figsize = (5.75 * n_panels, 5.2)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=figsize,
        sharey=True,
        facecolor='white',
        squeeze=False,
    )
    axes = axes[0]

    heating_global_erg_s = data['heating_global_erg_s']
    heating_within_erg_s = data['heating_within_erg_s']
    cooling_within_erg_s = data['cooling_within_erg_s']

    hsub_within = {}
    for config_index, config in enumerate(hsub_configs):
        ratio = np.zeros((len(radius_indices), len(lgM_list)))
        for panel_index, radius_index in enumerate(radius_indices):
            for mass_index, concentration_value in enumerate(concentration_values):
                ratio[panel_index, mass_index] = get_cumulative_heating_ratio_modelC(
                    radii_Rvir[radius_index, mass_index],
                    concentration_value,
                    alpha,
                    x_max=config['x_max'],
                    radial_bias_model=config.get('radial_bias_model', 'nfw'),
                    han16_gamma=config.get('han16_gamma', 1.33),
                )
        hsub_within[config_index] = heating_global_erg_s[None, :] * ratio

    for panel_index, radius_index in enumerate(radius_indices):
        ax = axes[panel_index]
        radius_name = radius_names[radius_index]

        if 'Heating_singlehost_p0p15_within_erg_s' in data.files:
            scatter_bands = [
                ('Heating_singlehost_p0p15_within_erg_s', 'Heating_singlehost_p99p85_within_erg_s', 0.10),
                ('Heating_singlehost_p2p5_within_erg_s', 'Heating_singlehost_p97p5_within_erg_s', 0.20),
                ('Heating_singlehost_p16_within_erg_s', 'Heating_singlehost_p84_within_erg_s', 0.40),
            ]
            for low_key, high_key, alpha_fill in scatter_bands:
                ax.fill_between(
                    lgM_list,
                    data[low_key][radius_index],
                    data[high_key][radius_index],
                    color=shmf_color,
                    alpha=alpha_fill,
                    linewidth=0,
                    zorder=1,
                )

        cooling_lower = np.min(cooling_within_erg_s[radius_index], axis=1)
        cooling_upper = np.max(cooling_within_erg_s[radius_index], axis=1)
        ax.fill_between(
            lgM_list,
            cooling_lower,
            cooling_upper,
            color=cooling_color,
            alpha=0.30,
            linewidth=0,
            zorder=1,
        )
        if cooling_within_erg_s.shape[2] >= 3:
            ax.plot(
                lgM_list,
                cooling_within_erg_s[radius_index, :, 2],
                color=cooling_color,
                linewidth=1.8,
                zorder=4,
            )

        ax.plot(
            lgM_list,
            heating_within_erg_s[radius_index],
            color=hgas_color,
            linewidth=1.7,
            linestyle='-',
            zorder=6,
        )
        if 'Heating_singlehost_median_within_erg_s' in data.files:
            ax.plot(
                lgM_list,
                data['Heating_singlehost_median_within_erg_s'][radius_index],
                color=hgas_color,
                linewidth=1.7,
                linestyle=':',
                zorder=5,
            )

        for config_index, config in enumerate(hsub_configs):
            ax.plot(
                lgM_list,
                hsub_within[config_index][panel_index],
                color=hsub_color,
                linewidth=1.8,
                linestyle='-',
                marker=_get_xmax_marker(config['x_max']),
                markersize=4.5,
                markerfacecolor='none',
                markevery=5,
                zorder=7,
            )

        ax.set_title(radius_labels.get(radius_name, radius_name), fontsize=13)
        ax.set_yscale('log')
        ax.set_xlabel(r'log$_{10}$ M [M$_\odot$/h]', fontsize=12)
        ax.tick_params(axis='both', direction='in')
        ax.grid(alpha=0.25)
        if panel_index == 0:
            ax.set_ylabel(r'Cumulative heating/cooling within radius [erg/s]', fontsize=12)
            first_hsub = hsub_configs[0]
            profile_name = first_hsub.get('radial_bias_model', 'nfw').lower()
            if profile_name == 'han16':
                hsub_text = rf'H$_{{\rm sub}}$: Han16, $\gamma={first_hsub.get("han16_gamma", 1.33):.2f}$'
            elif profile_name == 'jb17':
                hsub_text = r'H$_{\rm sub}$: JvBIII'
            else:
                hsub_text = r'H$_{\rm sub}$: NFW'
            ax.text(
                0.04,
                0.95,
                rf'$z={redshift:.0f}$, $\alpha={alpha:.0f}$, $f_g={f_gas:.2f}$' + '\n' + hsub_text,
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=10.5,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.7', alpha=0.9),
            )

    hsub_handles = [
        mlines.Line2D(
            [],
            [],
            color=hsub_color,
            linestyle='-',
            marker=_get_xmax_marker(config['x_max']),
            markerfacecolor='none',
            markersize=5,
            label=rf'$H_{{\rm sub}}$, $x_{{\max}}={config["x_max"]:.0f}$',
        )
        for config in hsub_configs
    ]
    legend_handles = [
        mlines.Line2D([], [], color=hgas_color, linewidth=1.8, linestyle='-', label=r'$H_{\rm gas}$ mean'),
        mlines.Line2D([], [], color=hgas_color, linewidth=1.8, linestyle=':', label=r'$H_{\rm gas}$ median'),
        mlines.Line2D([], [], color=shmf_color, linewidth=6, alpha=0.35, label='SHMF scatter'),
    ] + hsub_handles + [
        mlines.Line2D([], [], color=cooling_color, linewidth=6, alpha=0.30, label=r'$Z=10^{-3}-1Z_\odot$'),
        mlines.Line2D([], [], color=cooling_color, linewidth=1.8, linestyle='-', label=rf'$Z_{{\rm Dekel}}={Z_Dekel:.2f}Z_\odot$'),
    ]
    axes[-1].legend(handles=legend_handles, loc='best', fontsize=9.5, frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Hgas/Hsub cumulative overlay plot: {output_path}")
    return output_path



def plot_paper_cooling_heating_within_radius_z0_alpha0_fg0p05(
    output_path=(
        '/home/zwu/21cm_project/unified_model/Profile_results_for_paper/'
        'cooling_heating_within_radius/cooling_heating_within_radius_z0_alpha0_fg0p05.png'
    ),
):
    """Generate the z=0 paper figure for cumulative H_gas/H_sub heating and cooling."""
    return plot_overlay_Hgas_Hsub_cumulative_heating_cooling(output_path=output_path)



def plot_paper_cooling_heating_within_radius_z15_extreme_cases(
    upper_input_path=(
        '/home/zwu/21cm_project/unified_model/Analytic_HC_results_within_radius/'
        'modelA_cumulative_HC_minihalo_z15p00_alpha1p0_fgcosmic_ludlow16_SHMFscatter.npz'
    ),
    lower_input_path=(
        '/home/zwu/21cm_project/unified_model/Analytic_HC_results_within_radius/'
        'modelA_cumulative_HC_minihalo_z15p00_alpha0p0_fg0p050_ludlow16_SHMFscatter.npz'
    ),
    output_path=(
        '/home/zwu/21cm_project/unified_model/Profile_results_for_paper/'
        'cooling_heating_within_radius/cooling_heating_within_radius_z15_extreme_cases.png'
    ),
    radius_names_to_plot=('rs', '0p5Rvir'),
    hsub_configs=None,
):
    """Generate the z=15 paper figure for two extreme gas-profile/gas-fraction cases."""
    if hsub_configs is None:
        hsub_configs = [
            {'x_max': 1.0, 'radial_bias_model': 'nfw'},
            {'x_max': 2.0, 'radial_bias_model': 'nfw'},
        ]

    case_specs = [
        {
            'input_path': upper_input_path,
            'row_label': r'$z=15$, $\alpha=1$, $f_g=f_b$' + '\n' + r'$H_{\rm sub}$: NFW',
        },
        {
            'input_path': lower_input_path,
            'row_label': r'$z=15$, $\alpha=0$, $f_g=0.05$' + '\n' + r'$H_{\rm sub}$: NFW',
        },
    ]
    radius_labels = {
        'rs': r'$<r_s$',
        '0p5Rvir': r'$<0.5R_{\rm vir}$',
        'Rvir': r'$<R_{\rm vir}$',
    }
    cooling_colors = ['cyan', 'deepskyblue', 'royalblue']
    hgas_color = 'tab:red'
    shmf_color = 'orange'
    hsub_color = _get_subhalo_profile_color({'radial_bias_model': 'nfw'})

    fig, axes = plt.subplots(
        len(case_specs),
        len(radius_names_to_plot),
        figsize=(11.5, 8.4),
        sharex=True,
        sharey=True,
        facecolor='white',
        squeeze=False,
    )

    for row_index, case in enumerate(case_specs):
        data = np.load(case['input_path'], allow_pickle=True)
        lgM_list = data['lgM_list']
        concentration_values = data['concentration']
        radii_Rvir = data['radii_Rvir']
        radius_names = [str(name) for name in data['radius_names']]
        alpha = float(data['alpha'])
        heating_global_erg_s = data['heating_global_erg_s']
        heating_within_erg_s = data['heating_within_erg_s']
        cooling_within_erg_s = data['cooling_within_erg_s']
        f_H2_values = data['f_H2_values']

        radius_indices = []
        for radius_name in radius_names_to_plot:
            if radius_name not in radius_names:
                raise ValueError(f"Radius name {radius_name!r} not found in {radius_names}.")
            radius_indices.append(radius_names.index(radius_name))

        hsub_within = {}
        for config_index, config in enumerate(hsub_configs):
            ratio = np.zeros((len(radius_indices), len(lgM_list)))
            for panel_index, radius_index in enumerate(radius_indices):
                for mass_index, concentration_value in enumerate(concentration_values):
                    ratio[panel_index, mass_index] = get_cumulative_heating_ratio_modelC(
                        radii_Rvir[radius_index, mass_index],
                        concentration_value,
                        alpha,
                        x_max=config['x_max'],
                        radial_bias_model=config.get('radial_bias_model', 'nfw'),
                        han16_gamma=config.get('han16_gamma', 1.33),
                    )
            hsub_within[config_index] = heating_global_erg_s[None, :] * ratio

        for col_index, radius_index in enumerate(radius_indices):
            ax = axes[row_index, col_index]
            radius_name = radius_names[radius_index]

            scatter_bands = [
                ('Heating_singlehost_p0p15_within_erg_s', 'Heating_singlehost_p99p85_within_erg_s', 0.10),
                ('Heating_singlehost_p2p5_within_erg_s', 'Heating_singlehost_p97p5_within_erg_s', 0.20),
                ('Heating_singlehost_p16_within_erg_s', 'Heating_singlehost_p84_within_erg_s', 0.40),
            ]
            for low_key, high_key, alpha_fill in scatter_bands:
                if low_key in data.files and high_key in data.files:
                    ax.fill_between(
                        lgM_list,
                        data[low_key][radius_index],
                        data[high_key][radius_index],
                        color=shmf_color,
                        alpha=alpha_fill,
                        linewidth=0,
                        zorder=1,
                    )

            ax.plot(
                lgM_list,
                heating_within_erg_s[radius_index],
                color=hgas_color,
                linewidth=1.7,
                linestyle='-',
                zorder=6,
            )
            if 'Heating_singlehost_median_within_erg_s' in data.files:
                ax.plot(
                    lgM_list,
                    data['Heating_singlehost_median_within_erg_s'][radius_index],
                    color=hgas_color,
                    linewidth=1.7,
                    linestyle=':',
                    zorder=5,
                )

            for config_index, config in enumerate(hsub_configs):
                ax.plot(
                    lgM_list,
                    hsub_within[config_index][col_index],
                    color=hsub_color,
                    linewidth=1.8,
                    linestyle='-',
                    marker=_get_xmax_marker(config['x_max']),
                    markersize=4.5,
                    markerfacecolor='none',
                    markevery=5,
                    zorder=7,
                )

            for cooling_index, f_H2 in enumerate(f_H2_values):
                ax.plot(
                    lgM_list,
                    cooling_within_erg_s[radius_index, :, cooling_index],
                    color=cooling_colors[cooling_index % len(cooling_colors)],
                    linewidth=1.8,
                    linestyle='-',
                    zorder=4,
                )

            if row_index == 0:
                ax.set_title(radius_labels.get(radius_name, radius_name), fontsize=13)
            if col_index == 0:
                ax.text(
                    0.04,
                    0.95,
                    case['row_label'],
                    transform=ax.transAxes,
                    ha='left',
                    va='top',
                    fontsize=10.5,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.7', alpha=0.9),
                )
            if row_index == len(case_specs) - 1:
                ax.set_xlabel(r'log$_{10}$ M [M$_\odot$/h]', fontsize=12)

            ax.set_yscale('log')
            ax.tick_params(axis='both', direction='in')
            ax.grid(alpha=0.25)

    hsub_handles = [
        mlines.Line2D(
            [],
            [],
            color=hsub_color,
            linestyle='-',
            marker=_get_xmax_marker(config['x_max']),
            markerfacecolor='none',
            markersize=5,
            label=rf'$H_{{\rm sub}}$, $x_{{\max}}={config["x_max"]:.0f}$',
        )
        for config in hsub_configs
    ]
    cooling_handles = [
        mlines.Line2D(
            [],
            [],
            color=cooling_colors[i % len(cooling_colors)],
            linewidth=1.8,
            linestyle='-',
            label=rf'$f_{{\rm H_2}}=10^{{{int(np.log10(f_H2))}}}$',
        )
        for i, f_H2 in enumerate(case_specs[0].get('f_H2_values', []))
    ]
    if not cooling_handles:
        first_data = np.load(case_specs[0]['input_path'], allow_pickle=True)
        cooling_handles = [
            mlines.Line2D(
                [],
                [],
                color=cooling_colors[i % len(cooling_colors)],
                linewidth=1.8,
                linestyle='-',
                label=rf'$f_{{\rm H_2}}=10^{{{int(np.log10(f_H2))}}}$',
            )
            for i, f_H2 in enumerate(first_data['f_H2_values'])
        ]

    legend_handles = [
        mlines.Line2D([], [], color=hgas_color, linewidth=1.8, linestyle='-', label=r'$H_{\rm gas}$ mean'),
        mlines.Line2D([], [], color=hgas_color, linewidth=1.8, linestyle=':', label=r'$H_{\rm gas}$ median'),
        mlines.Line2D([], [], color=shmf_color, linewidth=6, alpha=0.35, label='SHMF scatter'),
    ] + hsub_handles + cooling_handles
    axes[0, -1].legend(
        handles=legend_handles,
        loc='upper left',
        fontsize=9.5,
        frameon=True,
    )
    fig.supylabel(r'Cumulative heating/cooling within radius [erg/s]', fontsize=12)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.95, wspace=0.08, hspace=0.13)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved z=15 cumulative H/C paper plot: {output_path}")
    return output_path


def test_cooling_heating_profile():
    """
    Lightweight sanity check for cumulative cooling and heating profile corrections.
    """
    test_cases = [
        {
            "label": "case 1 (z=0 halo)",
            "Mvir": 1.0e12,  # Msun/h
            "redshift": 0.0,
            "param_sets": [{"gas_metallicity": 0.0, "f_H2": 0.0}],
            "profile_configs": [
                {"alpha": 0.0, "x_max": 1.0, "radial_bias_model": "han16", "han16_gamma": 1.33},
                {"alpha": 1.0, "x_max": 1.0, "radial_bias_model": "han16", "han16_gamma": 1.33},
                {"alpha": 0.0, "x_max": 2.0, "radial_bias_model": "han16", "han16_gamma": 1.33},
                {"alpha": 1.0, "x_max": 2.0, "radial_bias_model": "han16", "han16_gamma": 1.33},
            ],
        },
        {
            "label": "case 2 (z=12 minihalo)",
            "Mvir": 1.0e6,  # Msun/h
            "redshift": 12.0,
            "param_sets": [{"gas_metallicity": 0.0, "f_H2": 1.0e-5}],
            "profile_configs": [
                {"alpha": 0.0, "x_max": 1.0, "radial_bias_model": "nfw"},
                {"alpha": 1.0, "x_max": 1.0, "radial_bias_model": "nfw"},
                {"alpha": 0.0, "x_max": 2.0, "radial_bias_model": "nfw"},
                {"alpha": 1.0, "x_max": 2.0, "radial_bias_model": "nfw"},
            ],
            "broken_y_ranges": [(0.0, 4.4), (6.5, 7.6)],
        },
        {
            "label": "case 3 (z=15 minihalo)",
            "Mvir": 1.0e6,  # Msun/h
            "redshift": 15.0,
            "param_sets": [{"gas_metallicity": 0.0, "f_H2": 1.0e-5}],
            "profile_configs": [
                {"alpha": 0.0, "x_max": 1.0, "radial_bias_model": "nfw"},
                {"alpha": 1.0, "x_max": 1.0, "radial_bias_model": "nfw"},
                {"alpha": 0.0, "x_max": 2.0, "radial_bias_model": "nfw"},
                {"alpha": 1.0, "x_max": 2.0, "radial_bias_model": "nfw"},
            ],
            "broken_y_ranges": [(0.0, 4.0), (5.5, 7.1)],
        },
    ]
    concentration_model = 'ludlow16'
    output_dir = '/home/zwu/21cm_project/unified_model/Profile_results_for_paper/cooling_heating_profile_corr'
    os.makedirs(output_dir, exist_ok=True)

    for case in test_cases:
        Mvir = case["Mvir"]
        redshift = case["redshift"]
        profile_configs = case["profile_configs"]
        alpha_values = sorted({config["alpha"] for config in profile_configs})
        concentration_value = get_concentration(Mvir / h_Hubble, redshift, concentration_model)
        rs_over_rvir = 1.0 / concentration_value
        radii_Rvir = [rs_over_rvir, 0.5, 1.0]
        r_grid = np.logspace(-3, 0, 300)
        mean_molecular_weight = mu_minihalo if redshift >= 10 else mu

        print(f"\nCooling-profile sanity check: {case['label']}")
        print(f"Mvir={Mvir:.2e} Msun/h, z={redshift:.1f}")
        print(f"concentration = {concentration_value:.6f}, r_s/Rvir = {rs_over_rvir:.6f}")
        print(f"radii_Rvir = {radii_Rvir}")

        for alpha in alpha_values:
            cooling_fraction = get_cumulative_cooling_fraction(
                radii_Rvir, concentration_value, alpha
            )
            heating_fraction = get_cumulative_heating_fraction_modelA(
                radii_Rvir, concentration_value, alpha
            )
            print(f"alpha = {alpha:.1f}, cooling fraction = {cooling_fraction}")
            print(f"alpha = {alpha:.1f}, heating fraction H_gas = {heating_fraction}")

        for config in profile_configs:
            alpha = config["alpha"]
            heating_ratio_C = get_cumulative_heating_ratio_modelC(
                radii_Rvir,
                concentration_value,
                alpha,
                x_max=config["x_max"],
                radial_bias_model=config["radial_bias_model"],
                han16_gamma=config.get("han16_gamma", 1.33),
            )
            print(
                f"alpha = {alpha:.1f}, heating ratio H_sub/H_global, "
                f"{_format_subhalo_profile_label(config)}, "
                f"x_max={config['x_max']:.1f} = {heating_ratio_C}"
            )

        radii_sorted, cooling_fraction_sorted, cumulative_cooling = (
            get_cumulative_cooling_and_heating_withinradius_singlehost(
                Mvir=Mvir,
                redshift=redshift,
                param_sets=case["param_sets"],
                radii_Rvir=radii_Rvir,
                alpha=alpha_values[0],
                concentration_model=concentration_model,
                f_gas=Omega_b / Omega_m,
                mean_molecular_weight=mean_molecular_weight,
                converge_when_setup=True,
            )
        )
        print("wrapper radii_sorted =", radii_sorted)
        print(f"wrapper cooling fraction alpha={alpha_values[0]:.1f} =", cooling_fraction_sorted)
        print("wrapper cumulative cooling shape =", cumulative_cooling.shape)

        broken_y_ranges = case.get("broken_y_ranges")
        if broken_y_ranges is None:
            fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
            plot_axes = [ax]
            legend_ax = ax
            bottom_ax = ax
        else:
            fig, (top_ax, bottom_ax) = plt.subplots(
                2,
                1,
                figsize=(8, 6),
                sharex=True,
                facecolor='white',
                gridspec_kw={'height_ratios': [1.0, 3.0], 'hspace': 0.05},
            )
            ax = bottom_ax
            plot_axes = [top_ax, bottom_ax]
            legend_ax = bottom_ax
            bottom_ax.set_ylim(*broken_y_ranges[0])
            top_ax.set_ylim(*broken_y_ranges[1])
            top_ax.spines['bottom'].set_visible(False)
            bottom_ax.spines['top'].set_visible(False)
            top_ax.tick_params(labeltop=False, bottom=False)
            bottom_ax.xaxis.tick_bottom()
            break_size = 0.010
            break_kwargs = dict(color='black', clip_on=False, linewidth=1.0)
            top_ax.plot((-break_size, +break_size), (-break_size, +break_size), transform=top_ax.transAxes, **break_kwargs)
            top_ax.plot((1 - break_size, 1 + break_size), (-break_size, +break_size), transform=top_ax.transAxes, **break_kwargs)
            bottom_ax.plot((-break_size, +break_size), (1 - break_size, 1 + break_size), transform=bottom_ax.transAxes, **break_kwargs)
            bottom_ax.plot((1 - break_size, 1 + break_size), (1 - break_size, 1 + break_size), transform=bottom_ax.transAxes, **break_kwargs)

        for alpha in alpha_values:
            cooling_fraction_grid = get_cumulative_cooling_fraction(
                r_grid, concentration_value, alpha
            )
            heating_fraction_grid = get_cumulative_heating_fraction_modelA(
                r_grid, concentration_value, alpha
            )
            for plot_ax in plot_axes:
                plot_ax.plot(
                    r_grid,
                    cooling_fraction_grid,
                    color='tab:blue',
                    linewidth=2,
                    linestyle=_get_alpha_linestyle(alpha),
                    label='_nolegend_',
                )
                plot_ax.plot(
                    r_grid,
                    heating_fraction_grid,
                    color='tab:orange',
                    linewidth=2,
                    linestyle=_get_alpha_linestyle(alpha),
                    label='_nolegend_',
                )

        for config in profile_configs:
            alpha = config["alpha"]
            heating_ratio_C_grid = get_cumulative_heating_ratio_modelC(
                r_grid,
                concentration_value,
                alpha,
                x_max=config["x_max"],
                radial_bias_model=config["radial_bias_model"],
                han16_gamma=config.get("han16_gamma", 1.33),
            )
            for plot_ax in plot_axes:
                plot_ax.plot(
                    r_grid,
                    heating_ratio_C_grid,
                    color=_get_subhalo_profile_color(config),
                    linewidth=1.7,
                    linestyle=_get_alpha_linestyle(alpha),
                    marker=_get_xmax_marker(config["x_max"]),
                    markersize=4.5,
                    markerfacecolor='none',
                    markevery=28,
                    label='_nolegend_',
                )

        for plot_ax in plot_axes:
            plot_ax.axvline(rs_over_rvir, color='black', linestyle='-.', linewidth=1.2)
            plot_ax.axvline(0.5, color='grey', linestyle=':', linewidth=1.2)
            plot_ax.set_xscale('log')
            plot_ax.set_xlim(1.0e-3, 1.0)
            plot_ax.tick_params(axis='both', direction='in')

        if broken_y_ranges is None:
            _, y_max = bottom_ax.get_ylim()
            bottom_ax.set_ylim(0.0, y_max)
        else:
            y_min, y_max = broken_y_ranges[0]
            bottom_ax.set_ylim(y_min, y_max)
        for plot_ax in plot_axes:
            plot_ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            plot_ax.tick_params(axis='y', which='minor', direction='in', labelleft=False)

        bottom_ax.text(rs_over_rvir * 1.05, 0.04, r'$r_s$', transform=bottom_ax.get_xaxis_transform(), fontsize=10)
        bottom_ax.text(0.5 * 1.05, 0.04, r'$0.5R_{\rm vir}$', transform=bottom_ax.get_xaxis_transform(), fontsize=10, color='0.35')
        bottom_ax.set_xlabel(r'$r/R_{\rm vir}$', fontsize=14)
        if broken_y_ranges is None:
            bottom_ax.set_ylabel(r'Cumulative profile factor', fontsize=14)
        else:
            fig.supylabel(r'Cumulative profile factor', fontsize=14)
        bottom_ax.text(
            0.3,
            0.16,
            rf'$z={redshift:.0f}$, $M_{{\rm host}}=10^{{{np.log10(Mvir):.0f}}}\,M_\odot/h$',
            transform=bottom_ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.7', alpha=0.9),
        )
        profile_handles = [
            mlines.Line2D([], [], color='tab:blue', linewidth=2, label='Cooling'),
            mlines.Line2D([], [], color='tab:orange', linewidth=2, label=r'$H_{\rm gas}$'),
        ]
        seen_profile_labels = set()
        for config in profile_configs:
            label = rf'$H_{{\rm sub}}$, {_format_subhalo_profile_label(config)}'
            if label in seen_profile_labels:
                continue
            profile_handles.append(
                mlines.Line2D(
                    [], [],
                    color=_get_subhalo_profile_color(config),
                    linewidth=2,
                    label=label,
                )
            )
            seen_profile_labels.add(label)

        style_handles = [
            mlines.Line2D([], [], color='black', linewidth=1.8, linestyle=_get_alpha_linestyle(alpha), label=rf'$\alpha={alpha:.1f}$')
            for alpha in alpha_values
        ]
        x_max_values = sorted({config['x_max'] for config in profile_configs})
        if len(x_max_values) > 1:
            style_handles.extend([
                mlines.Line2D(
                    [], [],
                    color='black',
                    linestyle='None',
                    marker=_get_xmax_marker(x_max),
                    markerfacecolor='none',
                    markersize=5,
                    label=rf'$H_{{\rm sub}}$, $x_{{\max}}={x_max:.0f}$',
                )
                for x_max in x_max_values
            ])

        profile_legend = legend_ax.legend(
            handles=profile_handles,
            loc='upper left',
            fontsize=10.5,
            frameon=True,
        )
        legend_ax.add_artist(profile_legend)
        legend_ax.legend(
            handles=style_handles,
            loc='center left',
            fontsize=10.5,
            frameon=True,
        )
        if broken_y_ranges is None:
            plt.tight_layout()
        else:
            fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.96, hspace=0.05)

        safe_label = case["label"].replace(" ", "_").replace("(", "").replace(")", "")
        filename = os.path.join(output_dir, f'cooling_heating_profile_corr_{safe_label}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"saved cooling/heating-fraction plot: {filename}")

if __name__ == "__main__":

    #1. cosmic DF heating (integrate over HMF)
    # z_list = [15, 12, 10, 8, 6, 3, 0]
    # snapNum_list = [1, 2, 4, 8, 13, 25, 99]

    # for z, snapNum in zip(z_list, snapNum_list):
    #     plot_cosmic_DFheating(z, snapNum)

    test_cooling_heating_profile()
    # plot_cosmic_DFheating_multi_z(redshifts=[0, 6, 12], snapNums=[99, 13, 2])
    # plot_peak_lgM_cosmic_DFheating()

    #2. compare heating and cooling for a single host halo
    # run_heating_cooling_singlehost()
    # plot_global_heating_cooling_multi_z()
    # plot_heating_cooling_ratio_singlehost()

    # plot_global_heating_cooling_singlehost_minihalo(redshift = 15)
