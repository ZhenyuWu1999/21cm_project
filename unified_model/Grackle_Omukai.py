########################################################################
#
# Free-fall example script
#
#
# Copyright (c) 2013-2016, Grackle Development Team.
#
# Distributed under the terms of the Enzo Public Licence.
#
# The full license is in the file LICENSE, distributed with this
# software.
########################################################################
"""
important reactions in Grackle:
*   @  k31 @     H2I + p --> 2HI (k_H2I_diss)

*   @  k27 @     HM + p --> HI + e (k_HM_detach)
*   @  k28 @     H2II + p --> HI + HII (k_H2II_detach, Abel 1996 Reaction 25)

*   @  k7  @     HI + e --> HM + photon (kform_1 = get_kHM_GP98(T))
*   @  k8  @     HI + HM --> H2I* + e (kform_2 =  get_K_Tegmark97(3, T, np.nan)) 
kform_HM_eff = kform_1 * kform_2*nH / (kform_2*nH + k_HM_detach)
*   @  k9  @     HI + HII --> H2II + photon (kform_3 = get_K_Tegmark97(5, T, np.nan))
*   @  k10 @     H2II + HI --> H2I* + HII (kform_4 = get_K_Tegmark97(6, T, np.nan))
kform_H2II_eff = kform_3 * kform_4 * nH / (kform_4 * nH + k_H2II_detach)

collisional dissociation of H2I:
*   @  k12 @     H2I + e --> 2HI + e
*   @  k13 @     H2I + HI --> 3HI (or use k13dd as a more precise rate?)
 3-body reaction:
*   @  k21 @     2HI + H2I --> H2I + H2I
 
 """
import matplotlib.pyplot as plt
import os
import sys
import yt
from matplotlib.colors import LogNorm
from scipy.optimize import fsolve
import numpy as np
from collections import defaultdict
from scipy.interpolate import interp1d

from pygrackle import \
    chemistry_data, \
    evolve_constant_density, \
    evolve_freefall, \
    setup_fluid_container
from pygrackle.utilities.physical_constants import \
    mass_hydrogen_cgs, sec_per_year, sec_per_Myr, cm_per_mpc, gravitational_constant_cgs
from pygrackle.utilities.data_path import grackle_data_dir
from pygrackle.utilities.model_tests import \
    get_model_set, \
    model_test_format_version
from pygrackle.fluid_container import \
    FluidContainer
from pygrackle.utilities.evolve import add_to_data, calculate_collapse_factor

from HaloProperties import get_gas_lognH_analytic, get_mass_density_analytic
from Analytic_Model import get_heating_per_lgM
from physical_constants import Msun, h_Hubble, Omega_m, Omega_b, rho_crit_z0_kgm3, G_grav, mp, mu_minihalo
from Yoshida03 import get_kHM_GP98, get_K_Tegmark97, get_ss_factor_WG11

def theta_from_z(z, z_m):
    x = np.pi * ((1 + z) / (1 + z_m))**(3/2)
    f = lambda theta: theta - np.sin(theta) - x
    theta_guess = np.pi  # a good starting point for collapse
    theta_solution = fsolve(f, theta_guess)[0]
    return theta_solution

def get_rho_DM_spherical_collapse(z, z_m):
    #z_m: turnaround redshift 
    theta = theta_from_z(z, z_m)
    Omega_DM = Omega_m - Omega_b
    rho = 9*np.pi**2 / 2 * ((1+z_m)/(1-np.cos(theta)))**3 * Omega_DM* rho_crit_z0_kgm3
    theta_zm = np.pi
    rho_zm = 9*np.pi**2 / 2 * ((1+z_m)/(1-np.cos(theta_zm)))**3 * Omega_DM* rho_crit_z0_kgm3
    theta_vir = np.pi/2
    
    if theta < theta_vir: #virial phase, rho = 8*rho_zm
        rho = 8*rho_zm
        # z_vir = ((theta_vir - np.sin(theta_vir))/np.pi)**(2/3) * (1 + z_m) - 1
        # print(f"z_vir: {z_vir}")
    return rho

def test_rho_DM():
    output_dir = '/home/zwu/21cm_project/unified_model/Grackle_Omukai_results'

    #test theta_from_z and get_rho_DM_spherical_collapse
    z_list = np.linspace(20, 0, 200)
    z_m = 16
    theta_list = [theta_from_z(z, z_m) for z in z_list]
    rho_list = [get_rho_DM_spherical_collapse(z, z_m) for z in z_list]
    print("rho/mp [cm^-3]:", rho_list/mp/1e6)
    lognH_list = np.array([get_gas_lognH_analytic(z) for z in z_list])
    print("nH(gas):", 10**lognH_list)
    halo_density_list = np.array([get_mass_density_analytic(z) for z in z_list])
    print("halo_density/mp [cm^-3]:", halo_density_list/mp/1e6)

    
    rho_zvir = get_mass_density_analytic(4.45)
    print(f"rho_zvir/mp [cm^-3]: {rho_zvir/mp/1e6}")

    #plot theta vs z
    fig = plt.figure(figsize=(8, 6))
    plt.plot(z_list, theta_list, label='theta from z')
    plt.xlabel('z')
    plt.gca().invert_xaxis()
    plt.ylabel('theta')
    plt.title('Theta vs Redshift')
    plt.legend()
    filename = os.path.join(output_dir, 'theta_vs_z.png')
    plt.savefig(filename, dpi=300)

    #plot rho vs z
    fig = plt.figure(figsize=(8, 6))
    plt.plot(z_list, rho_list, label='rho from z', color='orange')
    plt.xlabel('z')
    plt.gca().invert_xaxis()
    plt.ylabel('rho (kg/m^3)')
    plt.title('Density vs Redshift')
    plt.legend()
    filename = os.path.join(output_dir, 'rho_vs_z.png')
    plt.savefig(filename, dpi=300)
    


def make_k13dd_interpolators(lnT_for_ktable, k13dd_flat):
    """
    Build 14 interpolators for the density-dependent H2 collisional dissociation
    coefficients from Martin et al. (1996). 
    Each interpolator maps ln(T) -> coefficient value.
    """
    # Reshape into (14, 600): 14 sets of coefficients, each tabulated on 600 T points
    coefs_14x600 = k13dd_flat.reshape(14, -1)
    interps = [
        interp1d(lnT_for_ktable, coefs_14x600[j],
                 kind='linear', bounds_error=False, fill_value="extrapolate")
        for j in range(14)
    ]
    return interps


def k13_density_dependent(interps, T, nH):
    """
    Reproduce the Grackle/Fortran implementation of the density-dependent
    H2 collisional dissociation rate (USE_DENSITY_DEPENDENT_H2_DISSOCIATION_RATE).

    Parameters
    ----------
    T : float or array
        Gas temperature in K.
    nH : float or array
        Neutral hydrogen number density in cm^-3.

    Returns
    -------
    k13 : float or array
        The total collisional dissociation rate coefficient (cm^3 s^-1).
    """
    # Prepare interpolators for all 14 coefficient sets
    # interps = make_k13dd_interpolators(lnT_for_ktable, k13dd_flat)

    T = np.asarray(T, dtype=float)
    nH = np.asarray(nH, dtype=float)
    lnT = np.log(T)   # natural log, consistent with Grackle tables

    # Interpolate coefficients at current T
    c = np.vstack([f(lnT) for f in interps])  # shape (14, ...)

    # Apply density cap at 1e9 cm^-3
    nh = np.minimum(nH, 1.0e9)

    # Direct collisional dissociation (CID), using Martin+96 formula
    cid_log10 = (
        c[0] - c[1] / (1.0 + (nh / c[4]) ** c[6]) +
        c[2] - c[3] / (1.0 + (nh / c[5]) ** c[6])
    )

    # Dissociative tunneling (DT)
    dt_log10 = (
        c[7]  - c[8]  / (1.0 + (nh / c[11]) ** c[13]) +
        c[9]  - c[10] / (1.0 + (nh / c[12]) ** c[13])
    )

    # Convert back to linear scale; protect with tiny lower bound
    tiny = 1e-40
    k13_CID = np.maximum(10.0 ** cid_log10, tiny)
    k13_DT  = np.maximum(10.0 ** dt_log10, tiny)

    k13_total = k13_CID + k13_DT

    # Valid only for 500 K <= T < 1e6 K
    valid = (T >= 500.0) & (T < 1.0e6)
    k13 = np.where(valid, k13_total, tiny)

    return k13




def run_grackle_Omukai(initial_conditions, final_nH, use_DFheating_flag, 
                       volumetric_heating_rate, use_LW_flag, LW_J21, spectrum_type):
    # for modified reaction rates (Shang+ 2010 Table A1, Agarwal+ 2015, 2016)
    alpha_LW = 0.0; beta_LW = 0.0
    if spectrum_type == "T4":
        alpha_LW = 2000; beta_LW = 3
    elif spectrum_type == "T5":
        alpha_LW = 0.1; beta_LW = 0.9
    


    
    #get initial conditions
    #initial_nH: initial hydrogen nuclei number density in cm^-3, not total density

    initial_redshift = initial_conditions["initial_redshift"]
    initial_nH = initial_conditions["initial_nH"]
    initial_Tgas = initial_conditions["initial_Tgas"]
    initial_ye = initial_conditions["initial_ye"]
    initial_y_H2I = initial_conditions["initial_y_H2I"]

    #hard-coded initial conditions
    initial_y_H2II = 1.0e-13 #Figure 4 of Galli& Palla 1998
    initial_y_HM = 1.0e-12  #Figure 4 of Galli& Palla 1998

    # Just run the script as is.
    metallicity = 0.
    # dictionary to store extra information in output dataset
    extra_attrs = {}

    # Set solver parameters
    my_chemistry = chemistry_data()
    my_chemistry.use_grackle = 1
    my_chemistry.with_radiative_cooling = 1
    my_chemistry.primordial_chemistry = 3
    my_chemistry.metal_cooling = 0
    my_chemistry.dust_chemistry = 0
    my_chemistry.photoelectric_heating = 0
    my_chemistry.CaseBRecombination = 1
    my_chemistry.cie_cooling = 1   #Flag to enable H2 collision-induced emission cooling from Ripamonti & Abel (2004).
    my_chemistry.h2_optical_depth_approximation = 1  # H2 cooling attenuation from Ripamonti & Abel (2004)
    my_chemistry.grackle_data_file = os.path.join(grackle_data_dir, "cloudy_metals_2008_3D.h5")
    # my_chemistry.grackle_data_file = os.path.join(grackle_data_dir, "CloudyData_UVB=HM2012.h5")
    # my_chemistry.grackle_data_file = os.path.join(grackle_data_dir, "CloudyData_UVB=HM2012_shielded.h5")
    
    print("Using grackle data file: ", my_chemistry.grackle_data_file)
    my_chemistry.use_volumetric_heating_rate = 1

    my_chemistry.UVbackground = 0
    my_chemistry.self_shielding_method = 0 #for HI and HeI self-shielding, not relevant for H2 self-shielding
    my_chemistry.H2_self_shielding = 3
    my_chemistry.LWbackground_intensity = LW_J21


    # Set units
    my_chemistry.comoving_coordinates = 0
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1. / (1. + initial_redshift) / \
        my_chemistry.a_units
    my_chemistry.density_units  = mass_hydrogen_cgs
    my_chemistry.length_units   = cm_per_mpc
    my_chemistry.time_units     = sec_per_Myr
    my_chemistry.set_velocity_units()

    # ---------------------- setup fluid container ------------------------

    metal_mass_fraction = metallicity * my_chemistry.SolarMetalFractionByMass
    dust_to_gas_ratio = metallicity * my_chemistry.local_dust_to_gas_ratio

    rval = my_chemistry.initialize()

    # kUnit = (pow(my_units->a_units, 3) * mh) / (densityBase1 * timeBase1)
    kUnit = mass_hydrogen_cgs/(my_chemistry.density_units * my_chemistry.time_units)
    kUnit_3Bdy = kUnit * mass_hydrogen_cgs/my_chemistry.density_units
    print("kUnit in python:", kUnit)
    if rval == 0:
        raise RuntimeError("Failed to initialize chemistry_data.")

    temperature = initial_Tgas 
    n_points = 1

    tiny_number = 1e-20

    fc = FluidContainer(my_chemistry, n_points)
    fh = my_chemistry.HydrogenFractionByMass
    d2h = my_chemistry.DeuteriumToHydrogenRatio

    metal_free = 1 - metal_mass_fraction
    H_total = fh * metal_free
    He_total = (1 - fh) * metal_free
    # someday, maybe we'll include D in the total
    D_total = H_total * d2h

    initial_density     = initial_nH * mass_hydrogen_cgs  / H_total  # g / cm^3
    final_density       = final_nH * mass_hydrogen_cgs / H_total  # g / cm^3

    fc_density = initial_density / my_chemistry.density_units
    tiny_density = tiny_number * fc_density

    state_vals = {
        "density": fc_density,
        "metal_density": metal_mass_fraction * fc_density,
        "dust_density": dust_to_gas_ratio * fc_density
    }

    # if state == "neutral":
    state_vals["HI_density"] = H_total * fc_density
    state_vals["HeI_density"] = He_total * fc_density
    state_vals["DI_density"] = D_total * fc_density

    state_vals["HII_density"] = initial_ye * state_vals["HI_density"]  #initial H+ (mass) fraction
    state_vals["HeIII_density"] = tiny_density  
    state_vals["e_density"] = state_vals["HII_density"] + state_vals["HeIII_density"] / 2 #initial free electron number density
    state_vals["H2I_density"] = 2* initial_y_H2I * state_vals["HI_density"] #initial H2 mass fraction
    state_vals["H2II_density"] = 2* initial_y_H2II * state_vals["HI_density"]  #initial H2+ mass fraction
    state_vals["HM_density"] = initial_y_HM * state_vals["HI_density"]  #initial H2- mass fraction



    for field in fc.density_fields:
        fc[field][:] = state_vals.get(field, tiny_density)

    fc.calculate_mean_molecular_weight()
    fc["internal_energy"] = temperature / \
        fc.chemistry_data.temperature_units / \
        fc["mean_molecular_weight"] / (my_chemistry.Gamma - 1.0)
    fc["x_velocity"][:] = 0.0
    fc["y_velocity"][:] = 0.0
    fc["z_velocity"][:] = 0.0

    #debug: heating
    if my_chemistry.use_volumetric_heating_rate:
        fc["volumetric_heating_rate"][:] = volumetric_heating_rate



    # ---------------------- evolve Omukai model ------------------------
    safety_factor=0.01
    include_pressure=True

    #debug: why 4 pi ?
    # print("density_units:", my_chemistry.density_units)
    # print("time_units:", my_chemistry.time_units)
    gravitational_constant = (
        4.0 * np.pi * gravitational_constant_cgs *
        # gravitational_constant_cgs *
        my_chemistry.density_units * my_chemistry.time_units**2)

    # some constants for the analytical free-fall solution
    freefall_time_constant = np.power(((32. * gravitational_constant) /
                                       (3. * np.pi)), 0.5)

    data = defaultdict(list)
    current_time = 0.0
    while fc["density"][0] * my_chemistry.density_units < final_density:
        # calculate timestep based on free-fall solution
        dt = safety_factor * \
          np.power(((3. * np.pi) /
                    (32. * gravitational_constant *
                     fc["density"][0])), 0.5)
        add_to_data(fc, data, extra={"time": current_time})

        # compute the new density using the modified
        # free-fall collapse as per Omukai et al. (2005)
        if include_pressure:
            force_factor = calculate_collapse_factor(data["pressure"], data["density"])
        else:
            force_factor = 0.
        data["force_factor"].append(force_factor)

        # calculate new density from altered free-fall solution
        new_density = np.power((np.power(fc["density"][0], -0.5) -
                                (0.5 * freefall_time_constant * dt *
                                 np.power((1 - force_factor), 0.5))), -2.)

        print("Evolve Freefall - t: %e yr, rho: %e g/cm^3, T: %e K." %
              ((current_time * my_chemistry.time_units / sec_per_year),
               (fc["density"][0] * my_chemistry.density_units),
               fc["temperature"][0]))
        
        print("volumetric heating rate:", fc["volumetric_heating_rate"])
        # test if internal energy is just adiabatic heating

        # use this to multiply by elemental densities if you are tracking those
        density_ratio = new_density / fc["density"][0]

        # update densities
        for field in fc.density_fields:
            fc[field] *= density_ratio

        # now update energy for adiabatic heating from collapse
        fc["internal_energy"][0] += (my_chemistry.Gamma - 1.) * \
          fc["internal_energy"][0] * freefall_time_constant * \
          np.power(fc["density"][0], 0.5) * dt

        #H2 photo-dissociation and H- photo-detachment under LW background
        if use_LW_flag:
            #H2 photo-dissociation
            print(f"time units = {my_chemistry.time_units:.5e} s")
            #unit of k31 and k27: 1/[time], so k31~ = k31/[k31] = k31 * [time]
            k31 = 1.0e-12 * beta_LW * LW_J21 * my_chemistry.time_units
            fc.chemistry_data.k31 = k31
            #H- photo-detachment
            k27 = 1.0e-10 * alpha_LW * LW_J21 * my_chemistry.time_units
            fc.chemistry_data.k27 = k27



            print(f"Using LW J21 = {LW_J21}, k31 = {k31}, k27 = {k27}")

        fc.solve_chemistry(dt)
        # current_nH = fc["HI_density"]
        # print(f"current nH: {current_nH} cm^-3, final nH: {final_nH} cm^-3")
        # current_T = fc["temperature"]
        # print(f"current Temperature: {current_T} K")

        #to do: add rates to data 

        # update time
        current_time += dt

    for field in data:
        data[field] = np.squeeze(np.array(data[field]))

    NumberOfTemperatureBins = 600
    TemperatureStart = 1.0
    TemperatureEnd = 1.0e9
    # d_logT = (np.log10(TemperatureEnd) - np.log10(TemperatureStart)) / (NumberOfTemperatureBins - 1)
    T_for_ktable = np.logspace(np.log10(TemperatureStart), np.log10(TemperatureEnd), NumberOfTemperatureBins)
    lnT_for_ktable = np.log(T_for_ktable)
    k7_cgs = fc.chemistry_data.k7 * kUnit #k7 in cgs units [cm^3/s]
    k8_cgs = fc.chemistry_data.k8 * kUnit #k8
    k9_cgs = fc.chemistry_data.k9 * kUnit #k9
    k10_cgs = fc.chemistry_data.k10 * kUnit #k10

    k12_cgs = fc.chemistry_data.k12 * kUnit #k12
    k13_cgs = fc.chemistry_data.k13 * kUnit #k13
    k13dd = fc.chemistry_data.k13dd  #k13dd

    k21_cgs = fc.chemistry_data.k21 * kUnit_3Bdy #k21, 3-body reaction
    k_table = {
        "k7": k7_cgs,
        "k8": k8_cgs,
        "k9": k9_cgs,
        "k10": k10_cgs,
        "k12": k12_cgs,
        "k13": k13_cgs,
        "k13dd": k13dd,
        "k21": k21_cgs,
        "T_for_ktable": T_for_ktable,
        "lnT_for_ktable": lnT_for_ktable,
        "kUnit": kUnit,
    }

    return fc.finalize_data(data=data), k_table

def get_Jeans_length_cgs(T_list, rho_list, mu_list):
    G_cgs = gravitational_constant_cgs
    kB_cgs = 1.380649e-16  # erg/K
    mH_cgs = mass_hydrogen_cgs  # g
    Jeans_length = np.sqrt(
        (np.pi* kB_cgs * T_list) / (G_cgs * rho_list * mu_list * mH_cgs))
    
    return Jeans_length # cm

def main():

    #use Sugimara, Omukai & Inoue (2014) initial conditions
    initial_redshift = 16.0
    initial_nH = 1.0e-1 # cm^-3, lower than get_gas_lognH_analytic() because the halo has not collapsed yet
    initial_Tgas = 21
    initial_ye = 3.7e-4
    initial_y_H2I = 2.0e-6

    initial_conditions = {
        "initial_redshift": initial_redshift,
        "initial_nH": initial_nH,
        "initial_Tgas": initial_Tgas,
        "initial_ye": initial_ye,
        "initial_y_H2I": initial_y_H2I
    }


    volumetric_heating_rate = 0.0
    use_DFheating_flag = 0

    if use_DFheating_flag:
        lgMhalo = 7 #Msun/h
        DF_heating_data = get_heating_per_lgM([lgMhalo],[-3],[0],initial_redshift,'BestFit_z', mu_minihalo)
        tot_heating = DF_heating_data['Heating_singlehost']
        tot_heating_erg = tot_heating * 1.0e7 # erg/s
        halo_density = get_mass_density_analytic(initial_redshift)
        halo_volume = 10**lgMhalo * Msun / h_Hubble / halo_density
        halo_volume_cm3 = halo_volume * 1.0e6 # cm^3
        volumetric_heating_rate = tot_heating_erg / halo_volume_cm3
        print(f"DF heating rate: {tot_heating_erg} erg/s, volumetric heating rate: {volumetric_heating_rate} erg/cm^3/s")

        volumetric_heating_rate = volumetric_heating_rate[0]

    use_LW_flag = 1
    LW_J21 = 1800
    spectrum_type = "T5"  

    final_nH = 1e10 # cm^-3

    data, k_table = run_grackle_Omukai(initial_conditions, final_nH, use_DFheating_flag,
                       volumetric_heating_rate, use_LW_flag, LW_J21, spectrum_type)
    
    print(data.keys())    
    
    # dict_keys(['internal_energy', 'x_velocity', 'y_velocity', 'z_velocity', 'density',
    #   'HI_density', 'HII_density', 'HeI_density', 'HeII_density', 'HeIII_density', 'e_density',
    #     'H2I_density', 'H2II_density', 'HM_density', 'DI_density', 'DII_density', 'HDI_density', 
    #     'cooling_time', 'dust_temperature', 'gamma', 'pressure', 'temperature', 'cooling_rate', 
    #     'mean_molecular_weight', 'time', 'force_factor'])
    
    print(data['time'].in_units('Myr'))

    time = data["time"].in_units("Myr")
    density = data["density"].v
    print("density:", density)
    mu_list = data["mean_molecular_weight"]
    nH_list = data["HI_density"].v/ mass_hydrogen_cgs
    temperature_list = data["temperature"].v
    y_e = data["e_density"] / data["HI_density"]
    ne_list = y_e * nH_list
    y_H2I = data["H2I_density"]/2 / data["HI_density"]
    nH2_list = y_H2I * nH_list
    Jeans_length = get_Jeans_length_cgs(temperature_list, density, mu_list)
    #column density
    NH2_list = nH2_list * Jeans_length


    output_dir = '/home/zwu/21cm_project/unified_model/Grackle_Omukai_results'
    output_name = f"Grackle_Omukai"
    if use_DFheating_flag:
        output_name += f"_DFheating_lgM_{lgMhalo}"
    if use_LW_flag:
        output_name += f"_LW_{LW_J21}_spec_{spectrum_type}"

    output_name += ".png"
    output_name = os.path.join(output_dir, output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    #interpolate log temperature to get the reaction rates
    kUnit = k_table["kUnit"]
    lnT_for_ktable = k_table["lnT_for_ktable"]
    # k7_list = np.interp(np.log(temperature_list), lnT_for_ktable, k_table["k7"])
    # k8_list = np.interp(np.log(temperature_list), lnT_for_ktable, k_table["k8"])
    # k9_list = np.interp(np.log(temperature_list), lnT_for_ktable, k_table["k9"])
    # k10_list = np.interp(np.log(temperature_list), lnT_for_ktable, k_table["k10"])
    #use high-order interp
    k7_cubic = interp1d(lnT_for_ktable, k_table["k7"], kind='cubic')
    k8_cubic = interp1d(lnT_for_ktable, k_table["k8"], kind='cubic')
    k9_cubic = interp1d(lnT_for_ktable, k_table["k9"], kind='cubic')
    k10_cubic = interp1d(lnT_for_ktable, k_table["k10"], kind='cubic')
    k12_cubic = interp1d(lnT_for_ktable, k_table["k12"], kind='cubic')
    k13_cubic = interp1d(lnT_for_ktable, k_table["k13"], kind='cubic')
    k21_cubic = interp1d(lnT_for_ktable, k_table["k21"], kind='cubic')

    k7_list = k7_cubic(np.log(temperature_list))
    k8_list = k8_cubic(np.log(temperature_list))
    k9_list = k9_cubic(np.log(temperature_list))
    k10_list = k10_cubic(np.log(temperature_list))
    k12_list = k12_cubic(np.log(temperature_list))
    k13_list = k13_cubic(np.log(temperature_list))
    k21_list = k21_cubic(np.log(temperature_list))

    
    k13dd_flat = k_table["k13dd"]
    k13dd_interps = make_k13dd_interpolators(lnT_for_ktable, k13dd_flat)
    k13dd_GrackleUnit_list = k13_density_dependent(k13dd_interps, temperature_list, nH_list)
    k13dd_list = k13dd_GrackleUnit_list * kUnit #in cgs units

    if spectrum_type == "T4":
        alpha_LW = 2000; beta_LW = 3
    elif spectrum_type == "T5":
        alpha_LW = 0.1; beta_LW = 0.9

    ss_list = np.array([get_ss_factor_WG11(T, NH2) for T, NH2 in zip(temperature_list, NH2_list)])
    print("ss_list:", ss_list)
    k31 = 1.0e-12 * beta_LW * LW_J21 
    k27 = 1.0e-10 * alpha_LW * LW_J21
    k31_list = np.ones(len(temperature_list)) * k31 * ss_list
    k27_list = np.ones(len(temperature_list)) * k27 * ss_list
    print("nH_list:", nH_list)
    print("k7_list:", k7_list)
    kform_HM_eff = k7_list * k8_list * nH_list / (k8_list * nH_list + k27_list)
    kform_H2II_eff = k9_list #do not include H2II photo-detachment now, only H2+ formation
    kform_total = kform_HM_eff + kform_H2II_eff
    print("k12_list:", k12_list)
    #use k13dd_list instead of k13_list
    print("k13dd_list:", k13dd_list)
    kdiss_total = k31_list + k12_list * ne_list + k13dd_list * nH_list + k21_list * nH_list**2
    y_H2I_Eq = kform_total * y_e * nH_list / kdiss_total


    fig, ax1 = plt.subplots(figsize=(8, 6))
    # Temperature vs. nH (left y-axis)
    sc1 = ax1.scatter(nH_list, temperature_list, c=time, cmap='viridis', label="T$_{gas}$", s=20)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$n_{\rm H}$ [cm$^{-3}$]", fontsize=14)
    ax1.set_ylabel("T [K]", fontsize=14)
    ax1.tick_params(axis='y')
    # Second y-axis for H2 abundance
    ax3 = ax1.twinx()
    line2, = ax3.plot(nH_list, y_H2I, linestyle='-', color='blue', label=r"$n_{\rm H_2}/n_{\rm H}$")
    line3, = ax3.plot(nH_list, y_H2I_Eq, linestyle='--', color='blue', label=r"$n_{\rm H_2}/n_{\rm H}$ (eq)")
    line4, = ax3.plot(nH_list, y_e, linestyle=':', color='red', label=r"$y_e = n_e/n_{\rm H}$")
    ax3.set_yscale("log")
    ax3.set_ylabel("species abundance", fontsize=15, labelpad=12)  # add labelpad to space it out
    ax3.tick_params(axis='y')
    # Colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sc1, cax=cbar_ax)
    cbar.set_label("Time [Myr]", fontsize=14)
    # Combine legends from both axes
    lines = [sc1, line2, line3, line4]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=12)
    # Adjust layout
    fig.subplots_adjust(left=0.1, right=0.75, bottom=0.12, top=0.9)
    # fig.subplots_adjust(left=0.1, right=0.88, bottom=0.12, top=0.9)
    plt.savefig(output_name, dpi=300)
    plt.close()

    print(f"Output saved to {output_name}")


    #second figure:  compare reaction rates
    k7_GP98 = np.array([get_kHM_GP98(T) for T in temperature_list])
    k8_Tegmark = np.array([get_K_Tegmark97(3, T, np.nan) for T in temperature_list])
    k9_Tegmark = np.array([get_K_Tegmark97(5, T, np.nan) for T in temperature_list])
    k10_Tegmark = np.array([get_K_Tegmark97(6, T, np.nan) for T in temperature_list])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(nH_list, k7_list, label='k7 HM formation (Grackle)', color='red')
    ax.plot(nH_list, k7_GP98, label='k7 HM formation (GP98)', color='red', linestyle='--')
    ax.plot(nH_list, k8_list, label='k8 H2I formation via HM (Grackle)', color='orange')
    ax.plot(nH_list, k8_Tegmark, label='k8 H2I formation via HM (Tegmark97)', color='orange', linestyle='--')
    ax.plot(nH_list, k9_list, label='k9 H2+ formation (Grackle)', color='green')
    ax.plot(nH_list, k9_Tegmark, label='k9 H2+ formation (Tegmark97)', color='green', linestyle='--')
    ax.plot(nH_list, k10_list, label='k10 H2I formation via H2+ (Grackle)', color='blue')
    ax.plot(nH_list, k10_Tegmark, label='k10 H2I formation via H2+ (Tegmark97)', color='blue', linestyle='--')
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_{\rm H}$ [cm$^{-3}$]", fontsize=14)
    ax.set_ylabel("Reaction Rate [cm$^3$/s]", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc='upper right', fontsize=12)   
    ax.set_title("Reaction Rates vs. nH", fontsize=16)
    output_name = output_name.replace(".png", "_reaction_rates.png")
    plt.savefig(output_name, dpi=300)
    plt.close()
    print(f"Reaction rates plot saved to {output_name}")


    # save data arrays as a yt dataset
    # yt.save_as_dataset({}, f"{output_name}.h5",
    #                    data=data, extra_attrs=extra_attrs)

    

if __name__ == "__main__":
    main()
