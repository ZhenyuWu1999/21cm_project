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

import matplotlib.pyplot as plt
import os
import sys
import yt
from matplotlib.colors import LogNorm

from pygrackle import \
    chemistry_data, \
    evolve_constant_density, \
    evolve_freefall, \
    setup_fluid_container
from pygrackle.utilities.physical_constants import \
    mass_hydrogen_cgs, \
    sec_per_Myr, \
    cm_per_mpc
from pygrackle.utilities.data_path import grackle_data_dir
from pygrackle.utilities.model_tests import \
    get_model_set, \
    model_test_format_version

from HaloProperties import get_mass_density_analytic
from Analytic_Model import get_heating_per_lgM
from physical_constants import Msun, h_Hubble

def run_grackle_freefall(redshift,initial_lognH,final_lognH,initial_temperature, 
                         cooling_temperature, volumetric_heating_rate, lg_LW_J21):

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
    my_chemistry.self_shielding_method = 0
    my_chemistry.H2_self_shielding = 0
    my_chemistry.CaseBRecombination = 1
    my_chemistry.cie_cooling = 1   #Flag to enable H2 collision-induced emission cooling from Ripamonti & Abel (2004).
    my_chemistry.h2_optical_depth_approximation = 1  # H2 cooling attenuation from Ripamonti & Abel (2004)
    my_chemistry.grackle_data_file = os.path.join(
        grackle_data_dir, "cloudy_metals_2008_3D.h5")
    print("Using grackle data file: ", my_chemistry.grackle_data_file)
    my_chemistry.use_volumetric_heating_rate = 1
    my_chemistry.LWbackground_intensity = 10**lg_LW_J21

    # redshift = 0.

    # Set units
    my_chemistry.comoving_coordinates = 0
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1. / (1. + redshift) / \
        my_chemistry.a_units
    my_chemistry.density_units  = mass_hydrogen_cgs
    my_chemistry.length_units   = cm_per_mpc
    my_chemistry.time_units     = sec_per_Myr
    my_chemistry.set_velocity_units()

    # set initial density and temperature
    # initial_temperature = 50000.
    initial_nH = 10.**initial_lognH
    final_nH = 10.**final_lognH
    # initial_density     = 1e-1 * mass_hydrogen_cgs # g / cm^3
    # final_density       = 1e12 * mass_hydrogen_cgs

    initial_density     = initial_nH * mass_hydrogen_cgs # g / cm^3
    final_density       = final_nH * mass_hydrogen_cgs

    metal_mass_fraction = metallicity * my_chemistry.SolarMetalFractionByMass
    dust_to_gas_ratio = metallicity * my_chemistry.local_dust_to_gas_ratio
    fc = setup_fluid_container(
        my_chemistry,
        density=initial_density,
        temperature=initial_temperature,
        metal_mass_fraction=metal_mass_fraction,
        dust_to_gas_ratio=dust_to_gas_ratio,
        state="ionized",
        converge=False)

    # let the gas cool at constant density from the starting temperature
    # down to a lower temperature to get the species fractions in a
    # reasonable state.
    data0 = evolve_constant_density(
        fc, final_temperature=cooling_temperature,
        safety_factor=0.1)
    #debug: different H2 fractions
    fc["H2I_density"][:] = fc["density"][:]*1.0e-6
    # print(data0['time'].in_units('Myr'))  
    # print(data0["H2I_density"] / data0["density"])

    if my_chemistry.use_volumetric_heating_rate:
        fc["volumetric_heating_rate"][:] = volumetric_heating_rate

    # evolve density and temperature according to free-fall collapse
    data = evolve_freefall(fc, final_density,
                           safety_factor=0.01,
                           include_pressure=True)
    
    return data0, data

if __name__ == "__main__":
    # run the free-fall example

    redshift = 12.0
    initial_lognH = 1.0

    final_lognH = 12
    initial_temperature = 5000
    cooling_temperature = 100
    
    volumetric_heating_rate = 0.0
    use_DFheating_flag = 0

    if use_DFheating_flag:
        lgMhalo = 9 #Msun/h
        DF_heating_data = get_heating_per_lgM([lgMhalo],[-3],[-1],redshift,'BestFit_z')
        tot_heating = DF_heating_data['Heating_singlehost']
        tot_heating_erg = tot_heating * 1.0e7 # erg/s
        halo_density = get_mass_density_analytic(redshift)
        halo_volume = 10**lgMhalo * Msun / h_Hubble / halo_density
        halo_volume_cm3 = halo_volume * 1.0e6 # cm^3
        volumetric_heating_rate = tot_heating_erg / halo_volume_cm3
        print(f"DF heating rate: {tot_heating_erg} erg/s, volumetric heating rate: {volumetric_heating_rate}")

        volumetric_heating_rate = volumetric_heating_rate[0]

    use_LW_flag = 1
    lg_LW_J21 = 2
    data0, data = run_grackle_freefall(redshift, initial_lognH, final_lognH, initial_temperature, 
                                        cooling_temperature, volumetric_heating_rate, lg_LW_J21)
                       
    print(data.keys())    
    """
    dict_keys(['internal_energy', 'x_velocity', 'y_velocity', 'z_velocity', 'density',
      'HI_density', 'HII_density', 'HeI_density', 'HeII_density', 'HeIII_density', 'e_density',
        'H2I_density', 'H2II_density', 'HM_density', 'DI_density', 'DII_density', 'HDI_density', 
        'cooling_time', 'dust_temperature', 'gamma', 'pressure', 'temperature', 'cooling_rate', 
        'mean_molecular_weight', 'time', 'force_factor'])
    """
    print(data0['time'].in_units('Myr'))
    print(data['time'].in_units('Myr'))

    # output_dir = '/home/zwu/21cm_project/unified_model/Grackle_freefall_results'
    # output_name = os.path.join(output_dir, "Grackle_freefall.png")
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)


    time = data["time"].in_units("Myr")
    density = data["density"]
    nH = density / mass_hydrogen_cgs
    temperature = data["temperature"]
    f_H2 = data["H2I_density"] / density

    # print("mu for data0: ", data0["mean_molecular_weight"])
    # print("mu for data: ", data["mean_molecular_weight"])

    output_dir = '/home/zwu/21cm_project/unified_model/Grackle_freefall_results'
    output_name = f"Grackle_ff_lognH_{initial_lognH:.2e}_Tcool_{cooling_temperature}_z{redshift}"
    if use_DFheating_flag:
        output_name += "_DFheating_lgM_{lgMhalo}"
    if use_LW_flag:
        output_name += f"_LW_{lg_LW_J21}"
    output_name += ".png"
    output_name = os.path.join(output_dir, output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Temperature vs. density (left y-axis)
    sc1 = ax1.scatter(density, temperature, c=time, cmap='viridis', label="T$_{gas}$", s=20)
    
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\rho$ [g/cm$^3$]", fontsize=14)
    ax1.set_ylabel("T [K]", fontsize=14)
    ax1.tick_params(axis='y')

    # Second y-axis for f_H2
    ax3 = ax1.twinx()
    sc2 = ax3.scatter(density, f_H2, c=time, cmap='viridis', marker='o', s=5, facecolors='none', edgecolors='face')
    ax3.set_yscale("log")
    ax3.yaxis.set_label_coords(1.02, 0.5)  # [x, y] in axes fraction
    ax3.set_ylabel("f$_{H2}$", fontsize=15)
    ax3.tick_params(axis='y')

    # Top x-axis for n_H
    def density_to_nH(x):
        return x / mass_hydrogen_cgs
    def nH_to_density(x):
        return x * mass_hydrogen_cgs

    ax2 = ax1.secondary_xaxis('top', functions=(density_to_nH, nH_to_density))
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$n_{\rm H}$ [cm$^{-3}$]", fontsize=14)

    # Add manual colorbar to the right
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    cbar = fig.colorbar(sc1, cax=cbar_ax)
    cbar.set_label("Time [Myr]", fontsize=14)


    # Legend combining both plots
    handles = [sc1, sc2]
    labels = ["T$_{gas}$", "f$_{H2}$"]
    ax1.legend(handles, labels, loc="lower right")

    fig.subplots_adjust(left=0.1, right=0.8, bottom=0.12, top=0.9)
    plt.savefig(output_name, dpi=300)
    plt.close()


 


    # save data arrays as a yt dataset
    # yt.save_as_dataset({}, f"{output_name}.h5",
    #                    data=data, extra_attrs=extra_attrs)
