########################################################################
#
# Cooling rate example script
#
#
# Copyright (c) 2013-2016, Grackle Development Team.
#
# Distributed under the terms of the Enzo Public Licence.
#
# The full license is in the file LICENSE, distributed with this
# software.
########################################################################

import numpy as np
import os
import sys
import yt
import argparse
import h5py
import matplotlib.pyplot as plt

from pygrackle import \
    chemistry_data, \
    evolve_constant_density, \
    setup_fluid_container    
    
from pygrackle.utilities.data_path import grackle_data_dir
from pygrackle.utilities.physical_constants import \
    mass_hydrogen_cgs, \
    sec_per_Myr, \
    cm_per_mpc
from pygrackle.utilities.model_tests import \
    get_model_set, \
    model_test_format_version


def run_cool_rate(evolve_cooling,redshift,lognH,specific_heating_rate, volumetric_heating_rate, temperature, gas_metallicity):
    '''
    print(f"Current redshift = {redshift}")
    print(f"nH = rho/mH [1/cm^3], log(nH) = {lognH}")
    print(f"Specific heating rate = {specific_heating_rate} [erg/g/s]")
    print(f"Volumetric heating rate = {volumetric_heating_rate} [erg/cm^3/s]")
    print(f"Temperature = {temperature} [K]")
    print(f"Gas metallicity = {gas_metallicity} [Zsun]")
    '''
    
    tiny_number = 1e-20
    
    nH = 10**lognH 
    
    # dictionary to store extra information in output dataset
    extra_attrs = {}

    # Set solver parameters
    my_chemistry = chemistry_data()
    my_chemistry.use_grackle = 1
    
    if evolve_cooling:
        my_chemistry.with_radiative_cooling = 1
    else:
        my_chemistry.with_radiative_cooling = 0 
        
    my_chemistry.primordial_chemistry = 3
    my_chemistry.metal_cooling = 1
    my_chemistry.UVbackground = 0
    my_chemistry.self_shielding_method = 0
    my_chemistry.H2_self_shielding = 0
    my_chemistry.grackle_data_file = \
        os.path.join(grackle_data_dir, "CloudyData_UVB=HM2012.h5")

    my_chemistry.use_specific_heating_rate = 1
    my_chemistry.use_volumetric_heating_rate = 1

    # Set units
    my_chemistry.comoving_coordinates = 0 # proper units
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1.0 / (1.0 + redshift) / \
        my_chemistry.a_units
    my_chemistry.density_units = mass_hydrogen_cgs # rho = 1.0 is 1.67e-24 g
    my_chemistry.length_units = cm_per_mpc         # 1 Mpc in cm
    my_chemistry.time_units = sec_per_Myr          # 1 Gyr in s
    my_chemistry.set_velocity_units()
    my_chemistry.Compton_xray_heating = 0
    
    if redshift <= 6:
        state = "ionized"
    else:
        state = "neutral"

    density = nH * mass_hydrogen_cgs
    
    if np.log10(gas_metallicity)<-6:
        gas_metallicity = 1e-6  #valid range minimum value
    #metallicity = 0.0 # Solar   #assume primordial gas
    metal_mass_fraction = gas_metallicity * my_chemistry.SolarMetalFractionByMass
    #(SolarMetalFractionByMass: 0.01295)
    #metal_mass_fraction = gas_metallicity
    
    # Call convenience function for setting up a fluid container.
    # This container holds the solver parameters, units, and fields.
    
    #temperature = np.logspace(1, 9, 200)
    max_iterations = 100000
    if(specific_heating_rate != 0.0 or volumetric_heating_rate != 0.0):
        print("Include DF Heating in Fluid Container")
    else:
        print("Zero Heating in Fluid Container") 
    
    fc = setup_fluid_container(
        my_chemistry,
        density=density,
        temperature=temperature,
        state=state,
        metal_mass_fraction=metal_mass_fraction,
        dust_to_gas_ratio=None,
        converge=True,
        tolerance=0.01,
        max_iterations=max_iterations)
    

    if my_chemistry.use_specific_heating_rate:
        fc["specific_heating_rate"][:] = specific_heating_rate
    if my_chemistry.use_volumetric_heating_rate:
        fc["volumetric_heating_rate"][:] = volumetric_heating_rate
    
    
    if evolve_cooling:
        final_time = final_time = 100. # Myr
        data = evolve_constant_density(
            fc, final_time=final_time,
            safety_factor=0.01)
    else:    
        # get data arrays with symbolic units
        data = fc.finalize_data()
    
    return data



def plot_single_cooling_rate(data, output_filename):
    fig = plt.figure(figsize=(8, 6),facecolor='white')
    plt.loglog(data["temperature"], np.abs(data["cooling_rate"]),
          color="black")
    plt.xlabel('T [K]')
    plt.ylabel('$\\Lambda$ [erg s$^{-1}$ cm$^{3}$]')
    plt.tight_layout()
    plt.savefig(output_filename)


def print_attrs(name, obj):
    """Helper function to print the name of an HDF5 object and its attributes."""
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))

def display_hdf5_contents(filepath):
        # Open the HDF5 file in read-only mode
    with h5py.File(filepath, 'r') as f:       
        # Display all groups and datasets in the file
        for name, item in f.items():
            if isinstance(item, h5py.Group):
                print(f"Group: {name}")
                for key, val in item.attrs.items():
                    print(f"    {key}: {val}")
                for subname, subitem in item.items():
                    print_attrs(f"{name}/{subname}", subitem)
            elif isinstance(item, h5py.Dataset):
                print(f"Dataset: {name}")
                for key, val in item.attrs.items():
                    print(f"    {key}: {val}")

def read_hdf5_data(filepath):
    data_dict = {}
    # Open the HDF5 file in read-only mode
    with h5py.File(filepath, 'r') as f:
        # Function to recursively read data
        def read_recursive(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Store dataset data in dictionary
                data_dict[name] = np.array(obj)
                #print(f"Dataset: {name} loaded")
                # for key, val in obj.attrs.items():
                #     print(f"    {key}: {val}")
            elif isinstance(obj, h5py.Group):
                #print(f"Group: {name}")
                #for key, val in obj.attrs.items():
                    #print(f"    {key}: {val}")
                for subname, subitem in obj.items():
                    read_recursive(f"{name}/{subname}", subitem)
        # Read data starting from root
        f.visititems(read_recursive)
    return data_dict

def plot_multiple_cooling_rates(data_list, output_filename):
    colors = ['red', 'orange', 'yellow', 'green', 'blue']
    fig = plt.figure(facecolor='white',figsize=(6,6))
    ax = fig.add_subplot(111)
    
    for i, data in enumerate(data_list):
        temperature = data["data/temperature"]
        cooling_time = data["data/cooling_time"]
        cooling_rate = data["data/cooling_rate"]
        turning_point = np.where(cooling_time<0)[0][0]
        #cooling_rate = np.abs(cooling_rate)
        
        ax.loglog(temperature[0:turning_point], cooling_rate[0:turning_point], color=colors[i], linestyle='--')
        ax.loglog(temperature[turning_point:], -cooling_rate[turning_point:], color=colors[i], linestyle='-', label=f"log(nH)={lognH_list[i]}")
    
   
    ax.set_ylim(1e-29, 1e-21)    
    ax.set_xlabel('T [K]')
    ax.set_ylabel('$\\Lambda$ [erg s$^{-1}$ cm$^{3}$]')

    ax.legend()
    plt.tight_layout()
    
    plt.savefig(output_filename,dpi=300)


if __name__ == "__main__":
    
    # Lists of parameters
    data_name = 'zero_metallicity_noUVB'
    current_redshift_list = [20.0]
    #current_redshift_list = [20.0, 15.2, 15.0, 10.0, 0.0]
    
    lognH_list = [-5, -2, 0, 1, 3]
    temperature = np.logspace(1, 9, 200)
    specific_heating_rate = 0.0
    volumetric_heating_rate = 0.0
    gas_metallicity = 0.0
    evolve_cooling = False
    
    
    for redshift in current_redshift_list:
        for lognH in lognH_list:
            data = run_cool_rate(evolve_cooling,redshift,lognH,specific_heating_rate, volumetric_heating_rate, temperature, gas_metallicity)
            print(f"Current redshift = {redshift}")
            print(f"nH = rho/mH [1/cm^3], log(nH) = {lognH}")

            output_filename = f'new_data/{data_name}/DF_cooling_rate_z{redshift:.1f}_lognH{lognH:.0f}'
            ds_name = output_filename + '.h5'
            im_name = output_filename + '.png'
            
            #yt.save_as_dataset({}, ds_name, data)
            
            #plot_single_cooling_rate(data, im_name) 
                   
    
    '''
    for current_redshift in current_redshift_list:
        file_list = []
        data_list = []
        for lognH in lognH_list: 
            
            filename = f'new_data/{data_name}/DF_cooling_rate_z{current_redshift:.1f}_lognH{lognH:.0f}.h5'
            file_list.append(filename)
            data_list.append(read_hdf5_data(filename))
        
        output_filename = f'new_figures/{data_name}/DF_cooling_rate_z{current_redshift:.1f}.png'
        plot_multiple_cooling_rates(data_list, output_filename)
    '''
    
    '''
    redshift = 15.2
    lognH = 0
    temperature = 1e5
    data = run_cool_rate(evolve_cooling,redshift,lognH,specific_heating_rate, volumetric_heating_rate, temperature, gas_metallicity)
    
    fig = plt.figure(figsize=(8, 6),facecolor='white')
    p1, = plt.loglog(data["time"].to("Myr"),
                        data["temperature"],
                        color="black", label="T")
    plt.xlabel("Time [Myr]")
    plt.ylabel("T [K]")
    plt.twinx()
    p2, = plt.semilogx(data["time"].to("Myr"),
                          data["mean_molecular_weight"],
                          color="red", label="$\\mu$")
    plt.ylabel("$\\mu$")
    plt.legend([p1,p2],["T","$\\mu$"], fancybox=True,
                  loc="center left")
    plt.tight_layout()
    plt.savefig('new_figures/zero_metallicity_CoolingCell/DF_cooling_rate_z15.2_T_mu.png')
    
    
    #plot cooling rate
    fig = plt.figure(figsize=(8, 6),facecolor='white')
    plt.plot(data["time"].to("Myr"), -data["cooling_rate"],
          color="black")
    plt.xlabel('Time [Myr]')
    plt.ylabel('$\\Lambda$ [erg s$^{-1}$ cm$^{3}$]')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('new_figures/zero_metallicity_CoolingCell/DF_cooling_rate_z15.2.png')
    '''
    
    