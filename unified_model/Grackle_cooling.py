
import numpy as np
import os
import matplotlib.pyplot as plt

from pygrackle.utilities.physical_constants import sec_per_Myr, cm_per_mpc, mass_hydrogen_cgs
from pygrackle.utilities.data_path import grackle_data_dir
from pygrackle import \
    chemistry_data, \
    setup_fluid_container    
    
from physical_constants import kB, eV, Zsun, Mpc, Myr, h_Hubble
from Grackle_evolve import *
from Config import simulation_set
from TNGDataHandler import load_processed_data
from HaloProperties import get_gas_lognH_analytic, get_gas_lognH_numerical

def run_constdensity_model(evolve_cooling,redshift,lognH, specific_heating_rate, 
                           volumetric_heating_rate, temperature, gas_metallicity, 
                           f_H2, **kwargs):
    '''
    Wrapper function to set up and run constant density chemistry model.
    
    Parameters:
    -----------
    evolve_cooling : bool
        Whether to evolve the gas with radiative cooling
    redshift : float
        Cosmological redshift
    lognH : float
        Log10 of hydrogen number density in cm^-3
    specific_heating_rate : float
        Specific heating rate in [erg/g/s]
    volumetric_heating_rate : float or callable or tuple
        Can be:
        - float: constant volumetric heating rate in [erg/cm^3/s]
        - callable: function that takes time (in code units) and returns heating rate
        - tuple: (times, rates) arrays for interpolation
        temperature : float
        Initial gas temperature in K
    gas_metallicity : float
        Gas metallicity in [Zsun]
    f_H2 : float
        H2 mass fraction
    **kwargs:
        UVB_flag : bool
            Whether to include UV background
        Compton_Xray_flag : bool
            Whether to include Compton X-ray heating
        dynamic_final_flag : bool
            Whether to use dynamic final time evolution
        final_time : float, optional
            Final time in Myr (default: 50)
        data_file: str, optional
            cooling/heating data files used for interpolation (default: "CloudyData_UVB=HM2012.h5")

    '''
    
    UVB_flag = kwargs.get('UVB_flag', True)
    Compton_Xray_flag = kwargs.get('Compton_Xray_flag', False)
    dynamic_final_flag = kwargs.get('dynamic_final_flag', False)
    final_time = kwargs.get('final_time', 50.0)
    data_file = kwargs.get('data_file', "CloudyData_UVB=HM2012.h5")

  
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
    if UVB_flag:
        my_chemistry.UVbackground = 1
    
    my_chemistry.self_shielding_method = 0
    my_chemistry.H2_self_shielding = 0
    my_chemistry.grackle_data_file = \
        os.path.join(grackle_data_dir, data_file)

    my_chemistry.use_specific_heating_rate = 1
    my_chemistry.use_volumetric_heating_rate = 1

    # Set units
    my_chemistry.comoving_coordinates = 0 # proper units
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1.0 / (1.0 + redshift) / \
        my_chemistry.a_units
    my_chemistry.density_units = mass_hydrogen_cgs # rho = 1.0 is 1.67e-24 g
    my_chemistry.length_units = cm_per_mpc         # 1 Mpc in cm
    my_chemistry.time_units = sec_per_Myr 
    my_chemistry.set_velocity_units()
    my_chemistry.Compton_xray_heating = 0
    if Compton_Xray_flag:
        my_chemistry.Compton_xray_heating = 1
    
    if redshift <= 6:
        state = "ionized"
    else:
        state = "neutral"

    density = nH * mass_hydrogen_cgs
    
    if gas_metallicity == 0 or np.log10(gas_metallicity)<-6:
        gas_metallicity = 1e-6  #valid range minimum value
    #metallicity = 0.0 # Solar   #assume primordial gas
    metal_mass_fraction = gas_metallicity * my_chemistry.SolarMetalFractionByMass
    #(SolarMetalFractionByMass: 0.01295)
    #metal_mass_fraction = gas_metallicity
    
    # Call convenience function for setting up a fluid container.
    # This container holds the solver parameters, units, and fields.
    
    #temperature = np.logspace(1, 9, 200)
    max_iterations = 10000
    
    fc = setup_fluid_container(
        my_chemistry,
        density=density,
        f_H2=f_H2,
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
        # Check if it's a constant value (float/int)
        if isinstance(volumetric_heating_rate, (float, int, np.number)):
            fc["volumetric_heating_rate"][:] = volumetric_heating_rate
            heating_function = None
            heating_data = None
        # Check if it's a function
        elif callable(volumetric_heating_rate):
            # Just set initial value, the function will be passed to evolve_constant_density_dynamic_tfinal
            heating_function = volumetric_heating_rate
            heating_data = None
        # Check if it's data for interpolation (tuple of arrays)
        elif isinstance(volumetric_heating_rate, tuple) and len(volumetric_heating_rate) == 2:
            # Set initial value from the first point in the data
            heating_function = None
            heating_data = volumetric_heating_rate
        else:
            raise ValueError("volumetric_heating_rate must be a float, callable, or tuple of (times, rates)")
    
    if evolve_cooling:
        if dynamic_final_flag == False:
            data = evolve_constant_density(
            fc, final_time=final_time,
            safety_factor=0.01)
        else:
            # Use the new evolve_constant_density_dynamic_tfinal with time-dependent heating
            data = evolve_constant_density_with_heating(
                fc, 
                final_temperature=None,
                final_time=final_time, 
                safety_factor=0.01, 
                convergence_check_interval=50,
                heating_function=heating_function,
                heating_data=heating_data)

    else:    
        # get data arrays with symbolic units
        data = fc.finalize_data()
    my_chemistry.__del__()
    
    return data


def plot_cooling_curve(output_dir, redshift, metallicity_Zsun, f_H2):
    evolve_cooling = False #equilibrium cooling rate
    UVB_flag = False
    Compton_Xray_flag = False
    dynamic_final_flag = False

    #debug: LWbackground_intensity = ?

    temperature = np.logspace(2.0, 9.0, 200)
    #also compare with Dekel08 
    temperature_Dekel08 = np.logspace(5.0, 9.0, 100)
    T6_Dekel08 = temperature_Dekel08 / 1.0e6
    Lambda23_Dekel08 = 6.0*(metallicity_Zsun/0.3)**0.7 * T6_Dekel08**(-1) + 0.2*T6_Dekel08**(1/2)  #Lambda in 1e-23 erg/s*cm^3

    lognH_list = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])

    specific_heating_rate = 0.0
    volumetric_heating_rate = 0.0
    gas_metallicity = metallicity_Zsun
    data_alldensity = []
    for lognH in lognH_list:
        data = run_constdensity_model(evolve_cooling,redshift,lognH,specific_heating_rate, 
                volumetric_heating_rate, temperature, gas_metallicity, f_H2=f_H2,
                UVB_flag=UVB_flag, Compton_Xray_flag=Compton_Xray_flag, dynamic_final_flag=dynamic_final_flag)
        data_alldensity.append(data)

    fig, ax = plt.subplots(figsize=(8,6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(lognH_list)))
    for i, data in enumerate(data_alldensity):
        cooling_rate = data["cooling_rate"].v
        #use dashed line for net heating
        neg_mask = (cooling_rate <= 0)
        pos_mask = (cooling_rate > 0)
        ax.plot(data["temperature"].v[neg_mask], -cooling_rate[neg_mask], color=colors[i], label=f'log(nH)={lognH_list[i]}')
        ax.plot(data["temperature"].v[pos_mask], cooling_rate[pos_mask], color=colors[i], linestyle='dashed')

    #plot Dekel08 cooling rate
    ax.plot(temperature_Dekel08, Lambda23_Dekel08*1.0e-23, color='black', linestyle='dotted', label='Dekel+08')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Temperature [K]', fontsize=14)
    ax.set_ylabel(r'Cooling rate $\Lambda$/$n_H^2$ [erg cm$^3$/s]', fontsize=14)
    ax.set_title(f'Cooling rate at z={redshift}, Z={metallicity_Zsun}Z$_\odot$', fontsize=16)
    ax.legend()
    ax.set_ylim(1e-29, 1e-20)
    # ax.set_ylim(1e-27, 1e-21)
    ax.tick_params(axis='both', direction='in')
    plt.tight_layout()
    filename_ext = ''
    if UVB_flag:
        filename_ext += '_UVB'
    if Compton_Xray_flag:
        filename_ext += '_ComptonX'
    if f_H2 > 0:
        filename_ext += f'_fH2_{f_H2:.1e}'
    
    filename = f'Cooling_rate_z{redshift}_Z{metallicity_Zsun:.1e}Zsun{filename_ext}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300)

"""
def test_coolingcell():
    UVB_flag = False
    Compton_Xray_flag = False
    dynamic_final_flag = True

    redshift = 15.0
    lognH = -1.0
    nH = 10**lognH
    specific_heating_rate = 0.0
    volumetric_heating_rate = 0.0
    temperature = 1.0e3
    gas_metallicity = 1.0e-6

    cooling_Eq = run_constdensity_model(False,redshift,lognH,specific_heating_rate, volumetric_heating_rate, temperature, gas_metallicity, 
                             UVB_flag=UVB_flag, Compton_Xray_flag=Compton_Xray_flag, dynamic_final_flag=dynamic_final_flag)

    print("Equilibrium cooling rate:")
    print(cooling_Eq["cooling_rate"].v)

    heating = -cooling_Eq["cooling_rate"].v[0]
    volumetric_heating_rate = 1.0* heating * nH**2

    cooling_NonEq = run_constdensity_model(True,redshift,lognH,specific_heating_rate, volumetric_heating_rate, temperature, gas_metallicity,
                             UVB_flag=UVB_flag, Compton_Xray_flag=Compton_Xray_flag, dynamic_final_flag=dynamic_final_flag)

    print("time = ", cooling_NonEq["time"].v)
    print("Non-equilibrium cooling rate:")
    print(cooling_NonEq["cooling_rate"])
    print("temperature = ", cooling_NonEq["temperature"].v)


def thermal_bremsstrahlung(T):
    T_keV = kB*T/(1.0e3*eV)
    return 7.2*1.0e-24*np.sqrt(T_keV)


def test_TNGhalo_cooling(snapNum):
    base_dir = '/home/zwu/21cm_project/unified_model/TNG_results/'
    processed_file = os.path.join(base_dir, simulation_set, f'snap_{snapNum}', 
                                f'processed_halos_snap_{snapNum}.h5')
    data = load_processed_data(processed_file)
    redshift = data.header['Redshift']
    print(f"Redshift: {redshift}")

    #get basic info: mass, Tvir, DF_heating_without_I, Mach nunber, nH, metallicity
    host_indices = data.subhalo_data['host_index'].value
    host_masses = data.halo_data['GroupMass'].value[host_indices]  # Msun/h
    host_Tvir = data.halo_data['GroupTvir'].value[host_indices]  # K
    host_gasmetallicity = data.halo_data['GroupGasMetallicity'].value[host_indices]  # dimensionless
    host_gasmetallicity_Zsun = host_gasmetallicity / Zsun
    host_t_ff = data.halo_data['Group_t_ff'].value[host_indices]  # s
    host_M200 = data.halo_data['Group_M_Crit200'].value[host_indices]  # Msun/h
    host_R200 = data.halo_data['Group_R_Crit200'].value[host_indices]  # Mpc
    host_lognH = get_gas_lognH_numerical(host_M200/h_Hubble, host_R200)
    host_lognH_analytic = get_gas_lognH_analytic(redshift)

    sub_masses = data.subhalo_data['SubMass'].value  # Msun/h
    mach_numbers = data.subhalo_data['mach_number'].value
    DF_heating_withoutIDF = data.subhalo_data['DF_heating_withoutIDF'].value  # J/s
    rel_vel_mags = data.subhalo_data['relative_velocity_magnitude'].value  # m/s

    print("number of host halos: ", len(np.unique(host_indices)))
    print("number of subhalos: ", len(sub_masses))

    #select test subhalos
    # test group 1: T closest to 1e5 K, Mach ~ 1.5
    mask1 = (host_Tvir > 9.9e4) & (host_Tvir < 1.1e5) & (mach_numbers > 1.4) & (mach_numbers < 1.6)
    test1_indices = np.where(mask1)[0]
    print("number of test halos: ", len(test1_indices))
    print("host halo indices: ", host_indices[test1_indices])
    print("subhalo indices: ", test1_indices)
    print("DF_heating_withoutIDF: ", DF_heating_withoutIDF[test1_indices])
    
    print("------------------------------------------------------------")
    #test group 2 

    mask2 = (host_Tvir > 3.3e5)
    test2_indices = np.where(mask2)[0]
    print("number of test halos: ", len(test2_indices))
    print("host halo indices: ", host_indices[test2_indices])
    print("subhalo indices: ", test2_indices)
    print("DF_heating_withoutIDF: ", DF_heating_withoutIDF[test2_indices])
    print("------------------------------------------------------------")
    
    print("\n\n")
    for sub_index in test2_indices:
        print(f"Test subhalo {sub_index}:")
        print(f"Host halo index: {host_indices[sub_index]}")
        print(f"Host halo mass: {host_masses[sub_index]} Msun/h")
        print(f"subhalo mass: {sub_masses[sub_index]} Msun/h")
        print("mass ratio: ", sub_masses[sub_index]/host_masses[sub_index])
        
        #run cooling model
        lognH = host_lognH[sub_index]
        temperature = host_Tvir[sub_index]
        gas_metallicity = host_gasmetallicity_Zsun[sub_index]
        print(f"lognH = {lognH}, Tvir = {temperature}, Z = {gas_metallicity} Zsun")
        print("thermal bremsstrahlung rate: ", thermal_bremsstrahlung(temperature), "erg cm^3/s")

        specific_heating_rate = 0.0
        volumetric_heating_rate = 0.0
        cooling_Eq = run_constdensity_model(False,redshift,lognH,specific_heating_rate, volumetric_heating_rate, temperature, gas_metallicity)
        print("Equilibrium cooling rate:")
        print(cooling_Eq["cooling_rate"])
        DF_heating_erg = DF_heating_withoutIDF[sub_index] * 1e7  #erg/s
        print("DF_heating_withoutIDF: ", DF_heating_erg, "erg/s")
        wake_volume_cm = (host_R200[sub_index]*Mpc*1.0e2)**3 * np.pi * 4/3
        wake_heating_rate = DF_heating_erg / wake_volume_cm
        print("wake heating rate: ", wake_heating_rate, "erg/cm^3/s")
        normalized_heating_rate = wake_heating_rate / (10**lognH)**2
        print("normalized wake heating rate: ", normalized_heating_rate, "erg cm^3/s")

        print("cooling time = ", cooling_Eq["cooling_time"], "= ", cooling_Eq["cooling_time"].v/Myr, "Myr")
        print("host t_ff = ", host_t_ff[sub_index], "= ", host_t_ff[sub_index]/Myr, "Myr")
        R200_tcross = host_R200[sub_index]*Mpc/ rel_vel_mags[sub_index]
        print("R200/rel_vel = ", R200_tcross, "= ", R200_tcross/Myr, "Myr")

        #now run non-equilibrium cooling model
        # evolve_cooling = True
        # dynamic_final_flag = False
        # UVB_flag = False
        # Compton_Xray_flag = False
        # volumetric_heating_rate = np.abs(cooling_Eq["cooling_rate"][0]*1.1)
        # cooling_NonEq = run_constdensity_model(evolve_cooling,redshift,lognH,specific_heating_rate, volumetric_heating_rate, temperature, gas_metallicity,
        #                      UVB_flag=UVB_flag, Compton_Xray_flag=Compton_Xray_flag, dynamic_final_flag=dynamic_final_flag)

        # print("time:")
        # print(cooling_NonEq["time"])
        # print("Non-equilibrium cooling rate:")
        # print(cooling_NonEq["cooling_rate"])
        print("\n\n")

        # break
"""
        

if __name__ == "__main__":
    
    redshift = 2
    metallicity_Zsun = 0.3*10**(-0.17*redshift)  # Dekel & Birnboim (2006)
    f_H2 = 0.0
    output_dir = './Grackle_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_cooling_curve(output_dir, redshift, metallicity_Zsun, f_H2)
    
    # snapNum = 2
    # test_TNGhalo_cooling(snapNum)