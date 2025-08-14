
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

def run_constdensity_model(params: dict, **kwargs):
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
    
    required = [
        "evolve_cooling", "redshift", "lognH",
        "specific_heating_rate", "volumetric_heating_rate",
        "temperature", "gas_metallicity", "f_H2"
    ]
    missing = [k for k in required if k not in params]
    if missing:
        raise ValueError(f"Missing required params: {missing}")

    evolve_cooling          = params["evolve_cooling"]
    redshift                = params["redshift"]
    lognH                   = params["lognH"]
    specific_heating_rate   = params["specific_heating_rate"]
    volumetric_heating_rate = params["volumetric_heating_rate"]
    temperature             = params["temperature"]
    gas_metallicity         = params["gas_metallicity"]
    f_H2                    = params["f_H2"]

    DEFAULTS = {
    "UVB_flag": True,
    "Compton_Xray_flag": False,
    "dynamic_final_flag": False,
    "final_time": 50.0,
    "data_file": "CloudyData_UVB=HM2012.h5",
    "converge_when_setup": True,
    }

    opts = {**DEFAULTS, **kwargs}
    UVB_flag          = opts["UVB_flag"]
    Compton_Xray_flag = opts["Compton_Xray_flag"]
    dynamic_final_flag = opts["dynamic_final_flag"]
    final_time        = opts["final_time"]
    data_file         = opts["data_file"]
    converge_when_setup = opts["converge_when_setup"]

  
    tiny_number = 1e-20
    if f_H2 == 0.0:
        f_H2 = tiny_number
    
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
    
    if redshift <= 8:
        state = "ionized"
    else:
        state = "neutral"

    density = nH * mass_hydrogen_cgs
    
    if gas_metallicity == 0 or np.log10(gas_metallicity)<-8:
        gas_metallicity = 1.0e-8  #cloudy_metals_2008_3D.h5 valid range>1e-6; not important for other files
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
        converge=converge_when_setup,
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
        params_for_constdensity = {
            "evolve_cooling": evolve_cooling,
            "redshift": redshift,
            "lognH": lognH,
            "specific_heating_rate": specific_heating_rate,
            "volumetric_heating_rate": volumetric_heating_rate,
            "temperature": temperature,
            "gas_metallicity": gas_metallicity,
            "f_H2": f_H2
        }

        data = run_constdensity_model(params_for_constdensity,
                UVB_flag=UVB_flag, Compton_Xray_flag=Compton_Xray_flag, dynamic_final_flag=dynamic_final_flag,
                converge_when_setup=True)
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
    # ax.plot(temperature_Dekel08, Lambda23_Dekel08*1.0e-23, color='black', linestyle='dotted', label='Dekel+08')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Temperature [K]', fontsize=14)
    ax.set_ylabel(r'Cooling rate $\Lambda$/$n_H^2$ [erg cm$^3$/s]', fontsize=14)
    ax.set_title(f'Cooling rate at z={redshift} (Z={metallicity_Zsun}Z$_\odot$, H2 fraction={f_H2})', fontsize=16)
    ax.legend()
    ax.set_ylim(1e-29, 1e-20)
    # ax.set_ylim(1e-27, 1e-21)
    ax.tick_params(which='both', direction='in', labelsize=12)
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


    
    #also plot the final H2 fraction  (data["H2I_density"]/data["density"])
    fig, ax2 = plt.subplots(figsize=(8,6))
    for i, data in enumerate(data_alldensity):
        H2_fraction = data["H2I_density"].v / data["density"].v
        ax2.plot(data["temperature"].v, H2_fraction, color=colors[i], label=f'log(nH)={lognH_list[i]}')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Temperature [K]', fontsize=14)
    ax2.set_ylabel('H2 fraction', fontsize=14)
    ax2.set_title(f'final H2 fraction at z={redshift} (Z={metallicity_Zsun}Z$_\odot$, initial H2 fraction={f_H2})', fontsize=16)
    ax2.legend()
    ax2.tick_params(which='both', direction='in', labelsize=12)
    plt.tight_layout()
    filename = f'final_H2_fraction_z{redshift}_Z{metallicity_Zsun:.1e}Zsun{filename_ext}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300)

        

if __name__ == "__main__":
    
    # redshift = 2
    # metallicity_Zsun = 0.3*10**(-0.17*redshift)  # Dekel & Birnboim (2006)
    redshift = 15
    metallicity_Zsun = 1.0e-6
    f_H2 = 0.0
    output_dir = '/home/zwu/21cm_project/unified_model/debug'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    plot_cooling_curve(output_dir, redshift, metallicity_Zsun, f_H2)
    