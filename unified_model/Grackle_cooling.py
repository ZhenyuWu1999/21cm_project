
import numpy as np
import os
import matplotlib.pyplot as plt

from pygrackle.utilities.physical_constants import sec_per_Myr, cm_per_mpc, mass_hydrogen_cgs
from pygrackle.utilities.data_path import grackle_data_dir
from pygrackle import \
    chemistry_data, \
    setup_fluid_container    
    
from physical_constants import kB, eV, Zsun, Mpc, Myr, h_Hubble, hydrogen_mass_fraction
from Grackle_evolve import *
from Config import simulation_set

#for debugging
from pygrackle.fluid_container import FluidContainer
from pygrackle.utilities.convenience import check_convergence
from scipy.interpolate import interp1d
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

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

    density = nH * mass_hydrogen_cgs/hydrogen_mass_fraction
    
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


def setup_fluid_container_debug(
    my_chemistry,
    density=mass_hydrogen_cgs,
    f_H2=0.0,
    temperature=None,
    state="neutral",
    metal_mass_fraction=None,
    dust_to_gas_ratio=None,
    converge=False,
    tolerance=0.01,
    max_iterations=10000,
):
    rval = my_chemistry.initialize()
    if rval == 0:
        raise RuntimeError("Failed to initialize chemistry_data.")

    tiny_number = 1e-20
    if metal_mass_fraction is None:
        metal_mass_fraction = tiny_number
    if dust_to_gas_ratio is None:
        dust_to_gas_ratio = tiny_number
    if temperature is None:
        temperature = np.logspace(4, 9, 200)
    else:
        if not isinstance(temperature, np.ndarray):
            temperature = np.array([temperature])
    n_points = temperature.size

    fc = FluidContainer(my_chemistry, n_points)

    fh = my_chemistry.HydrogenFractionByMass
    d2h = my_chemistry.DeuteriumToHydrogenRatio
    metal_free = 1 - metal_mass_fraction
    H_total = fh * metal_free
    He_total = (1 - fh) * metal_free
    D_total = H_total * d2h

    fc_density = density / my_chemistry.density_units
    tiny_density = tiny_number * fc_density

    state_vals = {
        "density": fc_density,
        "metal_density": metal_mass_fraction * fc_density,
        "dust_density": dust_to_gas_ratio * fc_density
    }
    if state == "neutral":
        state_vals["HI_density"] = H_total * fc_density
        state_vals["HeI_density"] = He_total * fc_density
        state_vals["DI_density"] = D_total * fc_density
    elif state == "ionized":
        state_vals["HII_density"] = H_total * fc_density
        state_vals["HeIII_density"] = He_total * fc_density
        state_vals["DII_density"] = D_total * fc_density
        state_vals["e_density"] = state_vals["HII_density"] + state_vals["HeIII_density"] / 2
    else:
        raise ValueError("State must be either neutral or ionized.")

    state_vals["H2I_density"] = f_H2 * fc_density

    for field in fc.density_fields:
        fc[field][:] = state_vals.get(field, tiny_density)

    fc.calculate_mean_molecular_weight()
    fc["internal_energy"] = temperature / fc.chemistry_data.temperature_units / fc["mean_molecular_weight"] / (my_chemistry.Gamma - 1.0)
    fc["x_velocity"][:] = 0.0
    fc["y_velocity"][:] = 0.0
    fc["z_velocity"][:] = 0.0

    # ===== extract k_table =====
    kUnit = mass_hydrogen_cgs / (my_chemistry.density_units * my_chemistry.time_units)
    kUnit_3Bdy = kUnit * mass_hydrogen_cgs / my_chemistry.density_units
    NumberOfTemperatureBins = 600
    T_for_ktable = np.logspace(0, 9, NumberOfTemperatureBins)
    lnT_for_ktable = np.log(T_for_ktable)
    k_table = {
        "k7":  fc.chemistry_data.k7  * kUnit,
        "k8":  fc.chemistry_data.k8  * kUnit,
        "k9":  fc.chemistry_data.k9  * kUnit,
        "k10": fc.chemistry_data.k10 * kUnit,
        "k12": fc.chemistry_data.k12 * kUnit,
        "k13": fc.chemistry_data.k13 * kUnit,
        "k13dd": fc.chemistry_data.k13dd,
        "k21": fc.chemistry_data.k21 * kUnit_3Bdy,
        "T_for_ktable": T_for_ktable,
        "lnT_for_ktable": lnT_for_ktable,
        "kUnit": kUnit,
    }

    fc_last = fc.copy()
    val = fc.chemistry_data.with_radiative_cooling
    fc.chemistry_data.with_radiative_cooling = 0

    my_time = 0.0
    i = 0
    data = defaultdict(list)
    while converge and i < max_iterations:
        fc.calculate_cooling_time()
        dt = 0.01 * np.abs(fc["cooling_time"]).min()
        for field in fc.density_fields:
            fc_last[field] = np.copy(fc[field])
        fc.solve_chemistry(dt)
        fc.calculate_mean_molecular_weight()
        fc["internal_energy"] = temperature / \
            fc.chemistry_data.temperature_units / \
            fc["mean_molecular_weight"] / (my_chemistry.Gamma - 1.0)
        add_to_data(fc, data, extra={"time": my_time})
        if check_convergence(fc, fc_last, tol=tolerance):
            break
        my_time += dt
        i += 1

    fc.chemistry_data.with_radiative_cooling = val
    if i >= max_iterations:
        raise RuntimeError(
            f"ERROR: solver did not converge in {max_iterations} iterations.")

    for field in data:
        data[field] = np.squeeze(np.array(data[field]))

    for field in data:
        data[field] = np.squeeze(np.array(data[field]))
    converge_data = fc.finalize_data(data=data)
    return fc, k_table, converge_data



def run_constdensity_debug():
    from Grackle_Omukai import make_k13dd_interpolators, k13_density_dependent

    redshift                = 15
    # lognH:  -0.8071642449378659
    # nH:  0.1558962810384279
    lognH                   = -0.8
    specific_heating_rate   = 0.0
    volumetric_heating_rate = 0.0
    temperature             = np.array([3000,3500,4000,4500,5000,6000,7000,8000])
    gas_metallicity         = 1.0e-8
    f_H2                    = 1.0e-4

    evolve_cooling          = False
    UVB_flag          = False
    Compton_Xray_flag = False
    data_file         = "CloudyData_UVB=HM2012.h5"
    converge_when_setup = True

      
    tiny_number = 1e-20
    if f_H2 == 0.0:
        f_H2 = tiny_number
    
    nH = 10**lognH 
    
    # dictionary to store extra information in output dataset
    extra_attrs = {}

    # Set solver parameters
    my_chemistry = chemistry_data()
    my_chemistry.use_grackle = 1
    
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

    density = nH * mass_hydrogen_cgs/hydrogen_mass_fraction
    
    if gas_metallicity == 0 or np.log10(gas_metallicity)<-8:
        gas_metallicity = 1.0e-8  #cloudy_metals_2008_3D.h5 valid range>1e-6; not important for other files
    #metallicity = 0.0 # Solar   #assume primordial gas
    metal_mass_fraction = gas_metallicity * my_chemistry.SolarMetalFractionByMass


    fc, k_table, converge_data = setup_fluid_container_debug(
        my_chemistry, density=density, f_H2=f_H2,
        temperature=temperature, state=state,
        metal_mass_fraction=metal_mass_fraction,
        dust_to_gas_ratio=None, converge=converge_when_setup,
        tolerance=0.01, max_iterations=10000)

    # print("Debugging information:")
    # print(fc)
    # print("k_table keys:", k_table.keys())
    # print("Converged data keys:", converge_data.keys())
    # print("Converged data temperature:", converge_data["temperature"])
    # print("Converged data cooling_rate:", converge_data["cooling_rate"])
    # print("Converged data cooling_time:", converge_data["cooling_time"])
    # print("Converged data time:", converge_data["time"])
    # print("y_H2I:", converge_data["H2I_density"] / converge_data["density"])
    # print("shape of y_H2I:", (converge_data["H2I_density"] / converge_data["density"]).shape)
    # print("shape of time:", converge_data["time"].shape)
    # print("shape of temperature:", converge_data["temperature"].shape)


        # ===== post-processing: interpolate rates =====

    lnT_ktable = k_table["lnT_for_ktable"]
    lnT_input = np.log(temperature)
    kUnit = k_table["kUnit"]


    time_arr = converge_data["time"]   # (n_iter,)
    time_Myr = time_arr.in_units("Myr").v

    nH_hist = converge_data["HI_density"].v / mass_hydrogen_cgs     # (n_iter, nT)
    ne_hist = converge_data["e_density"].v / mass_hydrogen_cgs      # (n_iter, nT)
    ye_hist = ne_hist / nH_hist
    y_H2_hist = converge_data["H2I_density"].v / (2.0 * converge_data["HI_density"].v)

    # interpolate rates at input T-grid (shape: nT)
    k7  = interp1d(lnT_ktable, k_table["k7"],  kind='cubic')(lnT_input)
    k8  = interp1d(lnT_ktable, k_table["k8"],  kind='cubic')(lnT_input)
    k9  = interp1d(lnT_ktable, k_table["k9"],  kind='cubic')(lnT_input)
    k12 = interp1d(lnT_ktable, k_table["k12"], kind='cubic')(lnT_input)
    k13 = interp1d(lnT_ktable, k_table["k13"], kind='cubic')(lnT_input)

    # No LW in this debug run
    k27 = 0.0

    # formation and dissociation terms (all ~ 1/s after multiplying by densities)
    kform_HM_eff = k7 * k8 * nH_hist / (k8 * nH_hist + k27)   # (n_iter, nT)
    kform_H2II_eff = k9                                        # (nT,)
    kform_total = kform_HM_eff + kform_H2II_eff               # broadcast -> (n_iter, nT)

    R_form = kform_total * ye_hist * nH_hist                  # (n_iter, nT)
    R_k12  = k12 * ne_hist                                    # (n_iter, nT)
    R_k13  = k13 * nH_hist                                    # (n_iter, nT)
    R_diss = R_k12 + R_k13                                    # (n_iter, nT)

    tiny = 1e-80
    y_H2_analytic = R_form / np.maximum(R_diss, tiny)

    output_dir = '/home/zwu/21cm_project/unified_model/Grackle_results'
    cmap = plt.cm.coolwarm
    norm = Normalize(vmin=temperature.min(), vmax=temperature.max())

    # sparse sampling for markers
    step = max(1, len(time_Myr) // 15)
    idx = np.arange(0, len(time_Myr), step)

    fig, ax = plt.subplots(figsize=(10, 7))

    for j, T in enumerate(temperature):
        color = cmap(norm(T))

        # Grackle result: solid line
        ax.plot(time_Myr, y_H2_hist[:, j], '-', color=color, lw=1.6)

        # Analytic (full): circle markers
        ax.scatter(
            time_Myr[idx], y_H2_analytic[idx, j],
            marker='o', facecolors='none', edgecolors=color, s=30, zorder=5
        )

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Temperature [K]', fontsize=14)

    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, label='Grackle'),
        Line2D([0], [0], marker='o', color='gray', markerfacecolor='none',
       lw=0, markersize=7, label=r'Estimated equilibrium: $y_{\mathrm{H_2,eq}}=\frac{k_{\mathrm{form}}\,y_e\,n_{\mathrm{H}}}{k_{12}n_e+k_{13}n_{\mathrm{H}}}$')

    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='lower right')

    ax.set_yscale('log')
    ax.set_xlabel('Time [Myr]', fontsize=14)
    ax.set_ylabel(r'$n_{\rm H_2}/n_{\rm H}$', fontsize=14)
    ax.set_title(f'H$_2$ convergence at fixed nH (log nH = {lognH})', fontsize=15)
    ax.tick_params(which='both', direction='in', labelsize=12)

    plt.tight_layout()
    filename = f'debug_H2_convergence_lognH_{lognH}_initialfH2_{f_H2}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    print(f"Debug plot saved to {os.path.join(output_dir, filename)}")

    B_term = R_diss  # shape: (n_iter, nT)
    tiny = 1e-80
    tau_chem_Myr_all = 1.0 / np.maximum(B_term, tiny) / sec_per_Myr  # (n_iter, nT)

    print("\n=== tau_chem = 1/B for all iterations ===")
    for i, t_myr in enumerate(time_Myr):
        print(f"\n-- iter={i:4d}, time={t_myr:.6e} Myr --")
        for j, T in enumerate(temperature):
            print(
                f"T = {T:7.1f} K: "
                f"B = {B_term[i, j]:.3e} 1/s, "
                f"tau_chem = {tau_chem_Myr_all[i, j]:.3e} Myr"
            )

   

def plot_cooling_curve(output_dir, redshift, metallicity_Zsun, f_H2, converge_when_setup=True):
    evolve_cooling = False #equilibrium cooling rate
    UVB_flag = False
    Compton_Xray_flag = False
    dynamic_final_flag = False

    #debug: LWbackground_intensity = ?

    temperature = np.logspace(2.8, 4.5, 200)
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
                converge_when_setup = converge_when_setup)
        data_alldensity.append(data)


    #debug
    print("Debugging information:")
    selected_data = data_alldensity[2]  # e.g., for lognH= -1.0
    selected_lognH = lognH_list[2]
    selected_nH = 10**selected_lognH
    cooling_rate = selected_data["cooling_rate"].v
    temperature_vals = selected_data["temperature"].v
    cooling_timescale = selected_data["cooling_time"].v
    mean_molecular_weight = selected_data["mean_molecular_weight"].v
    for i in range(len(temperature_vals)):
        T = temperature_vals[i]
        Lambda = cooling_rate[i]
        tcool = cooling_timescale[i]
        mu = mean_molecular_weight[i]
        print(f"T={T:.2e} K, Cooling rate={Lambda:.2e} erg/cm^3/s, Cooling time={tcool:.2e} s")

        kB_cgs = 1.380649e-16  # Boltzmann constant in erg/K
        tcool_test =  1.5* kB_cgs * T / (Lambda * (selected_nH/0.76) * mu)
        print(f"  Test cooling time calculation: tcool={tcool_test:.2e} s")
        print(f"  Ratio of calculated to reported cooling time: {tcool_test/tcool:.2f}")

    

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
    ax.set_ylim(1e-31, 1e-20)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
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
    if converge_when_setup == False:
        filename_ext += '_NoConverge'
    
    filename = f'Cooling_rate_z{redshift}_Z{metallicity_Zsun:.1e}Zsun{filename_ext}.png'
    filename = os.path.join(output_dir, filename)
    plt.savefig(filename, dpi=300)
    print(f"Cooling curve plot saved to {filename}")

    
    #also plot the final H2 fraction  (data["H2I_density"]/data["density"])
    fig, ax2 = plt.subplots(figsize=(8,6))
    for i, data in enumerate(data_alldensity):
        H2_fraction = data["H2I_density"].v / data["density"].v
        ax2.plot(data["temperature"].v, H2_fraction, color=colors[i], label=f'log(nH)={lognH_list[i]}')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Temperature [K]', fontsize=14)
    ax2.set_ylabel('H2 fraction', fontsize=14)
    ax2.set_ylim(bottom=1e-8, top=1.0e-2)
    ax2.set_title(f'final H2 fraction at z={redshift} (Z={metallicity_Zsun}Z$_\odot$, initial H2 fraction={f_H2})', fontsize=16)
    ax2.legend()
    ax2.tick_params(which='both', direction='in', labelsize=12)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    filename = f'final_H2_fraction_z{redshift}_Z{metallicity_Zsun:.1e}Zsun{filename_ext}.png'
    filename = os.path.join(output_dir, filename)
    plt.savefig(filename, dpi=300)
    print(f"Plots saved to {filename}")
        

if __name__ == "__main__":
    
    # redshift = 2
    # metallicity_Zsun = 0.3*10**(-0.17*redshift)  # Dekel & Birnboim (2006)
    # redshift = 15
    # metallicity_Zsun = 1.0e-8
    # f_H2 = 1.0e-3
    # output_dir = '/home/zwu/21cm_project/unified_model/Grackle_results'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # plot_cooling_curve(output_dir, redshift, metallicity_Zsun, f_H2, converge_when_setup=True)
    
    run_constdensity_debug()