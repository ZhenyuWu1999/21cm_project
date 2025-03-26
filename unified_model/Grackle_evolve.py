########################################################################
#
# Functions for evolving a fluid container using Grackle
#
#
# Copyright (c) 2016, Grackle Development Team.
#
# Distributed under the terms of the Enzo Public Licence.
#
# The full license is in the file LICENSE, distributed with this
# software.
########################################################################

from collections import defaultdict
import numpy as np
from scipy.interpolate import interp1d


from pygrackle.utilities.physical_constants import gravitational_constant_cgs, sec_per_year

def evolve_freefall(fc, final_density, safety_factor=0.01,
                    include_pressure=True):
    my_chemistry = fc.chemistry_data

    # Set units of gravitational constant
    gravitational_constant = (
        4.0 * np.pi * gravitational_constant_cgs *
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

        # use this to multiply by elemental densities if you are tracking those
        density_ratio = new_density / fc["density"][0]

        # update densities
        for field in fc.density_fields:
            fc[field] *= density_ratio

        # now update energy for adiabatic heating from collapse
        fc["internal_energy"][0] += (my_chemistry.Gamma - 1.) * \
          fc["internal_energy"][0] * freefall_time_constant * \
          np.power(fc["density"][0], 0.5) * dt

        fc.solve_chemistry(dt)

        # update time
        current_time += dt

    for field in data:
        data[field] = np.squeeze(np.array(data[field]))
    return fc.finalize_data(data=data)

def calculate_collapse_factor(pressure, density):
    # Calculate the effective adiabatic index, dlog(p)/dlog(rho).

    if len(pressure) < 3:
        return np.array([0.])

    # compute dlog(p) / dlog(rho) using last two timesteps
    gamma_eff = np.log10(pressure[-1] / pressure[-2]) / \
        np.log10(density[-1] / density[-2])

    # compute a higher order derivative if more than two points available
    if len(pressure) > 2:
        gamma_eff += 0.5 * ((np.log10(pressure[-2] / pressure[-3]) /
                             np.log10(density[-2] / density[-3])) - gamma_eff)

    gamma_eff = np.clip(gamma_eff, a_min=0, a_max=4/3)

    # Equation 9 of Omukai et al. (2005)
    if gamma_eff < 0.83:
        force_factor = gamma_eff * 0
    elif gamma_eff < 1.0:
        force_factor = 0.6 + 2.5 * (gamma_eff - 1) - \
            6.0 * np.power((gamma_eff - 1.0), 2.)
    else:
        force_factor = 1.0 + 0.2 * (gamma_eff - (4./3.)) - \
            2.9 * np.power((gamma_eff - (4./3.)), 2.)
    force_factor = np.clip(force_factor, a_min=0, a_max=0.95)
    return force_factor

def evolve_constant_density(fc, final_temperature=None,
                            final_time=None, safety_factor=0.01):
    my_chemistry = fc.chemistry_data

    if final_temperature is None and final_time is None:
        raise RuntimeError("Must specify either final_temperature " +
                           "or final_time.")

    data = defaultdict(list)
    current_time = 0.0
    fc.calculate_cooling_time()
    dt = safety_factor * np.abs(fc["cooling_time"][0])
    fc.calculate_temperature()
    while True:
        if final_temperature is not None and fc["temperature"][0] <= final_temperature:
            break
        if final_time is not None and current_time >= final_time:
            break

        fc.calculate_temperature()
        print("Evolve constant density - t: %e yr, rho: %e g/cm^3, T: %e K." %
              (current_time * my_chemistry.time_units / sec_per_year,
               fc["density"][0] * my_chemistry.density_units,
               fc["temperature"][0]))
        fc.solve_chemistry(dt)
        add_to_data(fc, data, extra={"time": current_time})
        current_time += dt

    for field in data:
        data[field] = np.squeeze(np.array(data[field]))
    return fc.finalize_data(data=data)

def evolve_constant_density_with_heating(fc, final_temperature=None, final_time=None, 
                                          safety_factor=0.01, convergence_check_interval=50,
                                          heating_function=None, heating_data=None):
    """
    Evolve a constant density gas with time-dependent heating.
    
    Parameters:
    -----------
    fc : object
        The chemistry object that handles all chemical calculations
    final_temperature : float, optional
        Target temperature to stop the evolution
    final_time : float, optional
        Target time to stop the evolution
    safety_factor : float, default=0.01
        Factor to multiply with cooling time for timestep determination
    convergence_check_interval : float, default=50
        Interval for checking if temperature has converged
    heating_function : callable, optional
        Function that takes (time_in_code_units) and returns volumetric heating rate
    heating_data : tuple, optional
        Tuple of (times, heating_rates) arrays for interpolation
    """
    my_chemistry = fc.chemistry_data

    if final_temperature is None and final_time is None:
        raise RuntimeError("Must specify either final_temperature or final_time.")
    
    # Set up interpolation if heating_data is provided
    if heating_data is not None:
        times, heating_rates = heating_data
        heating_interpolator = interp1d(times, heating_rates, 
                                       bounds_error=False, fill_value=(heating_rates[0], heating_rates[-1]))

    data = defaultdict(list)
    current_time = 0.0
    next_convergence_check = 100.0

    while True:
        # Check termination conditions
        if final_temperature is not None and fc["temperature"][0] <= final_temperature:
            break
        if final_time is not None and current_time >= final_time:
            break
        if current_time >= next_convergence_check:
            if len(data["temperature"]) > 5:
                last_temperatures = np.squeeze(np.array(data["temperature"][-5:])) 
                if max(last_temperatures) / min(last_temperatures) - 1 < 0.01:
                    print("Temperature has stabilized.")
                    break
            next_convergence_check += convergence_check_interval

        # Update heating based on current time
        if heating_function is not None:
            # Use the provided function to get heating rate at current time
            fc["volumetric_heating_rate"][:] = heating_function(current_time)
        elif heating_data is not None:
            # Use interpolation to get heating rate at current time
            fc["volumetric_heating_rate"][:] = heating_interpolator(current_time)

        # Calculate next timestep
        fc.calculate_cooling_time()
        dt = safety_factor * np.abs(fc["cooling_time"][0])
        dt = min(dt, 10)
        
        # Evolve the system
        fc.calculate_temperature()
        current_time_yr = current_time * my_chemistry.time_units / sec_per_year
        print("Evolve constant density - t: %e yr, rho: %e g/cm^3, T: %e K." %
              (current_time_yr, fc["density"][0] * my_chemistry.density_units,
               fc["temperature"][0]))
        print("current_time: ", current_time)
        
        fc.solve_chemistry(dt)
        add_to_data(fc, data, extra={"time": current_time})
        current_time += dt

    # Convert lists to numpy arrays
    for field in data:
        data[field] = np.squeeze(np.array(data[field]))
    return fc.finalize_data(data=data)



def evolve_constant_density_dynamic_tfinal(fc, final_temperature=None,
                            final_time=None, safety_factor=0.01, convergence_check_interval=50):
    my_chemistry = fc.chemistry_data


    if final_temperature is None and final_time is None:
        raise RuntimeError("Must specify either final_temperature " +
                           "or final_time.")

    data = defaultdict(list)
    current_time = 0.0
    next_convergence_check = 100.0
    # fc.calculate_cooling_time()
    # initial_dt = safety_factor * np.abs(fc["cooling_time"][0])
    # fc.calculate_temperature()
    while True:

        if final_temperature is not None and fc["temperature"][0] <= final_temperature:
            break
        if final_time is not None and current_time >= final_time:
            break
        if current_time >= next_convergence_check:
            if len(data["temperature"]) > 5:
                last_temperatures = np.squeeze(np.array(data["temperature"][-5:])) 
                if max(last_temperatures) / min(last_temperatures) - 1 < 0.01:
                    print("Temperature has stabilized.")
                    break
            next_convergence_check += convergence_check_interval

        fc.calculate_cooling_time()
        dt = safety_factor * np.abs(fc["cooling_time"][0])
        dt = min(dt, 10)
        
        fc.calculate_temperature()
        print("Evolve constant density - t: %e yr, rho: %e g/cm^3, T: %e K." %
              (current_time * my_chemistry.time_units / sec_per_year,
               fc["density"][0] * my_chemistry.density_units,
               fc["temperature"][0]))
        print("current_time: ", current_time)
        fc.solve_chemistry(dt)
        add_to_data(fc, data, extra={"time": current_time})
        current_time += dt

        current_time_yr = current_time * my_chemistry.time_units / sec_per_year
        # current_time_Myr = current_time_yr / 1e6
        # if current_time_Myr > 10:  # Stop after 10 Myr for debugging
        #     fc["volumetric_heating_rate"][:] = 0.0

    for field in data:
        data[field] = np.squeeze(np.array(data[field]))
    return fc.finalize_data(data=data)

def add_to_data(fc, data, extra=None):
    """
    Add current fluid container values to the data structure.
    """

    for field in fc.all_fields:
        if field not in fc.input_fields:
            func = getattr(fc, f"calculate_{field}")
            if func is None:
                raise RuntimeError(f"No function for calculating {field}.")
            func()
        data[field].append(fc[field].copy())

    if extra is not None:
        for field in extra:
            data[field].append(extra[field])
