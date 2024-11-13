import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from DF_cooling_rate import run_cool_rate
from scipy.optimize import brentq
import datetime
import sys
import logging
from multiprocessing import Process, current_process
from multiprocessing import Pool, Manager

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

# import cProfile
# import pstats


def setup_logging(output_dir):
    process_id = current_process().pid
    logging.basicConfig(filename=output_dir + f'log_{process_id}.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def read_hdf5_data(filepath):
    data_dict = {}

    # Open the HDF5 file in read-only mode
    with h5py.File(filepath, 'r') as f:
        # Function to recursively read data
        def read_recursive(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Store dataset data in dictionary
                data_dict[name] = np.array(obj)
                print(f"Dataset: {name} loaded")
                # Uncomment to print attributes of each dataset
                for key, val in obj.attrs.items():
                    print(f"    {key}: {val}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")
                # Uncomment to print attributes of each group
                for key, val in obj.attrs.items():
                    print(f"    {key}: {val}")
                # Recursively visit items in the group
                for subname, subitem in obj.items():
                    read_recursive(f"{name}/{subname}", subitem)

        # Start the recursive read from the root of the HDF5 file
        f.visititems(read_recursive)

    return data_dict


#equilibrium temperature including DF heating, based on Grackle cooling rate
def get_Grackle_TDF(initial_T, lognH, metallicity, normalized_heating, redshift):
    # Define the net heating function
    def net_DFheating(T, lognH, metallicity, normalized_heating, redshift):
        cooling_data = run_cool_rate(False, redshift, lognH, 0.0, 0.0, T, metallicity)
        cooling_rate = cooling_data["cooling_rate"][0].v.item()
        return cooling_rate + normalized_heating
    
    def net_allheating(T, lognH, metallicity, normalized_heating, cooling_Tinit, redshift):
        additional_heating = - cooling_Tinit
        cooling_data = run_cool_rate(False, redshift, lognH, 0.0, 0.0, T, metallicity)
        cooling_rate = cooling_data["cooling_rate"][0].v.item()
        return cooling_rate + normalized_heating + additional_heating

    net_heating_flag = 0
    net_DFheating_Tinit = net_DFheating(initial_T, lognH, metallicity, normalized_heating, redshift)
    cooling_Tinit = net_DFheating_Tinit - normalized_heating
    
    if net_DFheating_Tinit < 0:
        net_heating_flag = -1
    else:
        net_heating_flag = 1
    
    # Set initial range and tolerances
    T_low = initial_T
    abs_tol = 100  # Absolute tolerance  100 K
    rel_tol = 1e-3  # Relative tolerance
    
    T_tests = np.array([10,100,1000,1e4,1e5])*T_low
    
    T_high_DFheating = None
    T_high_allheating = None
    
    for T_test in T_tests:
        if(net_heating_flag == 1 and T_high_DFheating is None):
            net_DFheating_Ttest = net_DFheating(T_test, lognH, metallicity, normalized_heating, redshift)
            if net_DFheating_Ttest < 0:
                T_high_DFheating = T_test
        
        if T_high_allheating is None:
            net_allheating_Ttest = net_allheating(T_test, lognH, metallicity, normalized_heating, cooling_Tinit, redshift)
            if net_allheating_Ttest < 0:
                T_high_allheating = T_test
    

    if net_heating_flag == 1 and T_high_DFheating is not None:
        T_DF_equilibrium = brentq(net_DFheating, T_low, T_high_DFheating, args=(lognH, metallicity, normalized_heating, redshift), xtol=abs_tol, rtol=rel_tol)
        cooling_data_TDF = run_cool_rate(False, redshift, lognH, 0.0, 0.0, T_DF_equilibrium, metallicity)
        cooling_rate_TDF = cooling_data_TDF["cooling_rate"][0].v.item()
        
    else:
        T_DF_equilibrium = -1
        cooling_rate_TDF = np.nan
    
    if T_high_allheating is not None:
        T_allheating_equilibrium = brentq(net_allheating, T_low, T_high_allheating, args=(lognH, metallicity, normalized_heating, cooling_Tinit, redshift), xtol=abs_tol, rtol=rel_tol)
        cooling_data_Tallheating = run_cool_rate(False, redshift, lognH, 0.0, 0.0, T_allheating_equilibrium, metallicity)
        cooling_rate_Tallheating = cooling_data_Tallheating["cooling_rate"][0].v.item()
    else:
        T_allheating_equilibrium = -1
        cooling_rate_Tallheating = np.nan
    
    return net_heating_flag, T_DF_equilibrium, T_allheating_equilibrium, cooling_Tinit, cooling_rate_TDF, cooling_rate_Tallheating
    
        
def process_subhalo(i, Sub_data, current_redshift):
    logging.info(f"Subhalo {i}")
    Tvir = Sub_data['Tvir_host'][i]
    
    #use host halo gas metallicity because subhalo gas metallicity is often not available
    gas_metallicity_sub = Sub_data['gas_metallicity_sub'][i]
    gas_metallicity_sub /= 0.01295
    gas_metallicity_host = Sub_data['gas_metallicity_host'][i]
    gas_metallicity_host /= 0.01295
    #logging.info("Zsub: ", gas_metallicity_sub, "Zsun")
    #logging.info("Zhost: ", gas_metallicity_host, "Zsun")
    Mach_rel = Sub_data['Mach_rel'][i]
    vel_rel = Sub_data['vel_rel'][i] #m/s
    
    heating = Sub_data['DF_heating'][i]
    heating *= 1e7 #convert to erg/s
    
    rho_g = Sub_data['rho_g'][i]
    overdensity = 1.0
    rho_g_wake = rho_g*(1+overdensity)
    nH = rho_g_wake/mass_hydrogen_kg
    nH_cm3 = nH/1e6
    lognH = np.log10(nH_cm3)
    
    #test which volume to use
    volume_wake_tdyn = Sub_data['Volume_wake'][i]
    volume_wake_tdyn_cm3 = volume_wake_tdyn*1e6
    volumetric_heating = heating/volume_wake_tdyn_cm3 #erg/s/cm^3
    normalized_heating = heating/volume_wake_tdyn_cm3/nH_cm3**2 #erg cm^3 s^-1
    Mgas_wake = volume_wake_tdyn*rho_g_wake
    specific_heating = heating/(Mgas_wake*1e3) #erg/s/g
    evolve_cooling = False
    
    #logging.info(f"\nTvir: {Tvir} K, lognH: {lognH}")
    #logging.info("Normalized heating rate: ", normalized_heating, "erg cm^3/s")

    net_heating_flag, T_DF, T_allheating,cooling_rate_Tvir, cooling_rate_TDF, cooling_rate_Tallheating = get_Grackle_TDF(Tvir, lognH, gas_metallicity_host, normalized_heating, current_redshift)
    
    #logging.info("Cooling rate (zero heating): ", cooling_rate_Tvir, "erg cm^3/s")
    
    subhalowake_info = (Tvir, lognH, rho_g_wake, gas_metallicity_host, volume_wake_tdyn_cm3, heating, specific_heating, volumetric_heating, normalized_heating, net_heating_flag, T_DF, T_allheating, cooling_rate_Tvir, cooling_rate_TDF,cooling_rate_Tallheating, Mach_rel, vel_rel)
    
    return subhalowake_info




#-------------------------------------------------------------------------------------
def main():

    TNG50_redshift_list = [20.05,14.99,11.98,10.98,10.00,9.39,9.00,8.45,8.01]
    snapNum = 8
    output_dir = "/home/zwu/21cm_project/grackle_DF_cooling/snap_"+str(snapNum)+"/"
    filepath = "/home/zwu/21cm_project/compare_TNG/results/TNG50-1/snap_"+str(snapNum)+"/"
    current_redshift = TNG50_redshift_list[snapNum]
    setup_logging(output_dir)
    logging.info("Start time: %s", datetime.datetime.now())
    
    #find the hdf5 file starting with "DF_heating_snap"
    for file in os.listdir(filepath):
        if file.startswith("DF_heating_snap"):
            filename = filepath + file
            break
    data_dict = read_hdf5_data(filename)
    Host_data = data_dict["HostHalo"]
    Sub_data = data_dict["SubHalo"]
    logging.info(f"Data Read from {filename}")
    logging.info(f"Output to {output_dir}")
    
    Model = 'SubhaloWake'
    logging.info(f"Model: {Model}")

   
    #output data type
    dtype_subhalowake = np.dtype([
        ('Tvir', np.float64),
        ('lognH', np.float64),
        ('rho_g_wake', np.float64),
        ('gas_metallicity_host', np.float64),
        ('volume_wake_tdyn_cm3', np.float64),
        ('heating', np.float64),
        ('specific_heating', np.float64),
        ('volumetric_heating', np.float64),
        ('normalized_heating', np.float64),
        ('net_heating_flag', np.int32),
        ('T_DF', np.float64),
        ('T_allheating', np.float64),
        ('cooling_rate_Tvir', np.float64),
        ('cooling_rate_TDF', np.float64),
        ('cooling_rate_Tallheating', np.float64),
        ('Mach_rel', np.float64),
        ('vel_rel', np.float64)
    ])
    
    
    manager = Manager()
    Output_SubhaloWake_Info = manager.list()
    Num_processes = 16
    pool = Pool(processes=Num_processes)

    
    loop_num = len(Sub_data['Tvir_host'])
    logging.info(f"Number of subhalos: {loop_num}")

    for i in range(loop_num):
        pool.apply_async(process_subhalo, args=(i, Sub_data, current_redshift), callback=lambda result: Output_SubhaloWake_Info.append(result))
    
    pool.close()
    pool.join()
    
    #end of loop over subhalo
        
    output_filename = output_dir+f"Grackle_Cooling_SubhaloWake_FullModel_snap"+str(snapNum)+"_testNp8.h5"
    with h5py.File(output_filename, 'w') as file:
        subhalowake_array = np.array(Output_SubhaloWake_Info, dtype=dtype_subhalowake)
        subhalowake_dataset = file.create_dataset('SubhaloWake', data=subhalowake_array)

    logging.info("End time: %s", datetime.datetime.now())
    
#-------------------------------------------------------------------------------------    
    
amu_kg           = 1.660538921e-27  # g
mass_hydrogen_kg = 1.007947*amu_kg  # g
if __name__ == "__main__":
    main()
