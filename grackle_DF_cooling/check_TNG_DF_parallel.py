import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from DF_cooling_rate import run_cool_rate, get_Grackle_TDF, get_Grackle_TDF_nonEquilibrium

import datetime
import sys
import logging
from multiprocessing import Process, current_process
from multiprocessing import shared_memory
#from multiprocessing import Pool, Manager
from multiprocessing import Pool
from multiprocessing import sharedctypes
import ctypes
from collections import namedtuple
Result = namedtuple('Result', ['Tvir', 'lognH', 'rho_g_wake', 'gas_metallicity_host', 'volume_wake_tdyn_cm3', 'heating', 'specific_heating', 'volumetric_heating', 'normalized_heating', 'net_heating_flag', 'T_DF', 'T_allheating', 'cooling_rate_Tvir', 'cooling_rate_TDF', 'cooling_rate_Tallheating', 'Mach_rel', 'vel_rel'])

ResultNonEq = namedtuple('ResultNonEq', ['index', 'Tvir', 'lognH', 'rho_g_wake', 'gas_metallicity_host', 'volume_wake_tdyn_cm3', 'heating', 'specific_heating', 'volumetric_heating', 'normalized_heating', 'tfinal', 'T_DF_NonEq', 'cooling_rate_Tvir', 'cooling_rate_TDF_NonEq', 'Mach_rel', 'vel_rel'])


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
from memory_profiler import profile

# import cProfile
# import pstats


# def setup_logging(output_dir):
#     process_id = current_process().pid
#     logging.basicConfig(filename=output_dir + f'log_{process_id}.log',
#                         level=logging.INFO,
#                         format='%(asctime)s - %(levelname)s - %(message)s')

def setup_logging(output_dir):
    process_id = current_process().pid
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_filename = f"{output_dir}/log_{process_id}_{timestamp}.log"
    log_file = open(log_filename, "a")
    sys.stdout = log_file
    sys.stderr = log_file  # Optional: Redirect stderr to the same file

    # Ensure that logging is configured to use this file
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        force=True)
    # logging.basicConfig(filename=log_filename,
    #                     level=logging.INFO,
    #                     format='%(asctime)s - %(levelname)s - %(message)s',
    #                     force=True)  
    

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


def process_subhalo(i, Sub_data, current_redshift, shm_name, loop_num, dtype_subhalowake):
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
    
    result = Result(Tvir, lognH, rho_g_wake, gas_metallicity_host, volume_wake_tdyn_cm3, heating, specific_heating, volumetric_heating, normalized_heating, net_heating_flag, T_DF, T_allheating, cooling_rate_Tvir, cooling_rate_TDF,cooling_rate_Tallheating, Mach_rel, vel_rel)
    
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    shared_array = np.ndarray(loop_num, dtype=dtype_subhalowake, buffer=existing_shm.buf)
    
    for field in result._fields:
        shared_array[i][field] = getattr(result, field)
    
    existing_shm.close()
    
    #shared_array[i] = result
    
    #return subhalowake_info

# @profile
def process_subhalo_NonEq(i, Sub_data, current_redshift, shm_name, loop_num, dtype_subhalowake):
    
    try:
        logging.info(f"Subhalo {i} ...")
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
        evolve_cooling = True
        
        #logging.info(f"\nTvir: {Tvir} K, lognH: {lognH}")
        #logging.info("Normalized heating rate: ", normalized_heating, "erg cm^3/s")
        
        tfinal, T_DF_NonEq, cooling_rate_TDF_NonEq, cooling_rate_Tvir = get_Grackle_TDF_nonEquilibrium(Tvir, lognH, gas_metallicity_host, normalized_heating, current_redshift)
        
        #logging.info("Cooling rate (zero heating): ", cooling_rate_Tvir, "erg cm^3/s")
        
        result = ResultNonEq(i, Tvir, lognH, rho_g_wake, gas_metallicity_host, volume_wake_tdyn_cm3, heating, specific_heating, volumetric_heating, normalized_heating, tfinal, T_DF_NonEq, cooling_rate_Tvir, cooling_rate_TDF_NonEq, Mach_rel, vel_rel)
        
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        shared_array = np.ndarray(loop_num, dtype=dtype_subhalowake, buffer=existing_shm.buf)
        
        for field in result._fields:
            shared_array[i][field] = getattr(result, field)
        
        #return subhalowake_info
        #shared_array[i] = result
        logging.info(f"Subhalo {i} Done")
        existing_shm.close()
        
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        subhalowake_info = ResultNonEq(i, -999, -999, -999, -999,
                            -999, -999, -999, -999,
                            -999, -999,-999,-999,
                            -999,-999,-999)
        #return subhalowake_info
        #shared_array[i] = subhalowake_info
        for field in result._fields:
            shared_array[i][field] = getattr(result, field)
        



#-------------------------------------------------------------------------------------
# @profile
def main():
    
    
    TNG50_redshift_list = [20.05,14.99,11.98,10.98,10.00,9.39,9.00,8.45,8.01]
    snapNum = 7
    output_dir = "/home/zwu/21cm_project/grackle_DF_cooling/snap_"+str(snapNum)+"/"
    filepath = "/home/zwu/21cm_project/compare_TNG/results/TNG50-1/snap_"+str(snapNum)+"/"
    current_redshift = TNG50_redshift_list[snapNum]
    
    setup_logging(output_dir)
    logging.info("Main Process Start time: %s", datetime.datetime.now())
    
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
    
    #Model = 'SubhaloWake'
    Model = 'SubhaloWakeNonEq'
    logging.info(f"Model: {Model}")

   
    #output data type
    if Model == 'SubhaloWake':
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
        ], align=True)
    elif Model == 'SubhaloWakeNonEq':
        dtype_subhalowake = np.dtype([
            ('index',np.int32),
            ('Tvir', np.float64),
            ('lognH', np.float64),
            ('rho_g_wake', np.float64),
            ('gas_metallicity_host', np.float64),
            ('volume_wake_tdyn_cm3', np.float64),
            ('heating', np.float64),
            ('specific_heating', np.float64),
            ('volumetric_heating', np.float64),
            ('normalized_heating', np.float64),
            ('tfinal', np.float64),
            ('T_DF_NonEq', np.float64),
            ('cooling_rate_Tvir', np.float64),
            ('cooling_rate_TDF_NonEq', np.float64),
            ('Mach_rel', np.float64),
            ('vel_rel', np.float64)
        ], align=True)
        

            
    # manager = Manager()
    # Output_SubhaloWake_Info = manager.list()
    
    Num_processes = 32
    #pool = Pool(processes=Num_processes)
    pool = Pool(processes=Num_processes, initializer=setup_logging, initargs=(output_dir,))
    
    loop_num = len(Sub_data['Tvir_host'])
    # bytes_per_element = dtype_subhalowake.itemsize
    # buffer = sharedctypes.RawArray(ctypes.c_byte, loop_num * bytes_per_element)  
    # shared_array = np.frombuffer(buffer, dtype=dtype_subhalowake)
    # shared_array = shared_array.reshape((loop_num,))
    
    shm = shared_memory.SharedMemory(create=True, size=loop_num * dtype_subhalowake.itemsize)
    shared_array = np.ndarray(loop_num, dtype=dtype_subhalowake, buffer=shm.buf)
    
    logging.info(f"Number of subhalos: {loop_num}")
    
    # test_result = ResultNonEq(0,10,10,10,
    #                10,10,10,10,
    #                10,10,10,10,
    #                10,10,10,-5.)
    # for field in test_result._fields:
    #     shared_array[0][field] = getattr(test_result, field)
        
    # print("shared_array[0]: ", shared_array[0])
    # print("shared_array[0].dtype: ", shared_array[0].dtype)
    # print("shared_array: ", shared_array)
    

    # for i in range(loop_num):
    #     if Model == 'SubhaloWake':
    #         pool.apply_async(process_subhalo, args=(i, Sub_data, current_redshift), callback=lambda result: Output_SubhaloWake_Info.append(result))
    #     elif Model == 'SubhaloWakeNonEq':
    #         pool.apply_async(process_subhalo_NonEq, args=(i, Sub_data, current_redshift), callback=lambda result: Output_SubhaloWake_Info.append(result))
                  
            
    for i in range(loop_num):
        if Model == 'SubhaloWake':
            pool.apply_async(process_subhalo, args=(i, Sub_data, current_redshift, shm.name, loop_num, dtype_subhalowake))
        elif Model == 'SubhaloWakeNonEq':
            pool.apply_async(process_subhalo_NonEq, args=(i, Sub_data, current_redshift, shm.name, loop_num, dtype_subhalowake))

    pool.close()
    pool.join()
    
    #end of loop over subhalo
    
    if Model == 'SubhaloWake':
        output_filename = output_dir+f"Grackle_Cooling_SubhaloWake_FullModel_snap"+str(snapNum)+".h5"
        # with h5py.File(output_filename, 'w') as file:
        #     subhalowake_array = np.array(Output_SubhaloWake_Info, dtype=dtype_subhalowake)
        #     subhalowake_dataset = file.create_dataset('SubhaloWake', data=subhalowake_array)
        with h5py.File(output_filename, 'w') as file:
            subhalowake_array = np.array(shared_array, dtype=dtype_subhalowake)
            subhalowake_dataset = file.create_dataset('SubhaloWake', data=subhalowake_array)


    elif Model == 'SubhaloWakeNonEq':
        
        output_filename = output_dir+f"Grackle_Cooling_SubhaloWake_NonEq_snap"+str(snapNum)+".h5"
        # with h5py.File(output_filename, 'w') as file:
        #     subhalowake_array = np.array(Output_SubhaloWake_Info, dtype=dtype_subhalowake)
        #     subhalowake_dataset = file.create_dataset('SubhaloWakeNonEq', data=subhalowake_array)
        with h5py.File(output_filename, 'w') as file:
            subhalowake_array = np.array(shared_array, dtype=dtype_subhalowake)
            subhalowake_dataset = file.create_dataset('SubhaloWakeNonEq', data=subhalowake_array)



    logging.info("Main Process End time: %s", datetime.datetime.now())
    
#-------------------------------------------------------------------------------------    
    
amu_kg           = 1.660538921e-27  # g
mass_hydrogen_kg = 1.007947*amu_kg  # g
if __name__ == "__main__":
    main()
