import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from DF_cooling_rate import run_cool_rate
from scipy.optimize import brentq
import datetime
import sys
import tracemalloc
from memory_profiler import profile
import gc
import cProfile
import pstats

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
    def net_heating(T, lognH, metallicity, normalized_heating, redshift):
        cooling_data = run_cool_rate(False, redshift, lognH, 0.0, 0.0, T, metallicity)
        cooling_rate = cooling_data["cooling_rate"][0].v.item()
        return cooling_rate + normalized_heating

    # Helper function to find equilibrium temperature
    def find_equilibrium(T_low, T_high, abs_tol, rel_tol):
        if net_heating(T_low, lognH, metallicity, normalized_heating, redshift) * net_heating(T_high, lognH, metallicity, normalized_heating, redshift) > 0:
            return None
        else:
            return brentq(net_heating, T_low, T_high, args=(lognH, metallicity, normalized_heating, redshift), xtol=abs_tol, rtol=rel_tol)

    # Set initial range and tolerances
    T_low = initial_T
    T_high = 10 * initial_T
    abs_tol = 10  # Absolute tolerance  10 K
    rel_tol = 1e-3  # Relative tolerance

    # First attempt to find the equilibrium temperature
    equilibrium_temperature = find_equilibrium(T_low, T_high, abs_tol, rel_tol)
    if equilibrium_temperature is None:
        T_high = 100 * initial_T
        equilibrium_temperature = find_equilibrium(T_low, T_high, abs_tol, rel_tol)
        if equilibrium_temperature is None:
            print("Warning: No equilibrium found within 100 times the initial temperature. Please check the parameters or extend the range further.")
    
    return equilibrium_temperature

#-------------------------------------------------------------------------------------
def main():
    print("Start time: ", datetime.datetime.now())
    tracemalloc.start()
    
    output_dir = "/home/zwu/21cm_project/grackle_DF_cooling/snap_4/"
    filepath = "/home/zwu/21cm_project/compare_TNG/results/TNG50-1/snap_4/"
    current_redshift = 10.0
    
    #find the hdf5 file starting with "DF_heating_snap"
    for file in os.listdir(filepath):
        if file.startswith("DF_heating_snap"):
            filename = filepath + file
            break
    print(f"Reading data from {filename}")
    data_dict = read_hdf5_data(filename)
    Host_data = data_dict["HostHalo"]
    Sub_data = data_dict["SubHalo"]
    

    Cooling_Heating_ratio_list = []
    
    
    Model = 'SubhaloWake'
    TemperatureModel = 'Virial'
    
    if Model == 'Hosthalo':
        Output_Hosthalo_Info = []
    elif Model == 'SubhaloWake':
        Output_SubhaloWake_Info = []
        
    
    #histogram of host halo gas metallicity
    gas_metallicity = Host_data['gas_metallicity_host']
    gas_metallicity /= 0.01295
    
    snapshot1 = tracemalloc.take_snapshot()
    
    if Model == 'SubhaloWake':   #use subhalo instead of host halo
        print("Using subhalo data")
        
        print("t_dyn:",Sub_data['t_dyn'])
        
        max_output_len = 50000
        batch_num = -1
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
            ('cooling_rate_zeroheating', np.float64),
            ('cooling_rate_TDF', np.float64),
            ('Mach_rel', np.float64),
            ('vel_rel', np.float64)
        ])
        
        #test_num = len(Sub_data['Tvir_host'])
        test_num = 1500 #for memory testing
        for i in range(test_num):
            print(f"Subhalo {i} (total {test_num})")
            Tvir = Sub_data['Tvir_host'][i]
            
            #use host halo gas metallicity because subhalo gas metallicity is often not available
            gas_metallicity_sub = Sub_data['gas_metallicity_sub'][i]
            gas_metallicity_sub /= 0.01295
            gas_metallicity_host = Sub_data['gas_metallicity_host'][i]
            gas_metallicity_host /= 0.01295
            print("Zsub: ", gas_metallicity_sub, "Zsun")
            print("Zhost: ", gas_metallicity_host, "Zsun")
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
          
            cooling_data_zeroheating = run_cool_rate(evolve_cooling,current_redshift,lognH,0.0, 0.0, Tvir,gas_metallicity_host)
            cooling_rate_zeroheating = cooling_data_zeroheating["cooling_rate"][0].v.item()
            
            print(f"\nTvir: {Tvir} K, lognH: {lognH}")
            print("Specific heating rate: ", specific_heating, "erg/g/s")
        
            print("Cooling rate (zero heating): ", cooling_rate_zeroheating, "erg cm^3/s")
            
            
            net_heating_flag = 0
            if cooling_rate_zeroheating + normalized_heating <= 0:
                #net cooling even with DF heating
                net_heating_flag = -1
                T_DF = Tvir
                cooling_rate_TDF = cooling_rate_zeroheating
                print("Net cooling even with DF heating")
            else:
                net_heating_flag = 1

                #for testing, set T_DF = -1
                T_DF = -1
                cooling_rate_TDF = -1
                
             
            subhalowake_info = (Tvir, lognH, rho_g_wake, gas_metallicity_host, volume_wake_tdyn_cm3, heating, specific_heating, volumetric_heating, normalized_heating, net_heating_flag, T_DF, cooling_rate_zeroheating, cooling_rate_TDF, Mach_rel, vel_rel)
            Output_SubhaloWake_Info.append(subhalowake_info)  
            print("\n\n")
            
            # gc.collect()
            
            
            
            if (i % max_output_len == 0 and i > 0) or i == test_num-1:
                batch_num += 1
                
                output_filename = output_dir+f"Grackle_Cooling_SubhaloWake_TestCoolingMach_{batch_num}.h5"
                with h5py.File(output_filename, 'w') as file:
                    subhalowake_array = np.array(Output_SubhaloWake_Info, dtype=dtype_subhalowake)
                    subhalowake_dataset = file.create_dataset('SubhaloWake', data=subhalowake_array)
                
                Output_SubhaloWake_Info = []
            
            if(i% 500 == 0):
                print("\n\n")
                print("Memory usage: ", tracemalloc.get_traced_memory())
                print("Current time: ", datetime.datetime.now())
                print("Current subhalo: ", i)
                snapshot_new = tracemalloc.take_snapshot()
                stats = snapshot_new.compare_to(snapshot1, 'lineno')
                for stat in stats[:5]:
                    print(stat)
                print("\n\n")
 
    print("End time: ", datetime.datetime.now())
    
#-------------------------------------------------------------------------------------    
    
amu_kg           = 1.660538921e-27  # g
mass_hydrogen_kg = 1.007947*amu_kg  # g
if __name__ == "__main__":
    profiler = cProfile.Profile()
    print("Start main")
    profiler.run('main()')
    stats = pstats.Stats(profiler)
    stats.dump_stats('profile.stats')
    print("End main")
    
    stats = pstats.Stats('profile.stats')
    stats.strip_dirs().sort_stats('cumulative').print_stats()