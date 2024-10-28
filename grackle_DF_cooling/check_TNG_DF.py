import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from DF_cooling_rate import run_cool_rate
from scipy.optimize import brentq
import datetime
import sys
import tracemalloc


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
amu_kg           = 1.660538921e-27  # g
mass_hydrogen_kg = 1.007947*amu_kg  # g
if __name__ == "__main__":
    
    print("Start time: ", datetime.datetime.now())
    tracemalloc.start()
    
    output_dir = "/home/zwu/21cm_project/grackle_DF_cooling/snap_3/"
    filepath = "/home/zwu/21cm_project/compare_TNG/results/TNG50-1/snap_3/"
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
    fig = plt.figure(facecolor='white',figsize=(6,6))
    plt.hist(gas_metallicity, bins=100)
    plt.xlabel("Z/Zsun")
    plt.ylabel("count")
    plt.savefig(output_dir+"Host_halo_gas_metallicity_distribution.png",dpi=300)
    
    
    snapshot1 = tracemalloc.take_snapshot()
    
    if Model == 'Hosthalo':
        print("Using host halo data")
        V_host = (4/3)*np.pi*Host_data['R_crit200_m']**3
        M_gas_kg = Host_data['M_gas_kg']
        M_crit200_kg = Host_data['M_crit200_kg']
        rho_g_analytic_200 = Host_data['rho_g_analytic_200']
        rho_allgas = M_gas_kg/V_host
        Omega_b0 = 0.0486; Omgea_m0 = 0.3089
        Mgas_crit200_kg = M_crit200_kg*(Omega_b0/Omgea_m0)
        rho_crit200 = Mgas_crit200_kg/V_host
        #compare these densities, rho_crit200 = rho_g_analytic_200 < rho_allgas (entire halo)
        # print("rho_g_analytic_200: ", rho_g_analytic_200)
        # print("rho_allgas: ", rho_allgas)
        # print("rho_crit200: ", rho_crit200)
        
        rho_g_host = np.where(rho_crit200 > 0, rho_crit200, rho_g_analytic_200)  #use rho_g_analytic_200 if rho_crit200 is missing
        
        nH_host = rho_g_host/mass_hydrogen_kg  #assume only hydrogen gas so that volumetric heating model = specific heating model
        nH_host_cm3 = nH_host/1e6 #convert to cm^-3
        lognH_host = np.log10(nH_host_cm3)
      
        #specific heating rate
        DF_heating_thishost_erg = Host_data['DF_heating_thishost']*1e7 #convert to erg/s
        V_host_cm3 = V_host*1e6
        Specific_heating_thishost = DF_heating_thishost_erg/(1e3*rho_g_host*V_host) #erg/s/g
        Normalized_heating_thishost = DF_heating_thishost_erg/V_host_cm3/nH_host_cm3**2 #erg cm^3 s^-1
        Volumetric_heating_thishost = DF_heating_thishost_erg/V_host_cm3 #erg/s/cm^3
        
          
        #plot histogram of log10(nH)
        fig = plt.figure(facecolor='white',figsize=(6,6))
        plt.hist(np.log10(nH_host_cm3), bins=100)
        plt.xlabel("log10(nH)")
        plt.ylabel("count")
        plt.savefig(output_dir+"Host_halo_gas_lognH_distribution.png",dpi=300)
        
        #plot histogram of Tvir
        fig = plt.figure(facecolor='white',figsize=(6,6))
        plt.hist(np.log10(Host_data['Tvir_host']), bins=100)
        plt.xlabel("log10(Tvir)")
        plt.ylabel("count")
        plt.savefig(output_dir+"Host_halo_gas_Tvir_distribution.png",dpi=300)
        
        
        num_halos = len(Host_data['M_crit200_kg'])
        
        if TemperatureModel == "SimpleAverage":
            lognH = 0.0 #use lognH=0 data
            cooling_filename = f'new_data/zero_metallicity/DF_cooling_rate_z{current_redshift:.1f}_lognH{lognH:.0f}.h5'
            cooling_data = read_hdf5_data(cooling_filename)
            
            
            temperature = cooling_data["data/temperature"]
            cooling_time = cooling_data["data/cooling_time"]
            cooling_rate = cooling_data["data/cooling_rate"]
            turning_point = np.where(cooling_time<0)[0][0]

            #compare cooling rate and DF_heating
            test_num = 10
            for i in range(test_num):
                T = Host_data['T_DFheating_host'][i]
                #interpolate cooling rate, use the lognH=-0.5 data
                cooling_rate_thishalo = np.interp(T, temperature, cooling_rate) #erg cm^3 s^-1
                heating = Host_data['DF_heating_thishost'][i]
                #convert DF_heating to erg cm^3 s^-1
                heating *= 1e7 #convert to erg/s
                V = V_host[i]
                V_cm3 = V*1e6
                nH_cm3 = nH_host_cm3[i]
                lognH = np.log10(nH_cm3)
                normalized_heating = heating/V_cm3/nH_cm3**2 #erg cm^3 s^-1
                specific_heating = Specific_heating_thishost[i] #erg/s/g
                
                
                print(f"\nTavg: {T} K, lognH: {lognH}")
                print("Specific heating rate: ", specific_heating, "erg/g/s")
                print("Normalized heating rate: ", normalized_heating, "erg cm^3 s^-1")
                print("cooling rate: ", cooling_rate_thishalo, "erg cm^3 s^-1")
               
                
      
        elif TemperatureModel == 'Virial':
            test_num = len(Host_data['Tvir_host'])
            for i in range(test_num):
                Tvir = Host_data['Tvir_host'][i]
                gas_metallicity = Host_data['gas_metallicity_host'][i]
                gas_metallicity /= 0.01295 #normalize to solar metallicity
                lognH = lognH_host[i]
                
                specific_heating = Specific_heating_thishost[i]
                normalized_heating = Normalized_heating_thishost[i]
                volumetric_heating = Volumetric_heating_thishost[i]
                
                
                #pass the parameters to run_cool_rate
                #use either specific_heating or volumetric_heating, not both
                evolve_cooling = False
                
                #compare with the case of zero heating
                cooling_data_zeroheating = run_cool_rate(evolve_cooling,current_redshift,lognH,0.0, 0.0, Tvir,gas_metallicity)
                cooling_rate_zeroheating = cooling_data_zeroheating["cooling_rate"][0]
                print(f"\nTvir: {Tvir} K, lognH: {lognH}")
                # print("Specific heating rate: ", specific_heating, "erg/g/s")
                # print("Volumetric heating rate: ", volumetric_heating, "erg/cm^3/s")
                print("Normalized heating rate: ", normalized_heating, "erg cm^3 s^-1")
                print("Cooling rate (zero heating): ", cooling_rate_zeroheating)
                cooling_rate_zeroheating = cooling_rate_zeroheating.v.item()
                
                net_heating_flag = 0
                if cooling_rate_zeroheating + normalized_heating <= 0:
                    #net cooling even with DF heating
                    net_heating_flag = -1
                    T_DF = Tvir
                    cooling_rate_TDF = cooling_rate_zeroheating
                    print("Net cooling even with DF heating")
                else:
                    net_heating_flag = 1
                    
                    T_DF = get_Grackle_TDF(Tvir, lognH, gas_metallicity, normalized_heating, current_redshift)
                    print("Equilibrium Temperature:", T_DF if T_DF is not None else "No equilibrium found.")
                    cooling_data_TDF = run_cool_rate(evolve_cooling,current_redshift,lognH, 0.0, 0.0, T_DF,gas_metallicity)
                    cooling_rate_TDF = cooling_data_TDF["cooling_rate"][0].v.item()
                    


                hosthalo_info = (Tvir, lognH, gas_metallicity, specific_heating, volumetric_heating, normalized_heating, net_heating_flag, T_DF, cooling_rate_zeroheating, cooling_rate_TDF)
                Output_Hosthalo_Info.append(hosthalo_info)
                
                print("\n\n")
    
                
    elif Model == 'SubhaloWake':   #use subhalo instead of host halo
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
        
        test_num = len(Sub_data['Tvir_host'])
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
            #cooling_data = run_cool_rate(evolve_cooling,current_redshift,lognH, specific_heating, 0.0, Tvir,gas_metallicity_host)
            #cooling_data_2 = run_cool_rate(evolve_cooling,current_redshift,lognH, 0.0, volumetric_heating, Tvir,gas_metallicity_host)
            cooling_data_zeroheating = run_cool_rate(evolve_cooling,current_redshift,lognH,0.0, 0.0, Tvir,gas_metallicity_host)
            
           
            
            
            print(f"\nTvir: {Tvir} K, lognH: {lognH}")
            print("Specific heating rate: ", specific_heating, "erg/g/s")
            # print("Volumetric heating rate: ", volumetric_heating, "erg/cm^3/s")
            # print("Normalized heating rate: ", normalized_heating, "erg cm^3 s^-1")
            
            #print("Cooling rate (with specific heating): ", cooling_data["cooling_rate"][0])
            #print("Cooling rate (with volumetric heating): ", cooling_data_2["cooling_rate"][0])  
            print("Cooling rate (zero heating): ", cooling_data_zeroheating["cooling_rate"][0])
            
            #print(cooling_data["cooling_time"])
            #print(cooling_data_zeroheating["cooling_time"])
            #print("cooling time ratio: ", cooling_data["cooling_time"]/cooling_data_zeroheating["cooling_time"])
            #print("cooling rate ratio: ", cooling_data_zeroheating["cooling_rate"]/cooling_data["cooling_rate"])
            cooling_rate_zeroheating = cooling_data_zeroheating["cooling_rate"][0].v.item()
            
            net_heating_flag = 0
            if cooling_rate_zeroheating + normalized_heating <= 0:
                #net cooling even with DF heating
                net_heating_flag = -1
                T_DF = Tvir
                cooling_rate_TDF = cooling_rate_zeroheating
                print("Net cooling even with DF heating")
            else:
                net_heating_flag = 1
                
                # T_DF = get_Grackle_TDF(Tvir, lognH, gas_metallicity_host, normalized_heating, current_redshift)
                # print("Equilibrium Temperature:", T_DF if T_DF is not None else "No equilibrium found.")
                # cooling_data_TDF = run_cool_rate(evolve_cooling,current_redshift,lognH, 0.0, 0.0, T_DF,gas_metallicity_host)
                # cooling_rate_TDF = cooling_data_TDF["cooling_rate"][0].v.item()
                
                #for testing, set T_DF = -1
                T_DF = -1
                cooling_rate_TDF = -1
                
             
            subhalowake_info = (Tvir, lognH, rho_g_wake, gas_metallicity_host, volume_wake_tdyn_cm3, heating, specific_heating, volumetric_heating, normalized_heating, net_heating_flag, T_DF, cooling_rate_zeroheating, cooling_rate_TDF, Mach_rel, vel_rel)
            Output_SubhaloWake_Info.append(subhalowake_info)  
            print("\n\n")
            
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
            
    #output the results to a file
    if Model == 'Hosthalo':
        dtype_hosthalo = np.dtype([
            ('Tvir', np.float64),
            ('lognH', np.float64),
            ('gas_metallicity_host', np.float64),
            ('specific_heating', np.float64),
            ('volumetric_heating', np.float64),
            ('normalized_heating', np.float64),
            ('net_heating_flag', np.int32),
            ('T_DF', np.float64),
            ('cooling_rate_zeroheating', np.float64),
            ('cooling_rate_TDF', np.float64)
        ])
        
        # output_filename =  output_dir+"Grackle_Cooling_Hosthalo_new.h5"
        # with h5py.File(output_filename, 'w') as file:
        #     hosthalo_array = np.array(Output_Hosthalo_Info, dtype=dtype_hosthalo)
        #     hosthalo_dataset = file.create_dataset('HostHalo', data=hosthalo_array)
        
        
    #read file
    # Halodata = read_hdf5_data(output_filename)
    # print(Halodata['HostHalo'].dtype.names)
    
    print("End time: ", datetime.datetime.now())