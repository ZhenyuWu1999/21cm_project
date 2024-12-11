import numpy as np
import matplotlib.pyplot as plt
from math import pi, erfc
from scipy.integrate import solve_ivp, quad
import warnings
from scipy.special import gamma
from scipy.integrate import nquad

from colossus.cosmology import cosmology
from colossus.lss import mass_function

from physical_constants import *
from linear_evolution import *
import h5py
import os

def print_attrs(name, obj):
    """Helper function to print the name of an HDF5 object and its attributes."""
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))

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
                # for key, val in obj.attrs.items():
                #     print(f"    {key}: {val}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")
                for key, val in obj.attrs.items():
                    print(f"    {key}: {val}")
                for subname, subitem in obj.items():
                    read_recursive(f"{name}/{subname}", subitem)
        # Read data starting from root
        f.visititems(read_recursive)
    return data_dict


def SFRD_comoving(z,f_star=0.01):
    #rho_crit_z0 in Msun/Mpc^3, return SFRD in Msun/yr/cMpc^3
    global rho_crit_z0, Omega_b
    delta_nl = 0.0
    dfdz = df_dz(z)
    dfdt = dfdz/dt_dz(z)
    yr_to_sec = 3.154e7
    dfdt_yr = dfdt * yr_to_sec
    #evaluate in comoving box, so density is just rho_crit_z0
    SFRD = rho_crit_z0*Omega_b*f_star*(1+delta_nl)*dfdt_yr
    return SFRD

def plot_SFRD_comoving():
    z_list = np.linspace(0,30,100)
    
    f_coll_list = np.array([f_coll(z) for z in z_list])
    f_coll2_list = np.array([f_coll2(z) for z in z_list])
    
    figure = plt.figure(figsize=(8,6),facecolor='w')
    plt.plot(z_list,f_coll_list,'b-')
    plt.plot(z_list,f_coll2_list,'r--')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('z')
    plt.ylabel('f$_{coll}$')
    plt.savefig('f_coll.png')

    
    #dfdz and dfdt
    figure = plt.figure(figsize=(8,6),facecolor='w')
    z_list = np.linspace(8,30,100)
    dfdz_list = np.array([df_dz(z) for z in z_list])
    dfdt_list = np.array([df_dz(z)/dt_dz(z) for z in z_list])
    figure = plt.figure(figsize=(8,6),facecolor='w')
    plt.plot(z_list,dfdz_list,'b-')
    plt.xlabel('z')
    plt.ylabel('df/dz')
    plt.savefig('dfdz.png')

    figure = plt.figure(figsize=(8,6),facecolor='w')
    plt.plot(z_list,dfdt_list,'b-')
    plt.xlabel('z')
    plt.ylabel('df/dt')
    plt.savefig('dfdt.png')

    #SFRD
    figure = plt.figure(figsize=(8,6),facecolor='w')
    z_list = np.linspace(8,30,100)
    SFRD_list = np.array([SFRD_comoving(z) for z in z_list])
    figure = plt.figure(figsize=(8,6),facecolor='w')
    log_SFRD_list = np.log10(SFRD_list)

    plt.plot(z_list,log_SFRD_list,'b-')
    plt.xlabel('z',fontsize=14)
    plt.ylabel(r'SFRD log$_{10}$[Msun yr$^{-1}$ cMpc$^{-3}$]',fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig('SFRD.png')


def Xray_comoving_emissivity(z,log10_LX = 40.0):
    SFRD = SFRD_comoving(z) #Msun/yr/cMpc^3
    LX = 10**log10_LX #erg/s / (Msun/yr)
    emissivity = LX * SFRD
    return emissivity

        

def process_xray_data(TNG50_redshift_list, snapNum_list, model_type='SubhaloWake',Volume = None):
    """
    Process X-ray emission data for SubhaloWake and SubhaloWakeNonEq models
    
    Args:
        TNG50_redshift_list: List of redshift values
        snapNum_list: List of snapshot numbers
        model_type: 'SubhaloWake' for equilibrium or 'SubhaloWakeNonEq' for non-equilibrium analysis
    """
    results = []
    
    for snapNum in snapNum_list:
        input_dir = f"/home/zwu/21cm_project/yt_Xray/snap_{snapNum}/"
        suffix = "NonEq_" if model_type == 'SubhaloWakeNonEq' else ""
        filename = f"{input_dir}Xray_emissivity_{suffix}snap{snapNum}.h5"
        if Volume is not None:
            filename = f"{input_dir}Xray_emissivity_{suffix}Volume{Volume}_snap{snapNum}.h5"
        data = read_hdf5_data(filename)
        
        # Calculate volume-weighted emissions
        volume = data['volume_wake_tdyn_cm3']
        
        #also read the original TNG50 data
        if model_type == 'SubhaloWakeNonEq':
            TNG_data_dir = f'/home/zwu/21cm_project/compare_TNG/results/TNG50-1/snap_{snapNum}/'
            #find the file starting with DF_heating_snap{snapNum}
            TNG_filename = [f for f in os.listdir(TNG_data_dir) if f.startswith(f'DF_heating_snap{snapNum}')][0]
            TNGdata_dict = read_hdf5_data(TNG_data_dir + TNG_filename)
            Sub_data = TNGdata_dict["SubHalo"]
            #Host_data = TNGdata_dict["HostHalo"]
             
            index = data['index']
            Volume_filling_factor = Sub_data['Volume_filling_factor'][index]
            
        
        if model_type == 'SubhaloWake':
            emissions = {
                'heating_rate': np.sum(data['heating_rate']),
                'Xray_Tvir_cloudy': np.sum(data['emissivity_Tvir_cloudy'] * volume),
                'Xray_allheating_cloudy': np.sum(data['emissivity_Tallheating_cloudy'] * volume),
                'Xray_Tvir_apec': np.sum(data['emissivity_Tvir_apec'] * volume),
                'Xray_allheating_apec': np.sum(data['emissivity_Tallheating_apec'] * volume)
            }
        else:  # SubhaloWakeNonEq
            emissions = {
                'Xray_TDF_NonEq_cloudy': np.sum(data['emissivity_TDF_NonEq_cloudy'] * volume),
                'Xray_TDF_NonEq_apec': np.sum(data['emissivity_TDF_NonEq_apec'] * volume),
                'Xray_subsonic': np.sum(data['emissivity_TDF_NonEq_cloudy'][data['Mach_rel'] < 1.0] * 
                                      volume[data['Mach_rel'] < 1.0]),
                'Xray_supersonic': np.sum(data['emissivity_TDF_NonEq_cloudy'][data['Mach_rel'] > 1.0] * 
                                        volume[data['Mach_rel'] > 1.0]),
                'Xray_T1e8': np.sum(data['emissivity_TDF_NonEq_cloudy'][data['T_DF_NonEq'] < 1e8] * 
                                   volume[data['T_DF_NonEq'] < 1e8]),
                'Xray_T1e7': np.sum(data['emissivity_TDF_NonEq_cloudy'][data['T_DF_NonEq'] < 1e7] * 
                                   volume[data['T_DF_NonEq'] < 1e7]),
                'VFF_1e-1': np.sum(data['emissivity_TDF_NonEq_cloudy'][Volume_filling_factor < 1e-1] *
                                      volume[Volume_filling_factor < 1e-1]),
                'VFF_1e-2': np.sum(data['emissivity_TDF_NonEq_cloudy'][Volume_filling_factor < 1e-2] *
                                        volume[Volume_filling_factor < 1e-2])
            }
        
        results.append(emissions)
    
    # Convert lists to arrays and normalize by box size
    comoving_boxsize = 51.7**3  # cMpc^3
    return {key: np.array([result[key] for result in results]) / comoving_boxsize 
            for key in results[0].keys()}
    
    
def plot_xray_emissivity(wake_results, wake_noneq_results, wake_noneq_volume10,wake_noneq_volume01,  TNG50_redshift_list, snapNum_list, z_list=None, output_path='./figures/Xray_emissivity_new_test.png'):
    """
    Plot X-ray emissivity for different models and conditions
    
    Args:
        wake_results: Results dictionary from SubhaloWake model
        wake_noneq_results: Results dictionary from SubhaloWakeNonEq model
        TNG50_redshift_list: List of redshift values
        snapNum_list: List of snapshot numbers
        z_list: Optional list of redshift values for SFR calculation
        output_path: Path to save the figure
    """
    # Set up the plot
    plt.figure(figsize=(8, 6), facecolor='w')
    plt.xlim([7, 25])
    
    # Plot SFR X-ray emissivity if z_list is provided
    if z_list is not None:
        LX_list = [38, 39, 40, 41, 42]
        colors = ['b', 'g', 'y', 'r', 'm']
        for i, LX in enumerate(LX_list):
            Xray_emissivity = np.array([Xray_comoving_emissivity(z, log10_LX=LX) for z in z_list])
            plt.plot(z_list, np.log10(Xray_emissivity), f'{colors[i]}-', 
                    label=f'log$_{{10}}$L$_X$={LX}')
    plt.legend(loc='lower left')
    # Get redshifts for selected snapshots
    selected_redshifts = np.array([TNG50_redshift_list[i] for i in snapNum_list])
    
    # Plot SubhaloWake results
    plt.scatter(selected_redshifts, np.log10(wake_results['Xray_allheating_cloudy']), 
               c='r', label='X-ray Tallheating (cloudy)')
    plt.scatter(selected_redshifts, np.log10(wake_results['Xray_Tvir_cloudy']), 
               c='b', label='X-ray Tvir (cloudy)')
    plt.scatter(selected_redshifts, np.log10(wake_results['Xray_allheating_apec']), 
               c='r', marker='^', label='X-ray Tallheating (apec)')
    plt.scatter(selected_redshifts, np.log10(wake_results['Xray_Tvir_apec']), 
               c='b', marker='^', label='X-ray Tvir (apec)')
    plt.scatter(selected_redshifts, np.log10(wake_results['heating_rate']), 
               c='g', label='DF heating rate')
    
    # Plot SubhaloWakeNonEq results
    plt.scatter(selected_redshifts, np.log10(wake_noneq_results['Xray_TDF_NonEq_cloudy']), 
               c='m', marker='o', label='X-ray NonEq (cloudy)')
    plt.scatter(selected_redshifts, np.log10(wake_noneq_results['Xray_subsonic']), 
               marker='<', facecolors='none', edgecolors='m', label='X-ray NonEq (subsonic)')
    plt.scatter(selected_redshifts, np.log10(wake_noneq_results['Xray_supersonic']), 
               marker='>', facecolors='none', edgecolors='m', label='X-ray NonEq (supersonic)')
    plt.scatter(selected_redshifts, np.log10(wake_noneq_results['Xray_T1e8']), 
               marker='v', facecolors='none', edgecolors='m', s=20, label='X-ray NonEq (T_DF < 1e8)')
    plt.scatter(selected_redshifts, np.log10(wake_noneq_results['Xray_T1e7']), 
               marker='v', facecolors='none', edgecolors='m', s=10, label='X-ray NonEq (T_DF < 1e7)')
    plt.scatter(selected_redshifts, np.log10(wake_noneq_results['VFF_1e-1']),
                marker='s', facecolors='none', edgecolors='m', s=10, label='X-ray NonEq (Volume < 1e-1)')
    plt.scatter(selected_redshifts, np.log10(wake_noneq_results['VFF_1e-2']),
                marker='s', facecolors='none', edgecolors='m', s=5, label='X-ray NonEq (Volume < 1e-2)')
    
    
    volume_redshit = np.array([14.99])
    plt.scatter(volume_redshit, np.log10(wake_noneq_volume10['Xray_TDF_NonEq_cloudy']), 
               c='m', marker='X', label='X-ray NonEq (cloudy, Volume x10)')
    plt.scatter(volume_redshit, np.log10(wake_noneq_volume01['Xray_TDF_NonEq_cloudy']),
                c='m', marker='x', label='X-ray NonEq (cloudy, Volume x0.1)')
    
    
    # Set labels and styling
    plt.xlabel('z', fontsize=14)
    plt.ylabel(r'log$_{10}$[X-ray emissivity (erg s$^{-1}$ cMpc$^{-3}$)]', fontsize=14)
    plt.legend(loc = 'lower right')
    plt.grid()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()

       
if __name__ == '__main__':
    TNG50_redshift_list = [20.05,14.99,11.98,10.98,10.00,9.39,9.00,8.45,8.01]
    
    snapNum_list = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # Process the data
    wake_noneq_results = process_xray_data(TNG50_redshift_list, snapNum_list, 'SubhaloWakeNonEq')
    wake_results = process_xray_data(TNG50_redshift_list, snapNum_list, 'SubhaloWake')
    
    snapNum_list_volume = [1]
    Volume = 10
    wake_noneq_volume10 = process_xray_data(TNG50_redshift_list, snapNum_list_volume, 'SubhaloWakeNonEq', Volume)
    Volume = 0.1
    wake_noneq_volume01 = process_xray_data(TNG50_redshift_list, snapNum_list_volume, 'SubhaloWakeNonEq', Volume)
    
    
    # Optional: Calculate SFR X-ray emissivity
    z_list = np.linspace(8, 30, 100)
    
    # Create the plot
    plot_xray_emissivity(wake_results, wake_noneq_results, wake_noneq_volume10, wake_noneq_volume01, TNG50_redshift_list, snapNum_list, z_list)
    