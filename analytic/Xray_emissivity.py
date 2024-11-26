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


def Xray_comoving_emissivity(z,log10_LX = 40.0):
    SFRD = SFRD_comoving(z) #Msun/yr/cMpc^3
    LX = 10**log10_LX #erg/s / (Msun/yr)
    emissivity = LX * SFRD
    return emissivity
    
    
if __name__ == '__main__':
    TNG50_redshift_list = [20.05,14.99,11.98,10.98,10.00,9.39,9.00,8.45,8.01]
    snapNum_list = [1, 2, 3, 4, 6, 8]
    TNG50_selected_redshifts = np.array([TNG50_redshift_list[i] for i in snapNum_list])
    
    data_dict_list = []; tot_heating_rate_list = []; tot_Xray_Tvir_list = []; tot_Xray_allheating_list = []; tot_Xray_Tvir_apec_list = []; tot_Xray_allheating_apec_list = []
    for i in range(len(snapNum_list)):
        snapNum = snapNum_list[i]
        z = TNG50_redshift_list[i]
        input_dir = "/home/zwu/21cm_project/yt_Xray/snap_"+str(snapNum)+"/"
        filename = input_dir + "Xray_emissivity_snap" + str(snapNum) + ".h5"
        data_dict = read_hdf5_data(filename)
        # print(data_dict.keys())
        # dict_keys(['T_DF', 'T_allheating', 'Tvir', 'emissivity_Tallheating_apec', 'emissivity_Tallheating_cloudy', 'emissivity_Tvir_cloudy', 'emissivity_Tvir_apec', 'heating_rate', 'volume_wake_tdyn_cm3'])
        
        data_dict_list.append(data_dict)
        tot_heating_rate = np.sum(data_dict['heating_rate'])
        tot_Xray_Tvir = np.sum(data_dict['emissivity_Tvir_cloudy']*data_dict['volume_wake_tdyn_cm3'])
        tot_Xray_allheating = np.sum(data_dict['emissivity_Tallheating_cloudy']*data_dict['volume_wake_tdyn_cm3'])
        tot_Xray_Tvir_apec = np.sum(data_dict['emissivity_Tvir_apec']*data_dict['volume_wake_tdyn_cm3'])
        tot_Xray_allheating_apec = np.sum(data_dict['emissivity_Tallheating_apec']*data_dict['volume_wake_tdyn_cm3'])
        
        tot_heating_rate_list.append(tot_heating_rate)
        
        tot_Xray_Tvir_list.append(tot_Xray_Tvir)
        tot_Xray_Tvir_apec_list.append(tot_Xray_Tvir_apec)
        tot_Xray_allheating_list.append(tot_Xray_allheating)
        tot_Xray_allheating_apec_list.append(tot_Xray_allheating_apec)
        # print(f"tot_heating_rate = {tot_heating_rate:.2e} erg/s")
        # print(f"tot_Xray_Tvir = {tot_Xray_Tvir:.2e} erg/s")
        # print(f"tot_Xray_allheating = {tot_Xray_allheating:.2e} erg/s")
        
    tot_heating_rate_list = np.array(tot_heating_rate_list)
    tot_Xray_Tvir_list = np.array(tot_Xray_Tvir_list)
    tot_Xray_allheating_list = np.array(tot_Xray_allheating_list)
    tot_Xray_Tvir_apec_list = np.array(tot_Xray_Tvir_apec_list)
    tot_Xray_allheating_apec_list = np.array(tot_Xray_allheating_apec_list)

    comoving_boxsize = 51.7**3 #cMpc^3
    tot_heating_rate_list /= comoving_boxsize
    tot_Xray_Tvir_list /= comoving_boxsize
    tot_Xray_allheating_list /= comoving_boxsize
    tot_Xray_Tvir_apec_list /= comoving_boxsize
    tot_Xray_allheating_apec_list /= comoving_boxsize
    
    '''
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
    '''

    #Xray emissivity
    z_list = np.linspace(8,30,100)
    LX_list = [38,39,40,41,42]
    colors = ['b','g','y','r','m']
    Xray_emissivity_list = []
    for i in range(len(LX_list)):
        Xray_emissivity_list.append(np.array([Xray_comoving_emissivity(z,log10_LX = LX_list[i]) for z in z_list]))

    figure = plt.figure(figsize=(8,6),facecolor='w')
    for i in range(len(LX_list)):
        plt.plot(z_list,np.log10(Xray_emissivity_list[i]),colors[i]+'-',label='log$_{10}$L$_X$='+str(LX_list[i]))
    
    plt.scatter(TNG50_selected_redshifts,np.log10(tot_Xray_allheating_list),c='r',label='X-ray Tallheating (cloudy)')
    plt.scatter(TNG50_selected_redshifts,np.log10(tot_Xray_Tvir_list),c='b',label='X-ray Tvir (cloudy)')
    plt.scatter(TNG50_selected_redshifts,np.log10(tot_Xray_allheating_apec_list),c='r',marker='^',label='X-ray Tallheating (apec)')
    plt.scatter(TNG50_selected_redshifts,np.log10(tot_Xray_Tvir_apec_list),c='b',marker='^',label='X-ray Tvir_apec (apec)')
    
    plt.scatter(TNG50_selected_redshifts,np.log10(tot_heating_rate_list),c='g',label='DF heating rate')
    
    plt.xlabel('z',fontsize=14)
    plt.ylabel(r'log$_{10}$[X-ray emissivity (erg s$^{-1}$ cMpc$^{-3}$)]',fontsize=14)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('figures/Xray_emissivity.png',dpi=300)