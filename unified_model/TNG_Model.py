import numpy as np
import illustris_python as il
import os
import matplotlib.pyplot as plt

from physical_constants import *
from HaloProperties import *
from Config import snapNum, simulation_set
from DF_Ostriker99_wake_structure import Idf_Ostriker99_wrapper
from TNG_plots import plot_2D_histogram
from TNGDataHandler import *


def load_tng_data(basePath, snapNum):
    print("loading header ...")
    header = il.groupcat.loadHeader(basePath, snapNum)
    
    print("loading halos ...")
    # Define halo fields that need to be converted to float64
    halo_float_fields = [
        'GroupMass', 
        'GroupMassType',
        'GroupPos',
        'Group_M_Crit200',
        'Group_R_Crit200',
        'Group_M_Crit500',
        'Group_R_Crit500',
        'GroupVel',
        'GroupGasMetallicity'
    ]
    # Load halo data (GroupFirstSub and GroupNsubs are integer fields, keep as is)
    halos = il.groupcat.loadHalos(basePath, snapNum, 
                                 fields=['GroupFirstSub', 'GroupNsubs'] + halo_float_fields)
    
    print("loading subhalos ...")
    # Define subhalo fields that need to be converted to float64
    subhalo_float_fields = [
        'SubhaloMass',
        'SubhaloPos',
        'SubhaloVel',
        'SubhaloHalfmassRad',
        'SubhaloVmaxRad',
        'SubhaloMassType',
        'SubhaloGasMetallicity'
    ]
    # Load subhalo data (SubhaloGrNr is integer field, keep as is)
    subhalos = il.groupcat.loadSubhalos(basePath, snapNum, 
                                       fields=subhalo_float_fields + ['SubhaloGrNr'])
    
    # Convert float fields in halos to float64
    for field in halo_float_fields:
        if field in halos:
            halos[field] = halos[field].astype(np.float64)
    
    # Convert float fields in subhalos to float64
    for field in subhalo_float_fields:
        if field in subhalos:
            subhalos[field] = subhalos[field].astype(np.float64)
    
    return header, halos, subhalos



def TNG_model():    

    if simulation_set == 'TNG50-1':
        gas_resolution = 4.5e5 * h_Hubble #convert from Msun to Msun/h
        dark_matter_resolution = 8.5e4 * h_Hubble
    elif simulation_set == 'TNG100-1':
        gas_resolution = 1.4e6 * h_Hubble  
        dark_matter_resolution = 7.5e6 * h_Hubble  
    elif simulation_set == 'TNG300-1':
        gas_resolution = 1.1e7 * h_Hubble
        dark_matter_resolution = 5.9e7 * h_Hubble
    
    basePath = '/home/zwu/21cm_project/TNG_data/'+simulation_set+'/output'
    output_dir = '/home/zwu/21cm_project/unified_model/TNG_results/'+simulation_set+'/'
    output_dir += f'snap_{snapNum}/'
    #mkdir for this snapshot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("output_dir: ",output_dir)
    
    header, halos, subhalos = load_tng_data(basePath, snapNum)
    
    print("\n")
    print("redshift: ", header['Redshift'])
    current_redshift = header['Redshift']
    scale_factor = header['Time']
    print(scale_factor)
    print("box size: ", header['BoxSize']," ckpc/h")  
    volume = (header['BoxSize']/1e3 * scale_factor) **3  # (Mpc/h)^3

    print(header.keys())
    print(halos.keys())
    print(subhalos.keys())

    print("number of halos: ", halos['count'])
    print("number of subhalos: ", subhalos['count'])
    
    #select halos with mass > 100*dark_matter_resolution
    mask_groupmass = halos['GroupMass']*1e10 > 100*dark_matter_resolution #Msun/h
    mask_M200 = halos['Group_M_Crit200']*1e10 > 0
    mask_R200 = halos['Group_R_Crit200'] > 0
    mask_subhalo = halos['GroupNsubs'] > 1 #at least 1 subhalo besides the central subhalo
    num_unresolved = np.sum(~mask_groupmass)
    num_M200_zero = np.sum(~mask_M200)
    num_R200_zero = np.sum(~mask_R200)
    num_nosubhalo = np.sum(~mask_subhalo)
    
    print(f"number of unresolved halos: {num_unresolved}")
    print(f"number of halos with M_crit200 = 0: {num_M200_zero}")
    print(f"number of halos with R_crit200 = 0: {num_R200_zero}")
    print(f"number of halos with no subhalo: {num_nosubhalo}")
    #print(halos['GroupMass'][~mask_M200])

    
    mask = mask_groupmass & mask_M200 & mask_R200 & mask_subhalo
    N_selected = np.sum(mask)
    print("number of selected halos: ", N_selected)
    index_selected = np.where(mask)[0]
    
    AllTNGData = ProcessedTNGData()
    AllTNGData.header = header
    
    
    #basics of selected host halos
    M_all = halos['GroupMass'][index_selected] * 1e10  # Msun/h
    M_crit200_all = halos['Group_M_Crit200'][index_selected] * 1e10  # Msun/h
    R_crit200_all = halos['Group_R_Crit200'][index_selected] / 1e3 * scale_factor / h_Hubble  # Mpc
    R_crit200_m_all = R_crit200_all * Mpc  # meters
    group_vel_all = halos['GroupVel'][index_selected] * 1e3 / scale_factor  # m/s, shape (N_selected, 3)
    M_gas_all = halos['GroupMassType'][index_selected][:, 0] * 1e10  # Msun/h
    gas_metallicity_host_all = halos['GroupGasMetallicity'][index_selected]
    
    #derived quantities of host halos
    vel_host_all = np.sqrt(np.sum(group_vel_all**2, axis=1))  # m/s
    rho_halo_all = (M_crit200_all * Msun/h_Hubble) / (4/3 * np.pi * R_crit200_m_all**3)  # kg/m^3
    Tvir_host_all = np.array([Temperature_Virial_numerical(m/h_Hubble, r) 
                             for m, r in zip(M_crit200_all, R_crit200_all)])  # K
    Cs_host_all = np.sqrt(5.0/3.0 * kB * Tvir_host_all / (mu*mp))  # m/s
    t_ff_all = freefall_factor / np.sqrt(G_grav * rho_halo_all)  # s
    
    #add to AllTNGData
    AllTNGData.add_halo_quantity('original_index', index_selected, 'dimensionless','Original index in TNG data',dtype=np.int32)
    AllTNGData.add_halo_quantity('GroupMass', M_all, 'Msun/h', 'Total mass of the halo')
    AllTNGData.add_halo_quantity('Group_M_Crit200', M_crit200_all, 'Msun/h', 'M200')
    AllTNGData.add_halo_quantity('Group_R_Crit200', R_crit200_all, 'Mpc', 'R200')
    AllTNGData.add_halo_quantity('GroupVel', group_vel_all, 'm/s', 'Velocity of host halo')
    AllTNGData.add_halo_quantity('GroupVelMag', vel_host_all, 'm/s', 'Magnitude of velocity of host halo')
    AllTNGData.add_halo_quantity('GroupGasMetallicity', gas_metallicity_host_all, 'dimensionless', 'Gas metallicity of host halo')
    AllTNGData.add_halo_quantity('GroupTvir', Tvir_host_all, 'K', 'Virial temperature of host halo')
    AllTNGData.add_halo_quantity('GroupCs', Cs_host_all, 'm/s', 'Sound speed of host halo')
    AllTNGData.add_halo_quantity('Group_t_ff', t_ff_all, 's', 'Free-fall time of host halo')
    
    # print("AllTNGData.halo_data['GroupMass']: ", AllTNGData.halo_data['GroupMass'])
    #print(AllTNGData.halo_data['GroupMass'].value)
    
    # find all suitable subhalos of selected host halos
    all_subhalo_indices = []
    host_indices_for_subs = []  # index of host halo for each subhalo
    
    #i: index of host halo in index_selected; index_host: index of host halo in the original TNG data
    for i, index_host in enumerate(index_selected):  
        first_sub = int(halos['GroupFirstSub'][index_host])
        num_subs = int(halos['GroupNsubs'][index_host])
        # skip the first subhalo (central subhalo)
        sub_indices = range(first_sub + 1, first_sub + num_subs)
        
        #check conditions for subhalos
        for j in sub_indices:
            subhalo_mass = subhalos['SubhaloMass'][j] * 1e10 # Msun/h
            host_mass = M_all[i]
            if (subhalo_mass >= 100 * dark_matter_resolution) and (subhalo_mass / host_mass < 1):
                all_subhalo_indices.append(j)
                host_indices_for_subs.append(i)
                
    all_subhalo_indices = np.array(all_subhalo_indices)
    host_indices_for_subs = np.array(host_indices_for_subs)
    n_selected_subs = len(all_subhalo_indices)
    print(f"number of selected subhalos: {n_selected_subs}")
    
    #now get the basic properties of selected subhalos
    sub_mass_all = subhalos['SubhaloMass'][all_subhalo_indices] * 1e10  # Msun/h
    sub_halfrad_all = subhalos['SubhaloHalfmassRad'][all_subhalo_indices] / 1e3 * scale_factor / h_Hubble * Mpc  # meters
    sub_vmaxrad_all = subhalos['SubhaloVmaxRad'][all_subhalo_indices] / 1e3 * scale_factor / h_Hubble * Mpc  # meters
    sub_vel_all = subhalos['SubhaloVel'][all_subhalo_indices] * 1e3  # m/s, shape (n_selected_subs, 3)
    sub_metallicity_all = subhalos['SubhaloGasMetallicity'][all_subhalo_indices] # dimensionless

    #also get the corresponding host halo properties
    host_vel_for_subs = group_vel_all[host_indices_for_subs]  # m/s
    host_cs_for_subs = Cs_host_all[host_indices_for_subs]  # m/s
    host_tff_for_subs = t_ff_all[host_indices_for_subs]  # s

    
    #derived quantities of subhalos
    rel_vel_all = sub_vel_all - host_vel_for_subs
    rel_vel_mag_all = np.sqrt(np.sum(rel_vel_all**2, axis=1))  # m/s
    mach_number_all = rel_vel_mag_all / host_cs_for_subs
    
    sub_vmaxrad_tcross_all = crossing_time(host_cs_for_subs, sub_vmaxrad_all)  # s    
    A_number_all = np.array([get_A_number(0.5 * m * Msun/h_Hubble, cs, rad)
                        for m, cs, rad in zip(sub_mass_all, host_cs_for_subs, sub_halfrad_all)])

    
    #debug I_DF later 
    '''          
    I_DF = 0.0
    if Mach_rel <= 1:
        I_DF = Idf_Ostriker99_wrapper(Mach_rel, 0, Cs_host, t_dyn)
        t_evaluate = t_dyn
    else:
        rmin = 2.25*subhalo_halfrad  #Sánchez-Salcedo & Brandenburg 1999
        t_evaluate = t_dyn
        #check if rmin and t_evaluate satisfy the condition in Ostriker99, i.e. V*t - Cs*t > rmin
        if (vel*t_evaluate - Cs_host*t_evaluate <= rmin):
            print("Warning: vel*t - Cs*t <= rmin")
            #reset t_evaluate
            t_evaluate = rmin/(vel-Cs_host)*1.1
        
        I_DF = Idf_Ostriker99_wrapper(Mach_rel, rmin, Cs_host, t_evaluate)
        
    #compare Cs*t with host halo radius
    if (Cs_host*t_evaluate > R_crit200_m):
        print("Warning: Cs*t > R_crit200")
        print(f"Cs_host*t_evaluate: {Cs_host*t_evaluate}, R_crit200: {R_crit200_m}")
        print(f"Cs_host: {Cs_host}, t_evaluate: {t_evaluate/Myr} Myr")
        #debug: to be checked
    '''
    rho_g_analytic = rho_b0*(1+current_redshift)**3 *Msun/Mpc**3
    rho_m_analytic = rho_m0*(1+current_redshift)**3 *Msun/Mpc**3
    rho_g_analytic_200 = 200 *rho_g_analytic
    DF_heating_withoutIDF = 4 * np.pi * (G_grav * sub_mass_all * Msun/h_Hubble) ** 2 / rel_vel_mag_all * rho_g_analytic_200
    
    #add to AllTNGData
    AllTNGData.add_subhalo_quantity('SubMass', sub_mass_all, 'Msun/h', 'Subhalo mass')
    AllTNGData.add_subhalo_quantity('SubHalfmassRad', sub_halfrad_all, 'm', 'Half-mass radius')
    AllTNGData.add_subhalo_quantity('SubVmaxRad', sub_vmaxrad_all, 'm', 'Maximum velocity radius')
    AllTNGData.add_subhalo_quantity('SubVel', sub_vel_all, 'm/s', 'Subhalo velocity') #shape (n_selected_subs, 3)
    AllTNGData.add_subhalo_quantity('SubGasMetallicity', sub_metallicity_all, 'dimensionless', 'Gas metallicity')

    # add indices
    AllTNGData.add_subhalo_quantity('host_index', host_indices_for_subs, 'dimensionless', 'Index in the selected host array',dtype='int32')
    AllTNGData.add_subhalo_quantity('original_index', all_subhalo_indices, 'dimensionless', 'Original index in TNG data',dtype='int32')
    
    # add derived quantities
    AllTNGData.add_subhalo_quantity('relative_velocity', rel_vel_all, 'm/s', 'Relative velocity to host') # shape (n_selected_subs, 3)
    AllTNGData.add_subhalo_quantity('relative_velocity_magnitude', rel_vel_mag_all, 'm/s', 'Magnitude of relative velocity')
    AllTNGData.add_subhalo_quantity('mach_number', mach_number_all, 'dimensionless', 'Mach number')
    AllTNGData.add_subhalo_quantity('vmaxrad_tcross', sub_vmaxrad_tcross_all, 's', 'Crossing time at Vmax radius')
    AllTNGData.add_subhalo_quantity('host_t_ff', host_tff_for_subs, 's', 'Free-fall time of host halo')
    AllTNGData.add_subhalo_quantity('A_number', A_number_all, 'dimensionless', 'A number')
    AllTNGData.add_subhalo_quantity('DF_heating_withoutIDF', DF_heating_withoutIDF, 'J/s', 'DF heating rate without I_DF factor')

    # 
    # AllTNGData.metadata.update({
    #     'rho_gas_analytic': float(rho_g_analytic),
    #     'rho_matter_analytic': float(rho_m_analytic),
    #     'rho_gas_200': float(rho_g_analytic_200)
    # })
        
    processed_file = os.path.join(output_dir, f'processed_halos_snap_{snapNum}.h5')
    save_processed_data(processed_file, AllTNGData)
    
    
    
def analyze_processed_data():
    # Construct the path
    base_dir = '/home/zwu/21cm_project/unified_model/TNG_results/'
    processed_file = os.path.join(base_dir, simulation_set, f'snap_{snapNum}', 
                                f'processed_halos_snap_{snapNum}.h5')
    
    # Load the data
    data = load_processed_data(processed_file)
    
    # Example: Access host halo properties
    print("\nHost Halo Statistics:")
    print(f"Number of host halos: {len(data.halo_data['GroupMass'].value)}")
    print(f"Mass range: {data.halo_data['GroupMass'].value.min():.2e} - "
          f"{data.halo_data['GroupMass'].value.max():.2e} {data.halo_data['GroupMass'].units}")
    
    # Example: Access subhalo properties
    print("\nSubhalo Statistics:")
    print(f"Number of subhalos: {len(data.subhalo_data['SubMass'].value)}")
    print(f"Mass range: {data.subhalo_data['SubMass'].value.min():.2e} - "
          f"{data.subhalo_data['SubMass'].value.max():.2e} {data.subhalo_data['SubMass'].units}")
    
def find_abnormal_mach():
    #load the processed data
    base_dir = '/home/zwu/21cm_project/unified_model/TNG_results/'
    processed_file = os.path.join(base_dir, simulation_set, f'snap_{snapNum}', 
                                f'processed_halos_snap_{snapNum}.h5')
    data = load_processed_data(processed_file)
    
    mach_numbers = data.subhalo_data['mach_number'].value
    host_indices = data.subhalo_data['host_index'].value
    GroupCs_subhalo = data.halo_data['GroupCs'].value[host_indices]
    host_vel_mag = data.halo_data['GroupVelMag'].value[host_indices]
    rel_vel_mag = data.subhalo_data['relative_velocity_magnitude'].value
    sub_vel = data.subhalo_data['SubVel'].value
    sub_vel_mag = np.sqrt(np.sum(sub_vel**2, axis=1))
    
    #print some statistics
    print(f"Number of subhalos: {len(mach_numbers)}")
    print(f"Number of subhalos with Mach number > 5: {np.sum(mach_numbers>5)}")
    print("GroupCs_subhalo: ", GroupCs_subhalo)
    print("rel_vel_mag: ", rel_vel_mag)
    print("sub_vel_mag: ", sub_vel_mag)
    print("host_vel_mag: ", host_vel_mag)
    
    print("\n\ncheck mach numbers > 5 ...")
    mask = mach_numbers > 5
    print("mach_numbers[mask]: ", mach_numbers[mask])
    print("host_indices[mask]: ", host_indices[mask])
    print("GroupCs_subhalo[mask]: ", GroupCs_subhalo[mask])
    print("rel_vel_mag[mask]: ", rel_vel_mag[mask])
    print("sub_vel_mag[mask]: ", sub_vel_mag[mask])
    print("host_vel_mag[mask]: ", host_vel_mag[mask])
    
    #plot the Cs, vel distribution of normal cases and abnormal cases (1D histogram)
    mask_normal = ~mask
    mask_abnormal = mask
    
    #Cs distribution
    fig = plt.figure(figsize=(8, 6), facecolor='white')
    plt.hist(GroupCs_subhalo[mask_normal]/1e3, bins=50, alpha=0.5, density='True',label='Normal cases')
    plt.hist(GroupCs_subhalo[mask_abnormal]/1e3, bins=50, alpha=0.5,density='True',label='Abnormal cases')
    plt.xlabel('Cs (km/s)')
    plt.ylabel('PDF')
    plt.legend()
    plt.title('Sound speed distribution')
    plt.tight_layout()
    plt.savefig('debug/Cs_distribution.png')
    
    #relative velocity distribution
    fig = plt.figure(figsize=(8, 6), facecolor='white')
    plt.hist(rel_vel_mag[mask_normal]/1e3, bins=50, alpha=0.5,label='Normal cases')
    plt.hist(rel_vel_mag[mask_abnormal]/1e3, bins=50, alpha=0.5,label='Abnormal cases')
    plt.xlabel('Relative velocity (km/s)')
    plt.ylabel('counts')
    plt.legend()
    plt.title('Relative velocity distribution')
    plt.tight_layout()
    plt.savefig('debug/rel_vel_distribution_counts.png')
    
    fig = plt.figure(figsize=(8, 6), facecolor='white')
    plt.hist(rel_vel_mag[mask_normal]/1e3, bins=50, alpha=0.5,density='True',label='Normal cases')
    plt.hist(rel_vel_mag[mask_abnormal]/1e3, bins=50, alpha=0.5,density='True',label='Abnormal cases')
    plt.xlabel('Relative velocity (km/s)')
    plt.ylabel('PDF')
    plt.legend()
    plt.title('Relative velocity distribution')
    plt.tight_layout()
    plt.savefig('debug/rel_vel_distribution.png')
    
    

if __name__ == '__main__':
    #TNG_model()
    #analyze_processed_data()
    find_abnormal_mach()
    
    