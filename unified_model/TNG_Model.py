import numpy as np
import illustris_python as il
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from physical_constants import *
from HaloProperties import *
from Config import simulation_set, p_evolved, p_unevolved
from DF_Ostriker99_wake_structure import Idf_Ostriker99_wrapper
from TNGDataHandler import *
import Xray_field as xray
from HaloMassFunction import plot_hmf, fitFunc_lg_dNdlgx, get_cumulativeSHMF_sigma_correction

import argparse
parser = argparse.ArgumentParser(description="Run TNG_model with a specified snapNum.")
parser.add_argument("snapNum", type=int, help="The snapshot number to process.")
args = parser.parse_args()
snapNum = args.snapNum
print(f"Processing snapNum = {snapNum}")

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
        # 'Group_M_Mean200',
        # 'Group_R_Mean200'
    ]
    # Load halo data (GroupFirstSub and GroupNsubs are integer fields, keep as is)
    halos = il.groupcat.loadHalos(basePath, snapNum, 
                                 fields=['GroupFirstSub', 'GroupNsubs'] + halo_float_fields)
    
    print("loading subhalos ...")
    # Define subhalo fields that need to be converted to float64
    subhalo_float_fields = [
        'SubhaloFlag',
        'SubhaloMass',
        'SubhaloMassInMaxRad', #Sum of masses of all particles/cells within the radius of Vmax
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

    gas_resolution, dark_matter_resolution = get_simulation_resolution(simulation_set)
    
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
    simulation_volume = (header['BoxSize']/1e3 * scale_factor) **3  # (Mpc/h)^3

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
    M_dm_all = halos['GroupMassType'][index_selected][:, 1] * 1e10  # Msun/h
    M_dmlowres_all = halos['GroupMassType'][index_selected][:, 2] * 1e10  # Msun/h
    M_tracer_all = halos['GroupMassType'][index_selected][:, 3] * 1e10  # Msun/h
    M_starwind_all = halos['GroupMassType'][index_selected][:, 4] * 1e10  # Msun/h
    M_bh_all = halos['GroupMassType'][index_selected][:, 5] * 1e10  # Msun/h
    gas_metallicity_host_all = halos['GroupGasMetallicity'][index_selected]
    
    #derived quantities of host halos
    vel_host_all = np.sqrt(np.sum(group_vel_all**2, axis=1))  # m/s
    rho_halo_all = (M_crit200_all * Msun/h_Hubble) / (4/3 * np.pi * R_crit200_m_all**3)  # kg/m^3
    Tvir_host_all = np.array([Temperature_Virial_numerical(m/h_Hubble, r) 
                             for m, r in zip(M_crit200_all, R_crit200_all)])  # K
    '''
    #check analytical version
    Tvir_host_all_analytic_old = np.array([Temperature_Virial_analytic_oldversion(m/h_Hubble, current_redshift)
                                for m in M_crit200_all])  # K

    Tvir_host_all_analytic = np.array([Temperature_Virial_analytic(m/h_Hubble, current_redshift)
                                for m in M_crit200_all])  # K
    print("Tvir_host_all_analytic: ", Tvir_host_all_analytic)
    print("Tvir_host_all_analytic_old: ", Tvir_host_all_analytic_old)
    print("Tvir_host_all: ", Tvir_host_all)
    exit()
    '''
    
    Cs_host_all = np.sqrt(5.0/3.0 * kB * Tvir_host_all / (mu*mp))  # m/s
    t_ff_all = freefall_factor / np.sqrt(G_grav * rho_halo_all)  # s
    #add to AllTNGData
    AllTNGData.add_halo_quantity('original_index', index_selected, 'dimensionless','Original index in TNG data',dtype=np.int32)
    AllTNGData.add_halo_quantity('GroupMass', M_all, 'Msun/h', 'Total mass of the halo')
    AllTNGData.add_halo_quantity('GroupPos', halos['GroupPos'][index_selected], 'ckpc/h', 'Position of the halo')
    AllTNGData.add_halo_quantity('GroupGasMass', M_gas_all, 'Msun/h', 'Gas mass of the halo')
    AllTNGData.add_halo_quantity('GroupDMmass', M_dm_all, 'Msun/h', 'Dark matter mass of the halo')
    AllTNGData.add_halo_quantity('GroupStellarMass', M_starwind_all, 'Msun/h', 'Star and wind partile mass of the halo')
    AllTNGData.add_halo_quantity('GroupBHMass', M_bh_all, 'Msun/h', 'Black hole mass of the halo')
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
            subhaloflag = subhalos['SubhaloFlag'][j]
            #flag 0 (very rare): not of cosmological origin, may be a fragment of disk identified by SUBFIND; 1: galaxy or satellite
            if (subhalo_mass >= 50*dark_matter_resolution) and (subhalo_mass/host_mass < 1) and (subhaloflag == 1):
                all_subhalo_indices.append(j)
                host_indices_for_subs.append(i)
                
    all_subhalo_indices = np.array(all_subhalo_indices)
    host_indices_for_subs = np.array(host_indices_for_subs)
    n_selected_subs = len(all_subhalo_indices)
    print(f"number of selected subhalos: {n_selected_subs}")
    
    #now get the basic properties of selected subhalos
    sub_mass_all = subhalos['SubhaloMass'][all_subhalo_indices] * 1e10  # Msun/h
    # sub_massinvmaxrad_all = subhalos['SubhaloMassInMaxRad'][all_subhalo_indices] * 1e10  # Msun/h
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
    #fid: divided by Cs instead of rel_vel_mag, I_DF = 1, assume rho_gas = 200*rho_crit(baryon, z)
    DF_heating_fid = 4 * np.pi * (G_grav * sub_mass_all * Msun/h_Hubble) ** 2 / host_cs_for_subs * rho_g_analytic_200

    #plot full hmf and selected halos
    hmf_filename = os.path.join(output_dir, 'analysis', f'HMF_snap_{snapNum}.png')
    if not os.path.exists(os.path.join(output_dir, 'analysis')):
        os.makedirs(os.path.join(output_dir, 'analysis'))

    plot_hmf(halos, index_selected, current_redshift, dark_matter_resolution, simulation_volume, hmf_filename)

    #add to AllTNGData
    AllTNGData.add_subhalo_quantity('SubMass', sub_mass_all, 'Msun/h', 'Subhalo mass')
    # AllTNGData.add_subhalo_quantity('SubMassInMaxRad', sub_massinvmaxrad_all, 'Msun/h', 'Subhalo mass in Vmax radius')
    AllTNGData.add_subhalo_quantity('SubPos', subhalos['SubhaloPos'][all_subhalo_indices], 'ckpc/h', 'Position of subhalo')
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
    AllTNGData.add_subhalo_quantity('DF_heating_fid', DF_heating_fid, 'J/s', 'DF heating rate without I/Mach correction and rho_gas = 200*rho_crit(baryon, z)')
    # 
    # AllTNGData.metadata.update({
    #     'rho_gas_analytic': float(rho_g_analytic),
    #     'rho_matter_analytic': float(rho_m_analytic),
    #     'rho_gas_200': float(rho_g_analytic_200)
    # })

    processed_file = os.path.join(output_dir, f'processed_halos_snap_{snapNum}.h5')
    save_processed_data(processed_file, AllTNGData)

    #check subhalo flags
    subhalo_flags = subhalos['SubhaloFlag'][all_subhalo_indices]
    print("# of selected subhalos: ", len(subhalo_flags))
    print("# of flag = 0: ", np.sum(subhalo_flags == 0))
    print("# of flag = 1: ", np.sum(subhalo_flags == 1))

def plot_Mratio_dN_dlogMratio():

    base_dir = '/home/zwu/21cm_project/unified_model/TNG_results/'
    processed_file = os.path.join(base_dir, simulation_set, f'snap_{snapNum}', 
                                f'processed_halos_snap_{snapNum}.h5')
    data = load_processed_data(processed_file)
    output_dir = os.path.join(base_dir, simulation_set, f'snap_{snapNum}','analysis')

    # Get required quantities
    #test different definitions of subhalo mass and host mass
    sub_masses = data.subhalo_data['SubMass'].value  # Msun/h
    # sub_masses = data.subhalo_data['SubMassInMaxRad'].value  # Msun/h
    host_indices = data.subhalo_data['host_index'].value
    host_masses = data.halo_data['GroupMass'].value[host_indices]  # Msun/h
    # host_masses = data.halo_data['Group_M_Crit200'].value[host_indices] # Msun/h
    mass_ratios = sub_masses / host_masses
    host_logM = np.log10(host_masses)

    # Get simulation parameters
    current_redshift = data.header['Redshift']
    gas_resolution, dark_matter_resolution = get_simulation_resolution(simulation_set)

    # Divide the host halos into 5 mass bins, and plot the distribution of m/M for each bin respectively
    logM_min = np.min(host_logM)
    logM_max = np.max(host_logM)
    num_M_bins = 5
    logM_bins = np.linspace(logM_min, logM_max, num=num_M_bins+1)

    #initialize lists
    sub_host_Mratio_list_old = [] #the old list which doesn't deal with individual host halos
    sub_host_Mratio_list = []
    cumulative_sub_host_Mratio_list = []
    num_host_list = []
    critical_ratio_list = []
    subhalo_resolution = 50*dark_matter_resolution 

    # Create histogram bins
    num_ratio_bins = 50
    bins = np.linspace(-4, 0, num_ratio_bins+1) #(-4, 0) for high-z, (-5, 0) for low-z
    bin_edges = bins
    log_bin_widths = bins[1] - bins[0]
    artificial_small = 1e-10
    min_number_density = 1e10 #used to set the lower limit of y-axis for plotting


    # Loop over mass bins
    for i in range(num_M_bins):
        mask = (host_logM >= logM_bins[i]) & (host_logM < logM_bins[i+1])
        sub_host_Mratio_list_old.append(mass_ratios[mask])
        unique_host_indices = np.unique(host_indices[mask])
        len_unique_hosts = len(unique_host_indices)
        num_host_list.append(len_unique_hosts)
        #define critical ratio = subhalo_resolution/Mhost(left edge of bin)
        critical_ratio_list.append(subhalo_resolution/10**logM_bins[i])
        print(f"Number of host halos in bin {i}: {len_unique_hosts}")
        print(f"number of subhalos in bin {i}: {len(mass_ratios[mask])}")

        #individual SHMF
        sub_host_Mratio_matrix = np.zeros((len_unique_hosts, num_ratio_bins))
        for j, host_index in enumerate(unique_host_indices):
            host_subhalo_mask = (host_indices == host_index) & mask
            sub_host_Mratio_matrix[j, :], _ = np.histogram(np.log10(mass_ratios[host_subhalo_mask]), 
                                                            bins=bins)
        sub_host_Mratio_list.append(sub_host_Mratio_matrix)

        cumulative_matrix = np.cumsum(sub_host_Mratio_matrix[:, ::-1], axis=1)[:, ::-1]
        cumulative_sub_host_Mratio_list.append(cumulative_matrix)
    
    tot_num_host = len(np.unique(host_indices))
    print(f"Total number of host halos: {tot_num_host}")  #plot tot distribution separately




    # Plot mass ratio distributions
    colors = plt.cm.rainbow(np.linspace(0, 1, num_M_bins))
    labels = [f'[{logM_bins[i]:.2f}, {logM_bins[i+1]:.2f}]' for i in range(num_M_bins)]

    fig = plt.figure(facecolor='white')
    ax = fig.gca()
    number_density_list = []
    for i in range(num_M_bins):
        # counts, bin_edges = np.histogram(np.log10(sub_host_Mratio_list_old[i]), bins=bins)
        # number_density = counts/log_bin_widths/num_host_list[i]
        sub_host_Mratio_matrix = sub_host_Mratio_list[i]
        #sum the rows (all hosts)
        number_density = np.sum(sub_host_Mratio_matrix, axis=0) / log_bin_widths / num_host_list[i]

        min_number_density = min(min_number_density, np.min(number_density[number_density > 0]))
        
        # Handle zero counts
        mask = (number_density == 0)
        number_density[mask] = artificial_small
        
        plt.step(bin_edges[:-1], number_density, where='post', 
                color=colors[i], label=labels[i])
                
        
        number_density_list.append(number_density)
        plt.axvline(np.log10(critical_ratio_list[i]), 
                    color=colors[i], linestyle='--')
        
    # Add van den Bosch+ 2016 fitting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    JB16_evolved_lg_number_density = fitFunc_lg_dNdlgx(bin_centers, *p_evolved)
    plt.plot(bin_centers, 10**JB16_evolved_lg_number_density, linestyle='-',
             color='grey', label='Jiang & van den Bosch 2016, evolved')
    
    JB16_unevolved_lg_number_density = fitFunc_lg_dNdlgx(bin_centers, *p_unevolved)
    plt.plot(bin_centers, 10**JB16_unevolved_lg_number_density, linestyle='--',
             color='grey', label='Jiang & van den Bosch 2016, unevolved')
    #now calculate bestfit parameters with fitFunc_lg_dNdlgx
    #combine data points from all bins with mass ratio > their critical ratio and exclude artificial small
    all_bin_centers = []
    all_number_density = []
    for i in range(num_M_bins):
        number_density = number_density_list[i]
        fit_mask = (10**bin_centers > critical_ratio_list[i]) & (number_density > artificial_small)
        all_bin_centers.extend(bin_centers[fit_mask])
        all_number_density.extend(number_density[fit_mask])
    all_bin_centers = np.array(all_bin_centers)
    all_number_density = np.array(all_number_density)
    #sort the data points
    sort_indices = np.argsort(all_bin_centers)
    all_bin_centers_sorted = all_bin_centers[sort_indices]
    all_number_density_sorted = all_number_density[sort_indices]
    #fit the data points
    p_bestfit, pcov = curve_fit(fitFunc_lg_dNdlgx, all_bin_centers_sorted, np.log10(all_number_density_sorted), p0=p_unevolved)
    print("BestFit parameters: ",p_bestfit)
    bestfit_lg_number_density = fitFunc_lg_dNdlgx(bin_centers, *p_bestfit)
    plt.plot(bin_centers, 10**bestfit_lg_number_density, linestyle='-',color='black',label='BestFit')


    # Finalize plot
    plt.ylim(bottom=min_number_density/10)
    #xlim: > 1e-4 or > 1e-5
    plt.xlim([-4.1,0])
    plt.legend(loc='lower left')
    plt.xlabel(r'$\lg$($\psi$) = $\lg$(m/M)',fontsize=14)
    plt.ylabel(r'dN/d$\lg(\psi)$',fontsize=14)
    plt.yscale('log')
    ax.tick_params(direction='in', which='both', labelsize=12)
    plt.tight_layout()
    # Save plot
    plt.savefig(os.path.join(output_dir, f'SHMF_snap_{snapNum}_new.png'), dpi=300)

    # Save bestfit parameters
    bestfit_params_file = os.path.join(output_dir, f'SHMF_BestFit_params_snap_{snapNum}.txt')
    with open(bestfit_params_file, 'w') as f:
        header = ['alpha', 'beta/ln10', 'omega', 'lgA', 'beta', 'A']
        column_width = 12
        formatted_header = ' '.join(f'{h:<{column_width}}' for h in header)

        alpha = p_bestfit[0]
        beta_ln10 = p_bestfit[1]
        omega = p_bestfit[2]
        lgA = p_bestfit[3]
        beta = p_bestfit[1] * np.log(10)
        A = 10 ** p_bestfit[3]

        values = [alpha, beta_ln10, omega, lgA, beta, A]
        formatted_values = ' '.join(f'{v:{column_width}.5e}' for v in values)
        f.write('BestFit parameters:\n')  
        f.write(formatted_header + '\n') 
        f.write(formatted_values + '\n')  

    #calculate the cumulative shmf for the bestfit parameters
    N_bestfit = 10**bestfit_lg_number_density * log_bin_widths
    cumulative_bestfit = np.cumsum(N_bestfit[::-1])[::-1]

    #Now plot the cumulative subhalo mass function (no Poisson correction)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    for i in range(num_M_bins):
        cumulative_matrix = cumulative_sub_host_Mratio_list[i]
        mean_cumulative = np.mean(cumulative_matrix, axis=0) #mean over all hosts in this host mass bin
        var_cumulative = np.var(cumulative_matrix, axis=0)
        std_cumulative = np.sqrt(var_cumulative)
        resolved_mask = bins[:-1] >= np.log10(critical_ratio_list[i])
        plt.errorbar(bins[:-1][resolved_mask], mean_cumulative[resolved_mask], 
                    yerr=std_cumulative[resolved_mask],
                    fmt='o-', color=colors[i], label=labels[i], 
                    markersize=4, capsize=3, alpha=0.8)
        
        # plt.plot(bins[:-1][resolved_mask], mean_cumulative[resolved_mask], 'o-', color=colors[i], label=labels[i], markersize=4, linewidth=2)

        std_Poisson = np.sqrt(mean_cumulative)  # Poisson error
        print(f"Mean cumulative for bin {i}: {mean_cumulative}")
        one_sigma_upper = mean_cumulative + std_Poisson
        one_sigma_lower = mean_cumulative - std_Poisson
        # 1-sigma shaded region
        # plt.fill_between(bins[:-1][resolved_mask],one_sigma_lower[resolved_mask], one_sigma_upper[resolved_mask],
        #                  color=colors[i], alpha=0.15)
        plt.plot(bins[:-1][resolved_mask], one_sigma_upper[resolved_mask], linestyle='--', color=colors[i], alpha=0.7, linewidth=1)
        plt.plot(bins[:-1][resolved_mask], one_sigma_lower[resolved_mask], linestyle='--', color=colors[i], alpha=0.7, linewidth=1)

        plt.axvline(np.log10(critical_ratio_list[i]), color=colors[i], linestyle='--', alpha=0.8)
    plt.plot(bins[:-1], cumulative_bestfit, linestyle='-', color='black', label='BestFit')
    ax.set_xlabel(r'$\lg(\psi)$, $\psi = m/M$', fontsize=14)
    ax.set_ylabel(r'$N(>\psi)$', fontsize=14)
    ax.set_xlim([-4.1, 0])
    ax.set_ylim(bottom=1e-3)
    ax.set_yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    filename = os.path.join(output_dir, f'SHMF_cumulative_snap_{snapNum}.png')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Cumulative SHMF plot saved to {filename}")



    #Now plot the cumulative subhalo mass function (with Poisson correction)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    for i in range(num_M_bins):
        cumulative_matrix = cumulative_sub_host_Mratio_list[i]
        mean_cumulative = np.mean(cumulative_matrix, axis=0) #mean over all hosts in this host mass bin
        var_cumulative = np.var(cumulative_matrix, axis=0)
        std_cumulative = np.sqrt(var_cumulative)
        resolved_mask = bins[:-1] >= np.log10(critical_ratio_list[i])
        plt.errorbar(bins[:-1][resolved_mask], mean_cumulative[resolved_mask], 
                    yerr=std_cumulative[resolved_mask],
                    fmt='o-', color=colors[i], label=labels[i], 
                    markersize=4, capsize=3, alpha=0.8)
        
        # plt.plot(bins[:-1][resolved_mask], mean_cumulative[resolved_mask], 'o-', color=colors[i], label=labels[i], markersize=4, linewidth=2)

        std_Poisson = np.sqrt(mean_cumulative)  # Poisson error
        Poisson_corr = get_cumulativeSHMF_sigma_correction(mean_cumulative,'supersubPoisson')
        print(f"Mean cumulative for bin {i}: {mean_cumulative}")
        print(f"Poisson correction factor for bin {i}: {Poisson_corr}")
        std_Poissoncorr = std_Poisson * Poisson_corr  # corrected Poisson error
        one_sigma_upper = mean_cumulative + std_Poissoncorr
        one_sigma_lower = mean_cumulative - std_Poissoncorr
        # 1-sigma shaded region
        # plt.fill_between(bins[:-1][resolved_mask],one_sigma_lower[resolved_mask], one_sigma_upper[resolved_mask],
        #                  color=colors[i], alpha=0.15)
        plt.plot(bins[:-1][resolved_mask], one_sigma_upper[resolved_mask], linestyle='--', color=colors[i], alpha=0.7, linewidth=1)
        plt.plot(bins[:-1][resolved_mask], one_sigma_lower[resolved_mask], linestyle='--', color=colors[i], alpha=0.7, linewidth=1)

        plt.axvline(np.log10(critical_ratio_list[i]), color=colors[i], linestyle='--', alpha=0.8)
    plt.plot(bins[:-1], cumulative_bestfit, linestyle='-', color='black', label='BestFit')
    ax.set_xlabel(r'$\lg(\psi)$, $\psi = m/M$', fontsize=14)
    ax.set_ylabel(r'$N(>\psi)$', fontsize=14)
    ax.set_xlim([-4.1, 0])
    ax.set_ylim(bottom=1e-3)
    ax.set_yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    filename = os.path.join(output_dir, f'SHMF_cumulative_Poissoncorr_snap_{snapNum}.png')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Cumulative SHMF plot saved to {filename}")



    #plot the ratio between sigma and sigma_Poisson vs mean cumulative
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    for i in range(num_M_bins):
        cumulative_matrix = cumulative_sub_host_Mratio_list[i]
        mean_cumulative = np.mean(cumulative_matrix, axis=0)  # mean over all hosts in this host mass bin
        var_cumulative = np.var(cumulative_matrix, axis=0)
        std_cumulative = np.sqrt(var_cumulative)
        resolved_mask = bins[:-1] >= np.log10(critical_ratio_list[i])
        
        std_Poisson = np.sqrt(mean_cumulative)
        mask = (std_Poisson > 0) & (std_cumulative > 0) & resolved_mask
        ratio_masked = std_cumulative[mask] / std_Poisson[mask]
        mean_cumulative_masked = mean_cumulative[mask]
        plt.plot(mean_cumulative_masked, ratio_masked, 'o-', color=colors[i], label=labels[i], markersize=4, linewidth=2)
        
    #also plot correction from JB paperIII
    N_list = np.logspace(-3, 2, num=1000)  # N(>psi)
    JB_corrections = get_cumulativeSHMF_sigma_correction(N_list, 'supersubPoisson')
    plt.plot(N_list, JB_corrections, linestyle='--', color='grey', label='fit of Jiang&van den Bosch paperIII', linewidth=2)
    plt.axhline(1, color='black', linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel(r'$N(>\psi)$', fontsize=14)
    ax.set_ylabel(r'$\sigma / \sigma_{P}$', fontsize=14)
    ax.set_xlim(left=10**(-3.1))
    ax.set_ylim(bottom=0.6, top=2.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    filename = os.path.join(output_dir, f'SHMF_sigma_ratio_snap_{snapNum}.png')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Sigma ratio plot saved to {filename}")



        
   


def analyze_processed_data():
    # Construct the path
    base_dir = '/home/zwu/21cm_project/unified_model/TNG_results/'
    processed_file = os.path.join(base_dir, simulation_set, f'snap_{snapNum}', 
                                f'processed_halos_snap_{snapNum}.h5')
    
    # Load the data
    data = load_processed_data(processed_file)
    redshift = data.header['Redshift']
    print(f"Redshift: {redshift}")
    
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
    

    '''
    #now calculate X-ray emissivity
    lognH = get_gas_lognH_analytic(redshift)
    print("lognH: ", lognH)
    host_indices = data.subhalo_data['host_index'].value
    
    host_Tvir = data.halo_data['GroupTvir'].value[host_indices]
    host_gasmetallicity = data.halo_data['GroupGasMetallicity'].value[host_indices]
    gas_metallicity_Zsun = host_gasmetallicity / Zsun

    e_min = 0.5 # keV
    e_max = 2.0 # keV

    xray_emissivity = xray.calculate_xray_emissivity(lognH, host_Tvir, gas_metallicity_Zsun, 
                                   e_min, e_max, use_metallicity=True, redshift=0.,
                                   table_type='cloudy', data_dir=".", cosmology=None, dist=None)

    print("xray_emissivity: ", xray_emissivity)

    subhalo_volumes = 4/3 * np.pi * data.subhalo_data['SubHalfmassRad'].value**3 #in unit m^3
    subhalo_volumes_cm3 = subhalo_volumes * 1e6

    tot_xray_emissivity = np.sum(xray_emissivity.value * subhalo_volumes_cm3)
    comoving_boxsize = 51.7**3  # cMpc^3
    tot_xray_emissivity_per_cMpc3 = tot_xray_emissivity / comoving_boxsize
    print(f"Total X-ray emissivity: {tot_xray_emissivity_per_cMpc3:.2e} erg/s/cMpc^3")
    '''

if __name__ == '__main__':
    # TNG_model()
    # find_abnormal_mach()
    # analyze_processed_data()
    plot_Mratio_dN_dlogMratio()  
    