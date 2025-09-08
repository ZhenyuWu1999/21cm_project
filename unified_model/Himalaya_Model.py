import yt
import numpy as np
from utilities import read_hdf5_data, display_hdf5_contents
import os
from physical_constants import Msun, h_Hubble
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


from HaloMassFunction import HMF_Colossus
from Config import p_evolved, p_unevolved
from HaloMassFunction import fitFunc_lg_dNdlgx


HIMALAYA_consts = {
    'Omega0': 0.3106,
    'OmegaBaryon': 0.048976,
    'OmegaLambda': 0.6894
}

def read_himalaya(file_path):
    display_hdf5_contents(file_path)
    data_dict, attrs_dict = read_hdf5_data(file_path, include_attrs=True)
    # print(data_dict["Subhalo/SubhaloMass"])
    print(data_dict["Group/Group_M_Crit200"])
    print(attrs_dict)
    # print(attrs_dict.keys())
    #dict_keys(['Config', 'Header', 'Parameters'])

    current_redshift = attrs_dict["Header"]["Redshift"]
    print("current redshift:", current_redshift)
    
    #now read boxsize
    box_size = attrs_dict["Parameters"]["BoxSize"]
    print("Box size:", box_size)

    #debug: ask Balu about h and a factors in these units

    #now read units
    UnitLength_in_cm = attrs_dict["Parameters"]["UnitLength_in_cm"]
    UnitMass_in_g = attrs_dict["Parameters"]["UnitMass_in_g"]
    UnitVelocity_in_cm_per_s = attrs_dict["Parameters"]["UnitVelocity_in_cm_per_s"]
    print("Unit Length in cm:", UnitLength_in_cm)
    print("Unit Mass in g:", UnitMass_in_g)
    print("Unit Velocity in cm/s:", UnitVelocity_in_cm_per_s)
    print("Subhalo Velocity:", data_dict["Subhalo/SubhaloVel"])

    print("Group Mass:", data_dict["Group/GroupMass"])
    print("Group Mass Crit200:", data_dict["Group/Group_M_Crit200"])
    print("Group Pos:", data_dict["Group/GroupPos"])
    #max and min of pos
    print("Group Pos max:", np.max(data_dict["Group/GroupPos"], axis=0))
    print("Group Pos min:", np.min(data_dict["Group/GroupPos"], axis=0))

    #mass resolution? select only resolved halos


def test_file_reading():
    base_dir = "/ceph/cephfs/bsreedhar/HIMALAYA/HIMALAYA_DMO"

    # simulation_dir = os.path.join(base_dir, "zooms/000", "data")
    simulation_dir = os.path.join(base_dir, "parent", "data") #(not compatible with yt format now)
    
    snapNum = 10
    fof_sub_path = os.path.join(simulation_dir, "fof_subhalo_tab_%03d.hdf5" % snapNum)
    snap_path = os.path.join(simulation_dir, "snapshot_%03d.hdf5" % snapNum)
    baksnap_path = os.path.join(simulation_dir, "bak-snapshot_%03d.hdf5" % snapNum)
    sub_desc_path = os.path.join(simulation_dir, "subhalo_desc_%03d.hdf5" % snapNum)
    sub_prog_path = os.path.join(simulation_dir, "subhalo_prog_%03d.hdf5" % snapNum)
    sub_tree_path = os.path.join(simulation_dir, "subhalo_treelink_%03d.hdf5" % snapNum)


    data_dict, attrs_dict = read_hdf5_data(fof_sub_path, include_attrs=True)
    print("Attributes in fof_subhalo_tab file:\n", attrs_dict)
    print("\nKeys in fof_subhalo_tab file:\n", data_dict.keys())
    print(data_dict["Subhalo/SubhaloMass"][0:10])

    ds_group = yt.load(fof_sub_path)
    # print(ds_group.field_list)
    # ad = ds_group.all_data()
    # exit()

    data_sub_desc_dict = read_hdf5_data(sub_desc_path, include_attrs=False)
    print("\nKeys in subhalo_desc file:\n", data_sub_desc_dict.keys())

    data_sub_prog_dict = read_hdf5_data(sub_prog_path, include_attrs=False)
    print("\nKeys in subhalo_prog file:\n", data_sub_prog_dict.keys())

    data_sub_tree_dict = read_hdf5_data(sub_tree_path, include_attrs=False)
    print("\nKeys in subhalo_treelink file:\n", data_sub_tree_dict.keys())


    ds_snapshot = yt.load(snap_path)
    # print("Fields in snapshot file:\n")
    # print(ds_snapshot.field_list)
    # print(sorted(ds_snapshot.derived_field_list))
    # print([f for f in ds_snapshot.field_info])
    ad = ds_snapshot.all_data()
    print(ad["all", "particle_mass"][0:5])
    #zoom000: [1.57956743e+38 1.57956743e+38 1.57956743e+38 ... 1.57956743e+38
    # 1.57956743e+38 1.57956743e+38] g ?
    #parent: 8.08738524e+40 g ?

    data_snap_dict = read_hdf5_data(snap_path, include_attrs=False)
    print("\nKeys in snapshot file:\n", data_snap_dict.keys())
    # dict_keys(['PartType1/Coordinates', 'PartType1/Masses', 'PartType1/ParticleIDs', 'PartType1/Velocities', 
    # 'PartType2/Coordinates', 'PartType2/Masses', 'PartType2/ParticleIDs', 'PartType2/Velocities', 'PartType3/Coordinates', 
    # 'PartType3/Masses', 'PartType3/ParticleIDs', 'PartType3/Velocities', 'PartType4/Coordinates', 'PartType4/Masses', 
    # 'PartType4/ParticleIDs', 'PartType4/Velocities', 'PartType5/Coordinates', 'PartType5/Masses', 'PartType5/ParticleIDs', 'PartType5/Velocities'])
    
    
    print(data_snap_dict["PartType1/Masses"][0:10])
    print(data_snap_dict["PartType2/Masses"][0:10])
    print(data_snap_dict["PartType3/Masses"][0:10])
    print(data_snap_dict["PartType4/Masses"][0:10])
    print(data_snap_dict["PartType5/Masses"][0:10])
    # 5.3779854e-06, 4.3023883e-05, 0.00034419, 0.00275353, 0.17622583

    #parent:
    # dict_keys(['PartType1/Coordinates', 'PartType1/Masses', 'PartType1/ParticleIDs', 'PartType1/Velocities'])
    # 0.00275353 1e10 Msun/h = 8.08738524e+40 g?  or x1e10 Msun/h? 

def get_redshift_lists():
    #read redshift list from a file
    base_dir = "/ceph/cephfs/bsreedhar/HIMALAYA/HIMALAYA_DMO"
    simulation_label = "zooms000"  #or "parent"
    if simulation_label not in ["parent", "zooms000"]:
        raise ValueError("Invalid simulation label: %s" % simulation_label)
    if simulation_label == "parent":
        simulation_dir = os.path.join(base_dir, "parent", "data")
    elif simulation_label == "zooms000":
        simulation_dir = os.path.join(base_dir, "zooms/000", "data")

    results_dir = "/home/zwu/21cm_project/unified_model/HIMALAYA_results"
    redshift_file = os.path.join(results_dir, f"redshift_list_{simulation_label}.txt")

    #find all the files and store their redshifts in the txt file
    snap_files = []
    snapNum_list = []
    for file in os.listdir(simulation_dir):
        if file.startswith("fof_subhalo_tab_") and file.endswith(".hdf5"):
            snap_files.append(file)
            snapNum = int(file.split("_")[-1].split(".")[0])
            snapNum_list.append(snapNum)
    snap_files.sort() #sort by name, which is also by snap number
    snapNum_list.sort()
    print("Snap numbers found:", snapNum_list)
    print("Found %d snap files" % len(snap_files))
    redshift_list = []
    for file in snap_files:
        file_path = os.path.join(simulation_dir, file)
        print("Reading file:", file_path)
        data_dict, attrs_dict = read_hdf5_data(file_path, include_attrs=True)
        current_redshift = attrs_dict["Header"]["Redshift"]
        redshift_list.append(current_redshift)

    print("Redshift list:", redshift_list)
    #save to file (first row: info; starting from second row: snapNum, redshift)
    with open(redshift_file, "w") as f:
        info = simulation_dir + ": snapNum redshift list\n"
        f.write(info)
        for snapNum, z in zip(snapNum_list, redshift_list):
            f.write("%d %.6f\n" % (snapNum, z))
    print("Redshift list saved to:", redshift_file)


def plot_hmf_himalaya(data_dict, index_selected, current_redshift, dark_matter_resolution, simulation_volume, hmf_filename):
    """
    Plot the halo mass function (HMF) from the HIMALAYA simulation data.
    Parameters:
    - data_dict: Dictionary containing the simulation data read from the HDF5 file.
    - index_selected: Indices of the selected halos to be included in the HMF plot.
    - current_redshift: The redshift of the simulation snapshot.
    - dark_matter_resolution: The mass resolution of dark matter particles in the simulation (in Msun/h).
    - simulation_volume: The comoving volume of the simulation box (in (Mpc/h)^3).
    - hmf_filename: The filename to save the HMF plot
    """

    scale_factor = 1.0/(1.+current_redshift)
    comoving_factor = scale_factor**3
    #plot HMF (halo mass function)
    M_all = data_dict['Group/GroupMass']*1e10 * h_Hubble    #unit: Msun/h
    # M_all = halos['Group_M_Crit200']*1e10  #unit: Msun/h
    selected_M_all = M_all[index_selected]
    max_M = np.max(M_all)

    # Create a histogram (logarithmic bins and logarithmic mass)
    bins = np.logspace(np.log10(min(M_all[M_all > 0])), np.log10(max(M_all)), num=50)
    hist, bin_edges = np.histogram(M_all, bins=bins)
    hist_selected, _ = np.histogram(selected_M_all, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Convert counts to number density
    log_bin_widths = np.diff(np.log10(bins))
    number_density = hist / simulation_volume / log_bin_widths
    number_density_selected = hist_selected / simulation_volume / log_bin_widths

    # Plot the mass function
    fig = plt.figure(facecolor='white')
    ax = fig.gca()
 
    logM_limits = [6, np.log10(1.1*max_M)]  # Limits for log10(M [Msun/h])
    HMF_lgM_press74 = []
    HMF_lgM_sheth99 = []
    # HMF_lgM_tinker08 = []

    dlog10m = (logM_limits[1] - logM_limits[0]) / 60
    logM_list = np.arange(logM_limits[0], logM_limits[1], dlog10m)
    #plot analytical HMF
    for logM in logM_list:
        M = 10**(logM)
        HMF_lgM_press74.append(HMF_Colossus(10**logM, current_redshift, 'press74')* np.log(10)*M)  
        HMF_lgM_sheth99.append(HMF_Colossus(10**logM, current_redshift, 'sheth99')* np.log(10)*M)
        # HMF_lgM_tinker08.append(HMF_Colossus(10**logM, current_redshift, 'tinker08', mdef = '200c')* np.log(10)*M)
    HMF_lgM_press74 = np.array(HMF_lgM_press74)
    HMF_lgM_sheth99 = np.array(HMF_lgM_sheth99)
    # HMF_lgM_tinker08 = np.array(HMF_lgM_tinker08)
    # _, HMF_lgM_tinker08_test = HMF_py_dndlog10m(logM_limits[0], logM_limits[1], dlog10m, current_redshift, 'Tinker08', mdef_model = 'SOCritical', mdef_params = {'overdensity': 200})

    #plot the dark matter resolution and TNG HMF
    plt.yscale('log')
    plt.xscale('log')
    plt.axvline(100*dark_matter_resolution, color='black', linestyle='--')
    plt.scatter(bin_centers, number_density*comoving_factor, c='none', edgecolor='blue', marker='o', label='All HIMALAYA halos')
    plt.scatter(bin_centers, number_density_selected*comoving_factor, c='none', edgecolor='green', marker='^',label='Selected HIMALAYA halos')

    plt.plot(10**(logM_list),HMF_lgM_press74*comoving_factor, color='k',linestyle='-',label='Press74')
    plt.plot(10**(logM_list),HMF_lgM_sheth99*comoving_factor, color='red',linestyle='-',label='Sheth99')
    # plt.plot(10**(logM_list),HMF_lgM_tinker08*comoving_factor, color='blue',linestyle='-',label='Tinker08')
    # plt.plot(10**(logM_list),HMF_lgM_tinker08_test,color='orange',linestyle='-',label='Tinker08')
    plt.legend(fontsize=13)
    
    plt.xlabel(r'Mass [$\mathrm{M}_{\odot}/\mathrm{h}$]', fontsize=14)
    plt.ylabel(r'$\frac{\text{dN}}{\text{ d\lg M}}$ [$(\text{cMpc/h})^{-3}$]',fontsize=14)
    ax.tick_params(direction='in', which='both', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(hmf_filename,dpi=300)

    print("HMF plot saved to:", hmf_filename)
    plt.close()
    
    #also save the data
    # hmf_data_filename = hmf_filename.replace('.png','.txt')
    # with open(hmf_data_filename, 'w') as f:
    #     f.write('bin_edge_left, bin_edge_right, bin_center, number_density_all, number_density_selected (dN/dlgM [(Mpc/h)^(-3)])\n')
    #     for i in range(len(bin_centers)):
    #         f.write(str(bin_edges[i]) + ' ' + str(bin_edges[i+1]) + ' ' + str(bin_centers[i]) + ' ' + str(number_density[i]) + ' ' + str(number_density_selected[i]) + '\n')

def plot_Mratio_dN_dlogMratio_himalaya(data_dict, all_subhalo_indices, host_indices_for_subs, current_redshift, snapNum, dark_matter_resolution, output_dir):
    """
    Plot the subhalo mass function (SHMF) from the HIMALAYA simulation data, similar to plot_Mratio_dN_dlogMratio() in TNG_Model.py.
    Parameters:
    - data_dict: Dictionary containing the simulation data read from the HDF5 file.
    - all_subhalo_indices: Indices of the selected subhalos to be included in the SHMF plot.
    - host_indices_for_subs: Indices of the host halos corresponding to each subhalo.
    - current_redshift: The redshift of the simulation snapshot.
    - snapNum: The snapshot number of the simulation.
    - dark_matter_resolution: The mass resolution of dark matter particles in the simulation (in Msun/h).
    - output_dir: The directory to save the SHMF plot
    """


    sub_masses = data_dict['Subhalo/SubhaloMass'][all_subhalo_indices] #in code units (assume 1e10 Msun now)
    sub_masses_Msun_h = sub_masses * 1e10 * h_Hubble  #in Msun/h
    host_masses = data_dict['Group/GroupMass'][host_indices_for_subs] #in code units (assume 1e10 Msun now)
    host_masses_Msun_h = host_masses * 1e10 * h_Hubble  #in Msun/h

    mass_ratios = sub_masses_Msun_h / host_masses_Msun_h
    host_logM = np.log10(host_masses_Msun_h)

    # Divide the host halos into 5 mass bins, and plot the distribution of m/M for each bin respectively
    logM_min = np.min(host_logM)
    logM_max = np.max(host_logM)
    num_M_bins = 5
    logM_bins = np.linspace(logM_min, logM_max, num=num_M_bins+1)
    
    #initialize lists
    sub_host_Mratio_list = []
    cumulative_sub_host_Mratio_list = []
    num_host_list = []
    critical_ratio_list = []
    subhalo_resolution = 50*dark_matter_resolution 

    # Create histogram bins
    num_ratio_bins = 50
    bins = np.linspace(-4, 0, num_ratio_bins+1) #(-4, 0) 
    bin_edges = bins
    log_bin_widths = bins[1] - bins[0]
    artificial_small = 1e-10
    min_number_density = 1e10 #used to set the lower limit of y-axis for plotting


    # Loop over mass bins
    for i in range(num_M_bins):
        print(f"lg mass range: [{logM_bins[i]:.2f}, {logM_bins[i+1]:.2f}]")
        mask = (host_logM >= logM_bins[i]) & (host_logM < logM_bins[i+1])
        unique_host_indices = np.unique(host_indices_for_subs[mask])
        len_unique_hosts = len(unique_host_indices)
        num_host_list.append(len_unique_hosts)
        #define critical ratio = subhalo_resolution/Mhost(left edge of bin)
        critical_ratio_list.append(subhalo_resolution/10**logM_bins[i])
        print(f"Number of host halos in bin {i}: {len_unique_hosts}")
        print(f"number of subhalos in bin {i}: {len(mass_ratios[mask])}")

        #individual SHMF
        sub_host_Mratio_matrix = np.zeros((len_unique_hosts, num_ratio_bins))
        for j, host_index in enumerate(unique_host_indices):
            host_subhalo_mask = (host_indices_for_subs == host_index) & mask
            sub_host_Mratio_matrix[j, :], _ = np.histogram(np.log10(mass_ratios[host_subhalo_mask]), 
                                                            bins=bins)
        sub_host_Mratio_list.append(sub_host_Mratio_matrix)

        cumulative_matrix = np.cumsum(sub_host_Mratio_matrix[:, ::-1], axis=1)[:, ::-1]
        cumulative_sub_host_Mratio_list.append(cumulative_matrix)

    tot_num_host = len(np.unique(host_indices_for_subs))
    print(f"Total number of host halos: {tot_num_host}")  #plot tot distribution separately


    # 1. Plot mass ratio distributions and fit
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
    shmf_filename = os.path.join(output_dir, f'SHMF_z{current_redshift:.2f}_snap{snapNum:03d}.png')
    plt.savefig(shmf_filename,dpi=300)
    print("SHMF plot saved to:", shmf_filename)
    plt.close()

    #calculate the cumulative shmf for the bestfit parameters
    N_bestfit = 10**bestfit_lg_number_density * log_bin_widths
    cumulative_bestfit = np.cumsum(N_bestfit[::-1])[::-1]

    #2. plot the cumulative subhalo mass function (no Poisson correction)
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
    plt.tight_layout()
    filename = os.path.join(output_dir, f'SHMF_cumulative_z{current_redshift:.2f}_snap{snapNum:03d}.png')
    plt.savefig(filename, dpi=300)
    print("Cumulative SHMF plot saved to:", filename)
    plt.close()

def test_pipeline():
    base_dir = "/ceph/cephfs/bsreedhar/HIMALAYA/HIMALAYA_DMO"
    boxsize = 70.0 #cMpc
    boxsize_h = boxsize * h_Hubble  #cMpc/h
    comoving_simulation_volume = boxsize_h**3  #(cMpc/h)^3

    simulation_label = "zooms000"  
    if simulation_label not in ["parent", "zooms000"]:
        raise ValueError("Invalid simulation label: %s" % simulation_label)
    if simulation_label == "parent":
        simulation_dir = os.path.join(base_dir, "parent", "data")
        dark_matter_resolution =  0.00275353  #assume unit is 1e10 Msun, not 1e10 Msun/h, to be confirmed
        dark_matter_resolution_Msun_h = dark_matter_resolution * 1e10 * h_Hubble   #in Msun/h
    elif simulation_label == "zooms000":
        simulation_dir = os.path.join(base_dir, "zooms/000", "data")
        dark_matter_resolution = 0.00275353  #assume unit is 1e10 Msun, not 1e10 Msun/h, to be confirmed
        dark_matter_resolution_Msun_h = dark_matter_resolution * 1e10 * h_Hubble   #in Msun/h
    results_dir = f"/home/zwu/21cm_project/unified_model/HIMALAYA_results/{simulation_label}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    #read redshift list from a file
    redshift_file = os.path.join("/home/zwu/21cm_project/unified_model/HIMALAYA_results", f"redshift_list_{simulation_label}.txt")
    redshift_list = []
    snapNum_list = []
    with open(redshift_file, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.split()
            snapNum = int(parts[0])
            z = float(parts[1])
            snapNum_list.append(snapNum)
            redshift_list.append(z)
    print("Redshift list read from file:", redshift_list)
    print("SnapNum list read from file:", snapNum_list)

    #now read a specific snapshot
    snapNum = 35
    current_redshift = redshift_list[snapNum_list.index(snapNum)]
    print("Testing snapNum:", snapNum, "redshift:", current_redshift)
    scale_factor = 1.0/(1.+current_redshift)
    physical_simulation_volume = comoving_simulation_volume * scale_factor**3  #(Mpc/h)^3

    fof_sub_path = os.path.join(simulation_dir, "fof_subhalo_tab_%03d.hdf5" % snapNum)
    data_dict, attrs_dict = read_hdf5_data(fof_sub_path, include_attrs=True)

    # print(data_dict.keys())
    # dict_keys(['Group/GroupAscale', 'Group/GroupFirstSub', 'Group/GroupLen', 'Group/GroupLenType', 
    # 'Group/GroupMass', 'Group/GroupMassType', 'Group/GroupNsubs', 'Group/GroupOffsetType', 'Group/GroupPos', 
    # 'Group/GroupVel', 'Group/Group_M_Crit200', 'Group/Group_M_Crit500', 'Group/Group_M_Mean200', 
    # 'Group/Group_M_TopHat200', 'Group/Group_R_Crit200', 'Group/Group_R_Crit500', 'Group/Group_R_Mean200', 'Group/Group_R_TopHat200', 
    # 'Subhalo/SubhaloCM', 'Subhalo/SubhaloGroupNr', 'Subhalo/SubhaloHalfmassRad', 'Subhalo/SubhaloHalfmassRadType',
    #  'Subhalo/SubhaloIDMostbound', 'Subhalo/SubhaloLen', 'Subhalo/SubhaloLenType', 'Subhalo/SubhaloMass', 
    # 'Subhalo/SubhaloMassType', 'Subhalo/SubhaloOffsetType', 'Subhalo/SubhaloParentRank', 'Subhalo/SubhaloPos', 'Subhalo/SubhaloRankInGr', 
    # 'Subhalo/SubhaloSpin', 'Subhalo/SubhaloVel', 'Subhalo/SubhaloVelDisp', 'Subhalo/SubhaloVmax', 'Subhalo/SubhaloVmaxRad'])

    tot_num_halos = data_dict['Group/GroupMass'].shape[0]
    mask_groupmass = (data_dict['Group/GroupMass'] > 100 * dark_matter_resolution)  #only consider halos with >100 particles
    mask_M200 = (data_dict['Group/Group_M_Crit200'] > 0)
    mask_R200 = (data_dict['Group/Group_R_Crit200'] > 0)
    mask_subhalo = (data_dict['Group/GroupNsubs'] > 1)
    num_unresolved = np.sum(~mask_groupmass)
    num_M200_zero = np.sum(~mask_M200)
    num_R200_zero = np.sum(~mask_R200)
    num_nosubhalo = np.sum(~mask_subhalo)
    print(f"Total number of halos: {tot_num_halos}")
    print(f"number of unresolved halos: {num_unresolved}")
    print(f"number of halos with M_crit200 = 0: {num_M200_zero}")
    print(f"number of halos with R_crit200 = 0: {num_R200_zero}")
    print(f"number of halos with no subhalo: {num_nosubhalo}")

    mask = mask_groupmass & mask_M200 & mask_R200 & mask_subhalo
    N_selected = np.sum(mask)
    print("number of selected halos: ", N_selected)
    index_selected = np.where(mask)[0]

    # print(index_selected)

    #now add resolved subhalos
    all_subhalo_indices = []
    host_indices_for_subs = []  # index of host halo for each subhalo
    for i, index_host in enumerate(index_selected):  #loop over selected host halos
        num_subs = data_dict['Group/GroupNsubs'][index_host]
        first_sub_index = data_dict['Group/GroupFirstSub'][index_host]
        #skip the first subhalo, which is the central halo itself
        sub_indices = range(first_sub_index + 1, first_sub_index + num_subs)
        host_mass = data_dict['Group/GroupMass'][index_host]

        for j in sub_indices:
            subhalo_mass = data_dict['Subhalo/SubhaloMass'][j]
            if subhalo_mass > 50 * dark_matter_resolution:  #only consider subhalos with >50 particles
                all_subhalo_indices.append(j)
                host_indices_for_subs.append(index_host)

    all_subhalo_indices = np.array(all_subhalo_indices)
    host_indices_for_subs = np.array(host_indices_for_subs)
    n_selected_subs = len(all_subhalo_indices)
    print("number of selected subhalos: ", n_selected_subs)

    #now plot HMF 
    # hmf_filename = os.path.join(results_dir, f"HMF_z{current_redshift:.2f}_snap{snapNum:03d}.png")
    # plot_hmf_himalaya(data_dict, index_selected, current_redshift, dark_matter_resolution_Msun_h, physical_simulation_volume, hmf_filename)


    #plot SHMF
    plot_Mratio_dN_dlogMratio_himalaya(data_dict, all_subhalo_indices, host_indices_for_subs, current_redshift, snapNum, dark_matter_resolution_Msun_h, results_dir)




    
def main():
    pass
    

if __name__ == "__main__":
    print("yt version:", yt.__version__)
    test_pipeline()

