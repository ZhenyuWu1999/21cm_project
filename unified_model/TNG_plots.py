import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import matplotlib.lines as mlines
from TNGDataHandler import load_processed_data
from physical_constants import Zsun, Myr, kpc, Omega_b, Omega_m, h_Hubble



def maxwell_boltzmann_pdf(x, sigma):
    """
    Maxwell-Boltzmann probability density function in 3D
    
    Parameters:
    -----------
    x : array_like
        Points at which to evaluate the distribution
    sigma : float
        Scale parameter (standard deviation)
        
    Returns:
    --------
    array_like
        Probability density at points x
    """
    return 4 * np.pi * x**2 * np.exp(-x**2 / (2 * sigma**2)) / (sigma**3 * (2 * np.pi)**(3/2))

def fit_maxwell_boltzmann(data, bins, range_fit=None, initial_guess=None):
    """
    Fit Maxwell-Boltzmann distribution to data
    
    Parameters:
    -----------
    data : array_like
        Data to fit
    bins : int
        Number of bins for histogram
    range_fit : tuple, optional
        (min, max) range for fitting
    initial_guess : float, optional
        Initial guess for sigma parameter
        
    Returns:
    --------
    tuple
        (fitted parameters, parameter covariances, x points, fitted y points)
    
    example usage:
    popt, pcov, x_fit, y_fit = fit_maxwell_boltzmann(
        data=mach_numbers,
        bins=50,
        range_fit=(0, 5),
        initial_guess=1.0
    )
    sigma_fit = popt[0]
    """
    if range_fit is None:
        range_fit = (np.min(data), np.max(data))
    
    # Create histogram of data
    hist, bin_edges = np.histogram(data, bins=bins, range=range_fit, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if initial_guess is None:
        initial_guess = np.std(data)
    
    # Fit the Maxwell-Boltzmann distribution
    popt, pcov = curve_fit(maxwell_boltzmann_pdf, bin_centers, hist, p0=[initial_guess])
    
    # Generate points for the fitted curve
    x_fit = np.linspace(range_fit[0], range_fit[1], 200)
    y_fit = maxwell_boltzmann_pdf(x_fit, popt[0])
    
    return popt, pcov, x_fit, y_fit
    
def plot_host_halo_properties(data, snapNum, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    group_Tvir = data.halo_data['GroupTvir'].value
    group_metallicity = data.halo_data['GroupGasMetallicity'].value
    group_metallicity_Zsun = group_metallicity / Zsun
    #set lower limit of metallicity to be 1e-10 Zsun to avoid -inf in log10
    group_metallicity_Zsun[group_metallicity_Zsun < 1e-10] = 1e-10

    #plot host halo Tvir 1D histogram
    fig = plt.figure(figsize=(8, 6), facecolor='w')
    plt.hist(np.log10(group_Tvir), bins=50, histtype='step', linewidth=2)
    plt.xlabel(r'log$_{10}$(T$_{vir}$ [K])', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.savefig(os.path.join(output_dir, f'host_Tvir_snap_{snapNum}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    #plot 2D histogram of host halo Tvir vs metallicity
    fig = plt.figure(figsize=(8, 6), facecolor='w')
    plt.hist2d(np.log10(group_Tvir), np.log10(group_metallicity_Zsun), bins=50)
    plt.colorbar(label='Counts')
    plt.xlabel(r'log$_{10}$(T$_{vir}$ [K])', fontsize=14)
    plt.ylabel(r'log$_{10}$($\max(Z_{gas}/Z_{\odot}, 1e-10)$)', fontsize=14)
    plt.savefig(os.path.join(output_dir, f'host_Tvir_metallicity_snap_{snapNum}.png'), dpi=300, bbox_inches='tight')
    plt.close()


    #plot ratio of different components in host halo
    halo_gasmass = data.halo_data['GroupGasMass'].value
    halo_dmmass = data.halo_data['GroupDMmass'].value
    halo_stellarmass = data.halo_data['GroupStellarMass'].value
    halo_bhmass = data.halo_data['GroupBHMass'].value
    halo_mass = data.halo_data['GroupMass'].value

    outputfilename = os.path.join(output_dir, f'host_f_baryon_snap_{snapNum}.png')
    fig = plt.figure(facecolor='white')
    ax = fig.gca()

    scatter_size = 0.1
    gas = plt.scatter(halo_mass, halo_gasmass/halo_mass, s=scatter_size, c='g')
    dm = plt.scatter(halo_mass, halo_dmmass/halo_mass, s=scatter_size, c='gray')
    stars = plt.scatter(halo_mass, halo_stellarmass/halo_mass, s=scatter_size, c='r')
    bh = plt.scatter(halo_mass, halo_bhmass/halo_mass, s=scatter_size, c='b')
    line, = plt.plot([], [], 'k--', label=r'$\Omega_b/\Omega_m$')
    plt.axhline(y=Omega_b/Omega_m, color='k', linestyle='--')

    plt.xscale('log')
    plt.legend(
        [gas, dm, stars, bh, line],
        ['Gas', 'Dark Matter', 'Stars', 'Black Hole', r'$\Omega_b/\Omega_m$'],
        loc='best',
        markerscale=10  # Use a value greater than 1 to increase marker size in legend
    )
    plt.xlabel(r'Halo Mass [$M_{\odot}/h$]')
    plt.ylabel('Mass Fraction')
    plt.savefig(outputfilename, bbox_inches='tight', dpi=200)
    plt.close()


def plot_2D_histogram(data, snapNum, output_dir, fig_options):
    """
    Create 2D histograms of various TNG simulation properties.
    
    Parameters:
    -----------
    data : ProcessedTNGData
        Processed TNG data container
    snapNum : int
        Snapshot number
    output_dir : str
        Directory to save the output plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data from the ProcessedTNGData container
    host_indices = data.subhalo_data['host_index'].value
    
    # Host halo properties (need to use host_indices to match with subhalos)
    host_mass = data.halo_data['GroupMass'].value[host_indices]
    host_M200 = data.halo_data['Group_M_Crit200'].value[host_indices]
    host_R200 = data.halo_data['Group_R_Crit200'].value[host_indices] 
    host_pos = data.halo_data['GroupPos'].value[host_indices]
    
    # Subhalo properties
    subhalo_mass = data.subhalo_data['SubMass'].value
    halfmass_radius = data.subhalo_data['SubHalfmassRad'].value  #unit: m
    vmaxrad = data.subhalo_data['SubVmaxRad'].value
    mach_number = data.subhalo_data['mach_number'].value
    vmaxrad_tcross = data.subhalo_data['vmaxrad_tcross'].value
    host_tff = data.subhalo_data['host_t_ff'].value
    a_number = data.subhalo_data['A_number'].value
    subhalo_pos = data.subhalo_data['SubPos'].value  #unit: kpc/h


    # Total hosthalo mass vs subhalo mass
    if 'Mtot_msub' in fig_options:
        fig = plt.figure(figsize=(8, 6), facecolor='w')
        plt.hist2d(np.log10(host_mass), np.log10(subhalo_mass), bins=50)
        plt.colorbar(label='Counts')
        plt.xlabel(r'log$_{10}$(M$_{\mathrm{host}}$ [M$_{\odot}$/h])', fontsize=14)
        plt.ylabel(r'log$_{10}$(m$_{\mathrm{sub}}$ [M$_{\odot}$/h])', fontsize=14)
        plt.savefig(os.path.join(output_dir, f'Mtot_msub_snap_{snapNum}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    # M200 vs subhalo mass
    if 'M200_msub' in fig_options:
        fig = plt.figure(figsize=(8, 6), facecolor='w')
        plt.hist2d(np.log10(host_M200), np.log10(subhalo_mass), bins=50)
        plt.colorbar(label='Counts')
        plt.xlabel(r'log$_{10}$(M$_{200}$ [M$_{\odot}$/h])', fontsize=14)
        plt.ylabel(r'log$_{10}$(m$_{\mathrm{sub}}$ [M$_{\odot}$/h])', fontsize=14)
        plt.savefig(os.path.join(output_dir, f'M200_msub_snap_{snapNum}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # R200 vs subhalo halfmass radius
    if 'R200_rsubhalfmass' in fig_options:
        valid_indices = (host_R200 > 0) & (halfmass_radius > 0)
        print(f"Number of valid indices: {np.sum(valid_indices)}")
        print(f"Number of total indices: {len(valid_indices)}")
        host_R200_selected = host_R200[valid_indices]
        halfmass_radius_selected = halfmass_radius[valid_indices]
        fig = plt.figure(figsize=(8, 6), facecolor='w')
        plt.hist2d(np.log10(host_R200_selected*1.0e3), np.log10(halfmass_radius_selected/kpc), bins=50)
        plt.colorbar(label='Counts')
        plt.xlabel(r'log$_{10}$(R$_{200}$ [kpc])', fontsize=14)
        plt.ylabel(r'log$_{10}$(r$_{\mathrm{sub,halfmass}}$ [kpc])', fontsize=14)
        plt.savefig(os.path.join(output_dir, f'R200_rsubhalfmass_snap_{snapNum}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # R200 vs subhalo Vmax radius
    if 'R200_subhaloVmaxRad' in fig_options:
        fig = plt.figure(figsize=(8, 6), facecolor='w')
        plt.hist2d(np.log10(host_R200*1.0e3), np.log10(vmaxrad/kpc), bins=50)
        plt.colorbar(label='Counts')
        plt.xlabel(r'log$_{10}$(R$_{200}$ [kpc])', fontsize=14)
        plt.ylabel(r'log$_{10}$(r$_{\mathrm{sub,VmaxRad}}$ [kpc])', fontsize=14)
        plt.savefig(os.path.join(output_dir, f'R200_subhaloVmaxRad_snap_{snapNum}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Free-fall time vs crossing time
    if 'tff_tcross' in fig_options:
        fig = plt.figure(figsize=(8, 6), facecolor='w')
        plt.hist2d(np.log10(host_tff/Myr), np.log10(vmaxrad_tcross/Myr), bins=50)
        plt.colorbar(label='Counts')
        plt.xlabel(r'log$_{10}$(t$_{\mathrm{ff,host}}$ [Myr])', fontsize=14)
        plt.ylabel(r'log$_{10}$(t$_{\mathrm{cross}}$ [Myr])', fontsize=14)
        plt.savefig(os.path.join(output_dir, f'tff_tcross_snap_{snapNum}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    # M200 vs Mach number
    if 'M200_Mach' in fig_options:
        fig = plt.figure(figsize=(8, 6), facecolor='w')
        plt.hist2d(np.log10(host_M200), mach_number, bins=50)
        plt.colorbar(label='Counts')
        plt.xlabel(r'log$_{10}$(M$_{200}$ [M$_{\odot}$/h])', fontsize=14)
        plt.ylabel(r'$\mathcal{M}$', fontsize=14)
        plt.savefig(os.path.join(output_dir, f'M200_Mach_snap_{snapNum}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # M200 vs A number
    if 'M200_Anumber' in fig_options:
        valid_indices = (host_M200 > 0) & (a_number > 0) & np.isfinite(a_number)
        host_M200_selected = host_M200[valid_indices]
        a_number_selected = a_number[valid_indices]
        print(f"Number of valid indices: {len(host_M200_selected)}")
        print(f"Number of total indices: {len(host_M200)}")
        fig = plt.figure(figsize=(8, 6), facecolor='w')
        ax = fig.gca()
        plt.hist2d(np.log10(host_M200_selected), np.log10(a_number_selected), bins=50)
        plt.colorbar(label='Subhalo Counts')
        plt.xlabel(r'log$_{10}$(M$_{200}$ [M$_{\odot}$/h])', fontsize=14)
        plt.ylabel(r'log$_{10} \mathcal{A}$', fontsize=14)
        ax.tick_params(axis='both', direction='in')
        #add a text at left bottom corner for the redshift
        redshift = data.header['Redshift']
        textstr = f"z = {redshift:.2f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.05, 0.15, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        plt.savefig(os.path.join(output_dir, f'M200_Anumber_snap_{snapNum}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    if 'Mach_fit' in fig_options:
        #plot Mach number in different bins of M200 (1D plot, divide M200 into 5 bins)
        #cut off at Mach = 5 to avoid outliers
        mach_number_max = 5
        mach_selected_fraction = len(mach_number[mach_number < mach_number_max]) / len(mach_number)
        fig = plt.figure(figsize=(8, 6), facecolor='w')
        min_M200 = np.min(host_M200)
        max_M200 = np.max(host_M200)
        log_min_M200 = np.log10(min_M200)
        log_max_M200 = np.log10(max_M200)
        M200_bins = np.logspace(log_min_M200, log_max_M200, 6)
        # predefined_bins = np.arange(7, 14.5, 0.5)
        # lower_bins = predefined_bins[predefined_bins > log_min_M200]
        # upper_bins = predefined_bins[predefined_bins < log_max_M200]
        # lower_bin = lower_bins[0]
        # upper_bin = upper_bins[-1]
        # lgM200_bins = np.arange(lower_bin, upper_bin + 0.5, 0.5)
        # M200_bins = 10**lgM200_bins

        #save best fit sigma to a file
        bestfit_mach_filename = os.path.join(output_dir, f'best_fit_Mach_sigma_new.txt')
        with open(bestfit_mach_filename, 'w') as f:
            f.write(f"threshold Mach number: {mach_number_max}, fraction of Mach number < {mach_number_max}: {mach_selected_fraction}\n")
            f.write("M200_min, M200_max, Best Fit Sigma\n")

        colors = plt.cm.plasma(np.linspace(0, 1, 5))
        # colors = plt.cm.plasma(np.linspace(0, 1, len(predefined_bins)-1))

        for i in range(len(M200_bins)-1):
            mask = (host_M200 > M200_bins[i]) & (host_M200 < M200_bins[i+1]) & (mach_number < mach_number_max)
            plt.hist(mach_number[mask], bins=50, histtype='step', linewidth=2, density=True, color=colors[i],label=f'{M200_bins[i]:.2e} - {M200_bins[i+1]:.2e}')
        
            #fit Maxwell-Boltzmann distribution
            popt, pcov, x_fit, y_fit = fit_maxwell_boltzmann(
                data=mach_number[mask],
                bins=50,
                range_fit=(0, 5),
                initial_guess=1.0
            )
            plt.plot(x_fit, y_fit, linestyle='--', color=colors[i])

            with open(bestfit_mach_filename, 'a') as f:
                f.write(f"{M200_bins[i]:.2e}, {M200_bins[i+1]:.2e}, {popt[0]}\n")

        plt.xlabel('$\mathcal{M}$', fontsize=14)
        plt.ylabel('Probability Density', fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'M200_Mach_bins_snap_{snapNum}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    if 'dpos_subhaloVmaxRad' in fig_options:
        #plot distance between subhalo and host halo center vs subhalo Vmax radius
        valid_indices = (vmaxrad > 0)
        print(f"Number of valid indices: {np.sum(valid_indices)}")
        print(f"Number of total indices: {len(valid_indices)}")
        subhalo_pos_selected = subhalo_pos[valid_indices]
        subhalo_radius_selected = vmaxrad[valid_indices]
        host_pos_selected = host_pos[valid_indices]
        # Calculate distance between subhalo and host halo center
        dpos = np.sqrt(np.sum((subhalo_pos_selected - host_pos_selected)**2, axis=1))
        subhalo_radius_kpch = subhalo_radius_selected / (kpc/h_Hubble)
        dpos_rhalf_ratio = dpos / subhalo_radius_kpch

        fig = plt.figure(figsize=(8, 6), facecolor='w')
        plt.hist2d(np.log10(dpos), np.log10(subhalo_radius_kpch), bins=50)
        plt.colorbar(label='Counts')
        plt.xlabel(r'log$_{10}$(distance [kpc/h])', fontsize=14)
        plt.ylabel(r'log$_{10}$(r$_{\mathrm{sub,VmaxRad}}$ [kpc/h])', fontsize=14)
        plt.savefig(os.path.join(output_dir, f'dpos_subhaloVmaxRad_snap_{snapNum}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        #also plot the 1D histogram of dpos/rhalf
        fig = plt.figure(figsize=(8, 6), facecolor='w')
        #only plot the 99.7th percentile to exclude extremely large values
        dpos_rhalf_ratio = dpos_rhalf_ratio[dpos_rhalf_ratio < np.percentile(dpos_rhalf_ratio, 99.7)]
        plt.hist(dpos_rhalf_ratio, bins=50, histtype='step', linewidth=2)
        plt.xlabel(r'log$_{10}$(distance/r$_{\mathrm{sub,VmaxRad}}$)', fontsize=14)
        plt.ylabel('Counts', fontsize=14)
        plt.savefig(os.path.join(output_dir, f'dpos_subhaloVmaxRad_ratio_snap_{snapNum}.png'), dpi=300, bbox_inches='tight')



def compare_mach_numbers(simulation_set, snapNums):
    """
    Compare Mach number distributions at different redshifts.
    
    Parameters:
    -----------
    simulation_set : str
        Name of the simulation set
    snapNums : list
        List of snapshot numbers to compare
    """
    base_dir = '/home/zwu/21cm_project/unified_model/TNG_results/'
    num_snapshots = len(snapNums)
    
    #write best fit sigma to a file
    output_filename = os.path.join(base_dir, simulation_set, 'analysis', 'best_fit_Mach_sigma_allz.txt')
    with open(output_filename, 'w') as f:
        f.write("Snapshot, Redshift, Best Fit Sigma, F(Mach < 5)\n")

    # Setup the plot
    fig = plt.figure(figsize=(8, 6), facecolor='w')
    colors = plt.cm.rainbow(np.linspace(0, 1, num_snapshots))
    
    for snapNum, color in zip(snapNums, colors):
        # Load data
        processed_file = os.path.join(base_dir, simulation_set, f'snap_{snapNum}', 
                                    f'processed_halos_snap_{snapNum}.h5')
        data = load_processed_data(processed_file)
        
        # Get Mach numbers and filter out Mach > 5
        all_mach_numbers = data.subhalo_data['mach_number'].value
        mach_numbers = all_mach_numbers[all_mach_numbers < 5]
        selected_fraction = len(mach_numbers) / len(all_mach_numbers)
        redshift = data.header['Redshift']
        
        # Plot normalized distribution
        plt.hist(mach_numbers, bins=50, density=True, histtype='step', 
                color=color, label=f"z = {redshift:.2f}", linewidth=2)

        # Fit Maxwell-Boltzmann distribution and plot the best fit curve
        popt, pcov, x_fit, y_fit = fit_maxwell_boltzmann(
            data=mach_numbers,
            bins=50,
            range_fit=(0, 5),
            initial_guess=1.0
        )
        print("best fit sigma: ", popt[0])
        plt.plot(x_fit, y_fit, color=color, linestyle='--')

        #write best fit sigma to a file
        with open(output_filename, 'a') as f:
            f.write(f"{snapNum}, {redshift}, {popt[0]}, {selected_fraction}\n")

        # Print statistics
        print(f"\nSnapshot {snapNum}, z = {redshift:.2f}")
        print(f"Number of subhalos: {len(mach_numbers)}")
        print(f"Mean Mach number: {np.mean(mach_numbers):.2f}")
        print(f"Median Mach number: {np.median(mach_numbers):.2f}")
        print(f"Std Mach number: {np.std(mach_numbers):.2f}")
        print(f"Fraction subsonic (M < 1): {np.sum(mach_numbers < 1) / len(mach_numbers):.2%}")
    mb_fit_line = mlines.Line2D([], [], color='gray', linestyle='--', label='Maxwell-Boltzmann fit')
    plt.xlabel('Mach Number')
    plt.ylabel('Probability Density')
    plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + [mb_fit_line])
    plt.grid(True, alpha=0.3)
    plot_dir = os.path.join(base_dir, simulation_set, 'analysis')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'mach_number_distribution_cutMach5_allz.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    simulation_set = 'TNG50-1'
    
    snapNum_list = [0, 1, 2, 3, 4, 6, 8, 11, 13,
                    17,21,25,33,40,50,59,67,72,78,84,91,99]

    
    for snapNum in snapNum_list:
        print(f"Processing snapshot {snapNum} ...")
        base_dir = '/home/zwu/21cm_project/unified_model/TNG_results/'
        processed_file = os.path.join(base_dir, simulation_set, f'snap_{snapNum}', 
                                    f'processed_halos_snap_{snapNum}.h5')
        data = load_processed_data(processed_file)
        # Create plots
        output_dir = os.path.join(base_dir, simulation_set, f'snap_{snapNum}', 'analysis')
        # fig_options_2Dhistogram = ['Mtot_msub', 'M200_msub', 'R200_rsubhalfmass', 
        # 'R200_subhaloVmaxRad', 'tff_tcross', 'M200_Mach', 'M200_Anumber', 'Mach_fit']
        fig_options_2Dhistogram = ['M200_Anumber']
        plot_2D_histogram(data, snapNum, output_dir, fig_options_2Dhistogram)
        # plot_host_halo_properties(data, snapNum, output_dir)
    
    # snapNum_list = [1, 2, 3, 4, 6, 8, 11, 13, 17, 21, 25, 33, 50, 99]
    # # Compare Mach numbers across snapshots
    # compare_mach_numbers(simulation_set, snapNum_list)

