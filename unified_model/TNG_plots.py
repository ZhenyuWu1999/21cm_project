import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from scipy.optimize import curve_fit
from scipy.stats import truncnorm
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



def truncated_gaussian_pdf(x, mu, sigma):
    """
    PDF of a Gaussian truncated at x>=0
    """
    a, b = (0 - mu) / sigma, np.inf
    return truncnorm.pdf(x, a, b, loc=mu, scale=sigma)

def fit_truncated_gaussian(data, bins=50, range_fit=None, initial_guess=(1.0, 1.0)):
    """
    Fit a truncated Gaussian N(mu, sigma) on [0,inf) to data.
    Returns (popt, pcov, x_fit, y_fit)
    """
    if range_fit is None:
        range_fit = (np.min(data), np.max(data))

    # histogram
    hist, bin_edges = np.histogram(data, bins=bins, range=range_fit, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    popt, pcov = curve_fit(
        truncated_gaussian_pdf, bin_centers, hist, p0=initial_guess,
        bounds=([0, 1e-6], [np.inf, np.inf])  # mu>=0, sigma>0
    )

    x_fit = np.linspace(range_fit[0], range_fit[1], 200)
    y_fit = truncated_gaussian_pdf(x_fit, *popt)
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
        filename = os.path.join(output_dir, f'Mtot_msub_snap_{snapNum}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
        plt.close()
        
    # M200 vs subhalo mass
    if 'M200_msub' in fig_options:
        fig = plt.figure(figsize=(8, 6), facecolor='w')
        plt.hist2d(np.log10(host_M200), np.log10(subhalo_mass), bins=50)
        plt.colorbar(label='Counts')
        plt.xlabel(r'log$_{10}$(M$_{200}$ [M$_{\odot}$/h])', fontsize=14)
        plt.ylabel(r'log$_{10}$(m$_{\mathrm{sub}}$ [M$_{\odot}$/h])', fontsize=14)
        filename = os.path.join(output_dir, f'M200_msub_snap_{snapNum}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
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
        filename = os.path.join(output_dir, f'R200_rsubhalfmass_snap_{snapNum}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
        plt.close()
    
    # R200 vs subhalo Vmax radius
    if 'R200_subhaloVmaxRad' in fig_options:
        fig = plt.figure(figsize=(8, 6), facecolor='w')
        plt.hist2d(np.log10(host_R200*1.0e3), np.log10(vmaxrad/kpc), bins=50)
        plt.colorbar(label='Counts')
        plt.xlabel(r'log$_{10}$(R$_{200}$ [kpc])', fontsize=14)
        plt.ylabel(r'log$_{10}$(r$_{\mathrm{sub,VmaxRad}}$ [kpc])', fontsize=14)
        filename = os.path.join(output_dir, f'R200_subhaloVmaxRad_snap_{snapNum}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
        plt.close()
    
    # Free-fall time vs crossing time
    if 'tff_tcross' in fig_options:
        fig = plt.figure(figsize=(8, 6), facecolor='w')
        plt.hist2d(np.log10(host_tff/Myr), np.log10(vmaxrad_tcross/Myr), bins=50)
        plt.colorbar(label='Counts')
        plt.xlabel(r'log$_{10}$(t$_{\mathrm{ff,host}}$ [Myr])', fontsize=14)
        plt.ylabel(r'log$_{10}$(t$_{\mathrm{cross}}$ [Myr])', fontsize=14)
        filename = os.path.join(output_dir, f'tff_tcross_snap_{snapNum}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
        plt.close()
        
    # M200 vs Mach number
    if 'M200_Mach' in fig_options:
        fig = plt.figure(figsize=(8, 6), facecolor='w')
        plt.hist2d(np.log10(host_M200), mach_number, bins=50)
        plt.colorbar(label='Counts')
        plt.xlabel(r'log$_{10}$(M$_{200}$ [M$_{\odot}$/h])', fontsize=14)
        plt.ylabel(r'$\mathcal{M}$', fontsize=14)
        filename = os.path.join(output_dir, f'M200_Mach_snap_{snapNum}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
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
        filename = os.path.join(output_dir, f'M200_Anumber_snap_{snapNum}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
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
        filename = os.path.join(output_dir, f'M200_Mach_bins_snap_{snapNum}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
        plt.close()
        
    if 'Mach_fixedhostmass' in fig_options:

        fit_mode = "both"  # "maxwell-boltzmann" | "truncated-gaussian" | "both"

        mach_number_max = 5.0
        mach_selected_fraction = np.mean(mach_number < mach_number_max)

        logM200_min, logM200_max, step = 7.5, 14.0, 0.5
        logM200_bins = np.arange(logM200_min, logM200_max + step, step)
        M200_bins = 10 ** logM200_bins
        min_count = 50

        # Define output files
        out_trunc = os.path.join(output_dir, 'best_fit_Mach_truncatedgaussian_fixedhostmass.txt')
        out_maxwell = os.path.join(output_dir, 'best_fit_Mach_sigma_fixedhostmass.txt')

        # Write headers
        for fname, mode in [(out_trunc, "truncated-gaussian"), (out_maxwell, "maxwell-boltzmann")]:
            with open(fname, 'w') as f:
                f.write(f"threshold Mach number: {mach_number_max}, "
                        f"fraction of Mach number < {mach_number_max}: {mach_selected_fraction:.3f}\n")
                if mode == "truncated-gaussian":
                    f.write("logM200_min, logM200_max, N_sub, Best Fit Mu, Best Fit Sigma\n")
                else:
                    f.write("logM200_min, logM200_max, N_sub, Best Fit Sigma\n")

        # === Set up figure ===
        if fit_mode == "both":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, facecolor='w')
            axes = {"maxwell-boltzmann": ax1, "truncated-gaussian": ax2}
        else:
            fig, ax = plt.subplots(figsize=(8, 6), facecolor='w')
            axes = {fit_mode: ax}

        colors_all = plt.cm.plasma(np.linspace(0, 1, len(M200_bins) - 1))

        # === Loop over bins ===
        tot_bins = 0
        for i in range(len(M200_bins) - 1):
            mask = (host_M200 > M200_bins[i]) & (host_M200 < M200_bins[i + 1]) & (mach_number < mach_number_max)
            n_in_bin = int(np.sum(mask))
            if n_in_bin >= min_count:
                tot_bins += 1

        plot_bin_step = 2
        # if tot_bins >= 10:
        #     plot_bin_step = 2 #show less lines in the plot for clarity


        for i in range(len(M200_bins) - 1):
            draw_line = (i % plot_bin_step == 0)
            mask = ((host_M200 > M200_bins[i]) & (host_M200 < M200_bins[i + 1]) &
                    (mach_number < mach_number_max))
            n_in_bin = int(np.sum(mask))
            if n_in_bin < min_count:
                continue

            log_lo, log_hi = logM200_bins[i], logM200_bins[i + 1]
            label = f'{log_lo:.1f}–{log_hi:.1f}'
            color = colors_all[i]

            for mode in axes.keys():
                ax = axes[mode]
                if draw_line:
                    line = ax.hist(mach_number[mask], bins=50, histtype='step', linewidth=2,
                            density=True, color=color, label=label)[2][0]
                if mode == "truncated-gaussian":
                    try:
                        popt, pcov, x_fit, y_fit = fit_truncated_gaussian(
                            data=mach_number[mask],
                            bins=50,
                            range_fit=(0, mach_number_max),
                            initial_guess=(1.0, 1.0)
                        )
                        if draw_line:
                            ax.plot(x_fit, y_fit, '--', color=color)
                        mu_fit, sigma_fit = popt
                    except Exception:
                        mu_fit, sigma_fit = np.nan, np.nan
                    with open(out_trunc, 'a') as f:
                        f.write(f"{log_lo:.1f}, {log_hi:.1f}, {n_in_bin}, {mu_fit:.4f}, {sigma_fit:.4f}\n")
                elif mode == "maxwell-boltzmann":
                    try:
                        popt, pcov, x_fit, y_fit = fit_maxwell_boltzmann(
                            data=mach_number[mask],
                            bins=50,
                            range_fit=(0, mach_number_max),
                            initial_guess=1.0
                        )
                        if draw_line:
                            ax.plot(x_fit, y_fit, '--', color=color)
                        sigma_fit = popt[0]
                    except Exception:
                        sigma_fit = np.nan
                    with open(out_maxwell, 'a') as f:
                        f.write(f"{log_lo:.1f}, {log_hi:.1f}, {n_in_bin}, {sigma_fit:.4f}\n")

        # === Final formatting ===
        for mode, ax in axes.items():
            ax.set_xlabel(r'$\mathcal{M}$', fontsize=14)
            ax.set_ylabel('Probability Density', fontsize=14)
            title = "Maxwell–Boltzmann Fit" if mode == "maxwell-boltzmann" else "Truncated Gaussian Fit"
            ax.set_title(title, fontsize=14)

        # Common legend on the right panel
        handles, labels = ax2.get_legend_handles_labels() if fit_mode == "both" else ax.get_legend_handles_labels()
        if handles:
            axes[list(axes.keys())[-1]].legend(
                handles, labels,
                title=r'$\log_{10}(M_{200}\,[M_\odot/h])$',
                frameon=True, facecolor='white', edgecolor='black', framealpha=0.95,
                loc='best'
            )

        # === Save ===
        if fit_mode == "both":
            filename = os.path.join(output_dir, f'M200_Mach_both_fixedhostmass_snap_{snapNum}.png')
        elif fit_mode == "truncated-gaussian":
            filename = os.path.join(output_dir, f'M200_Mach_truncatedgaussian_fixedhostmass_snap_{snapNum}.png')
        elif fit_mode == "maxwell-boltzmann":
            filename = os.path.join(output_dir, f'M200_Mach_maxwellboltzmann_fixedhostmass_snap_{snapNum}.png')

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
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
        filename = os.path.join(output_dir, f'dpos_subhaloVmaxRad_ratio_snap_{snapNum}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
        plt.close()


def _host_equal_weights(host_indices):
    # Each subhalo gets a weight = 1 / (number of subhalos of its host halo)
    uh, cnt = np.unique(host_indices, return_counts=True)
    mp = dict(zip(uh, cnt))
    return np.array([1.0 / mp[i] for i in host_indices], dtype=float)

def _colnorm(H):
    # Normalize each column of the 2D histogram so that the sum in each X-bin is 1
    s = H.sum(axis=0, keepdims=True)
    s[s==0] = 1.0
    return H / s


def plot_conditional_logA(
    data, snapNum, output_dir,
    xmode="both",                # "msub" | "psi" | "both"
    nbins_x=50, nbins_y=50,
    x_range=None,
    x_range_msub=None, x_range_psi=None,
    y_range=None,
    weight_by_host=True,
    draw_quantiles=True,
    draw_A1=True
):
    """
    Plot conditional PDF of log10(A) given X.

    Parameters
    ----------
    data : object
        Processed data with subhalo_data, halo_data, and header.
    snapNum : int
        Snapshot number.
    output_dir : str
        Directory to save the figure.
    xmode : str
        'msub' : X = log10(m_sub)
        'psi'   : X = log10(psi = m_sub / M_host)
        'both' : Plot both columns side by side.
    nbins_x, nbins_y : int
        Number of bins in X and Y direction.
    x_range : tuple or None
        X-axis range when plotting a single mode.
    x_range_msub, x_range_mu : tuple or None
        X-axis range for "both" mode (can be set individually).
    y_range : tuple or None
        Y-axis range (log10 A).
    weight_by_host : bool
        If True, each host halo contributes equal total weight.
    draw_quantiles : bool
        If True, overlay 16/50/84% weighted quantiles.
    draw_A1 : bool
        If True, draw horizontal line at A = 1 (logA = 0).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- Extract data ----
    host_idx = data.subhalo_data['host_index'].value
    A = data.subhalo_data['A_number'].value
    Msub = data.subhalo_data['SubMass'].value
    M_host = data.halo_data['GroupMass'].value[host_idx]
    z = data.header['Redshift']

    # Apply validity mask
    valid = np.isfinite(A) & (A > 0) & np.isfinite(Msub) & (Msub > 0) \
            & np.isfinite(M_host) & (M_host > 0) & (Msub/M_host < 1)
    A = A[valid]
    Msub = Msub[valid]
    M_host = M_host[valid]
    host_idx_v = host_idx[valid]
    logA = np.log10(A)

    # ---- Assign weights ----
    weights = _host_equal_weights(host_idx_v) if weight_by_host else np.ones_like(logA)

    if y_range is None:
        y_range = tuple(np.percentile(logA, [0.1, 99.9]))

    # ---- Helper: single-panel plot ----
    def _plot_single(ax, X, xlabel, x_range, colorbar_label):
        if x_range is None:
            x_range = tuple(np.percentile(X, [0.5, 99.5]))

        # Bin edges
        x_edges = np.linspace(x_range[0], x_range[1], nbins_x + 1)
        y_edges = np.linspace(y_range[0], y_range[1], nbins_y + 1)

        # 2D histogram (note: histogram2d expects [y, x] order)
        H, _, _ = np.histogram2d(logA, X, bins=[y_edges, x_edges], weights=weights)
        Hc = _colnorm(H)

        # Heatmap
        Xmesh, Ymesh = np.meshgrid(x_edges, y_edges)
        im = ax.pcolormesh(Xmesh, Ymesh, Hc, shading='auto')
        plt.colorbar(im, ax=ax, label=colorbar_label)

        # Horizontal line at A=1
        if draw_A1 and (y_range[0] < 0 < y_range[1]):
            ax.axhline(0, lw=2, ls=':', color='black', alpha=1.0)

        # Weighted quantiles (16/50/84%)
        if draw_quantiles:
            x_bin_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            q16_vals = np.full(nbins_x, np.nan)
            q50_vals = np.full(nbins_x, np.nan)
            q84_vals = np.full(nbins_x, np.nan)

            for j in range(nbins_x):
                in_bin = (X >= x_edges[j]) & (X < x_edges[j+1])
                if not np.any(in_bin):
                    continue

                logA_bin = logA[in_bin]
                weights_bin = weights[in_bin]

                # Sort values by logA
                sort_idx = np.argsort(logA_bin)
                logA_sorted = logA_bin[sort_idx]
                weights_sorted = weights_bin[sort_idx]

                # Weighted CDF
                weighted_cdf = np.cumsum(weights_sorted) / np.sum(weights_sorted)

                # Interpolate quantiles
                q16_vals[j] = np.interp(0.16, weighted_cdf, logA_sorted)
                q50_vals[j] = np.interp(0.50, weighted_cdf, logA_sorted)
                q84_vals[j] = np.interp(0.84, weighted_cdf, logA_sorted)

            ax.plot(x_bin_centers, q50_vals, lw=2, color='grey', label='median')
            ax.plot(x_bin_centers, q16_vals, lw=2, ls='--', color='red', label='16/84%')
            ax.plot(x_bin_centers, q84_vals, lw=2, ls='--', color='red')
            ax.legend(frameon=True, facecolor='white', framealpha=1)

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.tick_params(axis='both', direction='in')

    # ---- Plot according to xmode ----
    if xmode == "both":
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False, facecolor='w')

        _plot_single(
            axes[0], np.log10(Msub),
            xlabel=r'$\log_{10}(m_{\rm sub}) [M_{\odot}/h]$',
            x_range=x_range_msub,
            colorbar_label=r'$p(\log_{10}\mathcal{A}\mid m_{\rm sub})$'
        )
        _plot_single(
            axes[1], np.log10(Msub / M_host),
            xlabel=r'$\log_{10}(\psi = m_{\rm sub}/M_{\rm host})$',
            x_range=x_range_psi,
            colorbar_label=r'$p(\log_{10}\mathcal{A}\mid \psi)$'
        )

        axes[0].set_ylabel(r'$\log_{10}\mathcal{A}$',fontsize=14)
        axes[0].text(0.02, 0.02, f'z = {z:.2f}', transform=axes[0].transAxes,
                     va='bottom', ha='left', fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

        suffix = "_wWeight" if weight_by_host else "_noWeight"
        fname = f'cond_logA_both_snap_{snapNum}{suffix}.png'

    else:
        fig, ax = plt.subplots(figsize=(7, 6), facecolor='w')
        if xmode == "msub":
            X = np.log10(Msub)
            xlabel = r'$\log_{10}(m_{\rm sub}) [M_{\odot}/h]$'
            colorbar_label = r'$p(\log_{10}\mathcal{A}\mid m_{\rm sub})$'
        elif xmode == "psi":
            X = np.log10(Msub / M_host)
            xlabel = r'$\log_{10}(\psi = m_{\rm sub}/M_{\rm host})$'
            colorbar_label = r'$p(\log_{10}\mathcal{A}\mid \psi)$'
        else:
            raise ValueError("xmode must be 'msub', 'psi' or 'both'")

        _plot_single(ax, X, xlabel, x_range, colorbar_label=colorbar_label)
        ax.set_ylabel(r'$\log_{10}\mathcal{A}$', fontsize=14)
        ax.text(0.02, 0.02, f'z = {z:.2f}', transform=ax.transAxes,
                va='bottom', ha='left', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

        suffix = "_wWeight" if weight_by_host else "_noWeight"
        fname = f'cond_logA_{xmode}_snap_{snapNum}{suffix}.png'

    out = os.path.join(output_dir, fname)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"[Saved] {out}")
    plt.close() 


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



def plot_sigma_vs_hostmass_over_snaps(
    snapNum_list,
    base_dir,
    simulation_set,
    *,
    fit_mode="maxwell-boltzmann",  # or "truncated-gaussian"
    min_count_for_plot=0,        
    cmap_name='plasma'            
):
    
    """
    Unified version: plot best-fit parameters (MB or Truncated Gaussian)
    vs host halo mass over multiple snapshots.

    Parameters
    ----------
    snapNum_list : list[int]
    base_dir : str
    simulation_set : str
    fit_mode : str
        "maxwell-boltzmann" → plot sigma only (single panel)
        "truncated-gaussian" → plot mu and sigma (2-panel)
    min_count_for_plot : int
        Optional threshold: skip bins with too few subhalos
    cmap_name : str
        Colormap name for redshift encoding
    """

    # Helper: get redshift from processed file header
    def _read_redshift(sim_set, snap):
        try:
            processed_file = os.path.join(
                base_dir, sim_set, f'snap_{snap}', f'processed_halos_snap_{snap}.h5'
            )
            # lightweight read: if you already have load_processed_data, you can reuse it directly
            data = load_processed_data(processed_file)
            print(f"Snapshot {snap}, Redshift: {data.header['Redshift']} loaded.")

            return float(data.header['Redshift'])
        except Exception:
            return np.nan

    #  File & label config 
    if fit_mode == "maxwell-boltzmann":
        filename_txt = 'best_fit_Mach_maxwellboltzmann_fixedhostmass.txt'
        output_png = 'bestfit_sigma_MB_vs_hostmass_over_snaps.png'
        ylabels = [r'Best-fit $\sigma$ (Maxwell)']
    elif fit_mode == "truncated-gaussian":
        filename_txt = 'best_fit_Mach_truncatedgaussian_fixedhostmass.txt'
        output_png = 'bestfit_mu_sigma_TG_vs_hostmass_over_snaps.png'
        ylabels = [r'Best-fit $\mu$ (Truncated Gaussian)',
                   r'Best-fit $\sigma$ (Truncated Gaussian)']
    else:
        raise ValueError(f"Unknown fit_mode: {fit_mode}")


    # collect data series
    series_list = []  # Each element: {'snap', 'z', 'logM_center', 'mu', 'sigma'}

    for snap in snapNum_list:
        txt_path = os.path.join(base_dir, simulation_set, f'snap_{snap}', 'analysis', filename_txt)
        if not os.path.exists(txt_path):
            print(f"[warn] File not found, skipping: {txt_path}")
            continue

        z = _read_redshift(simulation_set, snap)

        logM_centers = []
        mu_vals = []
        sigma_vals = []

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines[2:]:  # skip headers
            parts = [p.strip() for p in line.split(',')]
            try:
                log_lo = float(parts[0])
                log_hi = float(parts[1])
                n_sub = int(float(parts[2]))
                if fit_mode == "maxwell-boltzmann":
                    sigma = float(parts[3])
                    mu = None
                elif fit_mode == "truncated-gaussian":
                    mu = float(parts[3])
                    sigma = float(parts[4])
            except Exception:
                continue

            if not np.isfinite(sigma) or (fit_mode == "truncated-gaussian" and not np.isfinite(mu)):
                continue
            if min_count_for_plot > 0 and n_sub < min_count_for_plot:
                continue

            logM_centers.append(0.5 * (log_lo + log_hi))
            mu_vals.append(mu)
            sigma_vals.append(sigma)

        if len(logM_centers) == 0:
            print(f"[info] No valid bins for snap {snap} (z={z:.2f})")
            continue

        order = np.argsort(logM_centers)
        series_list.append({
            'snap': snap,
            'z': z,
            'logM_center': np.array(logM_centers)[order],
            'mu': np.array(mu_vals)[order] if fit_mode == "truncated-gaussian" else None,
            'sigma': np.array(sigma_vals)[order],
        })

    if not series_list:
        print("[error] No valid data found.")
        return

    # prepare color map
    series_list.sort(key=lambda d: d['z'], reverse=True)
    zs = np.array([d['z'] for d in series_list])
    zmin, zmax = np.min(zs), np.max(zs)

    cmap = plt.get_cmap(cmap_name)

    if fit_mode == "truncated-gaussian":
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor='w', sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(8.6, 6.4), facecolor='w')
        axes = [ax]

    handles, labels = [], []

    for d in series_list:
        z = d['z']
        t = (z - zmin) / (zmax - zmin) 
        color = cmap(t)
        label = f"z={z:.2f} (snap {d['snap']})"

        # σ panel (always present)
        h = axes[-1].plot(
            d['logM_center'], d['sigma'],
            marker='o', ms=3.5, lw=1.7, color=color
        )[0]
        handles.append(h)
        labels.append(label)

        # μ panel (TG only)
        if fit_mode == "truncated-gaussian":
            axes[0].plot(
                d['logM_center'], d['mu'],
                marker='o', ms=3.5, lw=1.7, color=color
            )

    # ------------------ Axis labels ------------------
    for i, ax in enumerate(axes):
        ax.set_xlabel(r'$\log_{10}(M_{200}\,[M_\odot/h])$', fontsize=13)
        ax.set_ylabel(ylabels[i], fontsize=13)
        ax.tick_params(axis='both', direction='in')

    # ------------------ Legend ------------------
    leg = axes[-1].legend(
        handles, labels, title='Snapshots',
        frameon=True, facecolor='white', edgecolor='black', framealpha=0.95,
        loc='best'
    )
    try:
        leg._legend_box.align = "left"
    except Exception:
        pass

    # ------------------ Optional colorbar ------------------
    norm = colors.Normalize(vmin=zmin, vmax=zmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, pad=0.01, shrink=0.95)
    cbar.set_label('Redshift z')

    # ------------------ Save figure ------------------
    outdir = os.path.join(base_dir, simulation_set, 'analysis')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, output_png)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {outpath}")
    plt.close()



if __name__ == '__main__':
    simulation_set = 'TNG50-1'

    # snapNum_list = [0, 1, 2, 3, 4, 6, 8, 11, 13, 17, 21, 25, 33, 40, 50, 59, 67, 72, 78, 84, 91, 99]
    # snapNum_list = [2, 13, 99]
    
    # for snapNum in snapNum_list:
    #     print(f"Processing snapshot {snapNum} ...")
    #     base_dir = '/home/zwu/21cm_project/unified_model/TNG_results/'
    #     processed_file = os.path.join(base_dir, simulation_set, f'snap_{snapNum}', 
    #                                 f'processed_halos_snap_{snapNum}.h5')
    #     data = load_processed_data(processed_file)
    #     # Create plots
    #     output_dir = os.path.join(base_dir, simulation_set, f'snap_{snapNum}', 'analysis')
    #     # fig_options_2Dhistogram = ['Mtot_msub', 'M200_msub', 'R200_rsubhalfmass', 
    #     # 'R200_subhaloVmaxRad', 'tff_tcross', 'M200_Mach', 'M200_Anumber', 'Mach_fit']
    #     fig_options_2Dhistogram = ['Mach_fixedhostmass']
    #     # plot_2D_histogram(data, snapNum, output_dir, fig_options_2Dhistogram)
    #     # plot_host_halo_properties(data, snapNum, output_dir)
    #     # plot_conditional_logA(data, snapNum, output_dir, xmode="both", weight_by_host=False)

    snapNum_list = [1, 2, 3, 4, 6, 8, 11, 13, 17, 21, 25, 33, 50, 99]
    # # Compare Mach numbers across snapshots
    # compare_mach_numbers(simulation_set, snapNum_list)

    plot_sigma_vs_hostmass_over_snaps(
    snapNum_list=snapNum_list,
    base_dir='/home/zwu/21cm_project/unified_model/TNG_results/',
    simulation_set=simulation_set,
    fit_mode="truncated-gaussian", # "maxwell-boltzmann" | "truncated-gaussian"
    min_count_for_plot=0,         
    )
