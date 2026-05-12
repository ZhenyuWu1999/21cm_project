import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from TNGDataHandler import get_simulation_resolution
from TNGDataHandler import load_processed_data
from physical_constants import h_Hubble


def minimum_image_displacement(pos, center, box_size):
    """Return displacement vectors in ckpc/h with periodic minimum-image wrapping."""
    delta = pos - center
    if box_size is not None and box_size > 0:
        delta -= box_size * np.round(delta / box_size)
    return delta


def get_subhalo_host_distance(data):
    """
    Return host-centric subhalo distance vectors and normalized radii.

    SubhaloPos and GroupPos are comoving ckpc/h. Group_R_Crit200 is physical Mpc.
    """
    host_indices = data.subhalo_data['host_index'].value
    subhalo_pos = data.subhalo_data['SubPos'].value
    host_pos = data.halo_data['GroupPos'].value[host_indices]
    host_R200 = data.halo_data['Group_R_Crit200'].value[host_indices]

    box_size = data.header.get('BoxSize', None)
    scale_factor = data.header.get('Time', 1.0)
    dpos_vec_ckpch = minimum_image_displacement(subhalo_pos, host_pos, box_size)
    dpos_ckpch = np.sqrt(np.sum(dpos_vec_ckpch**2, axis=1))
    dpos_phys_mpc = dpos_ckpch / 1.0e3 * scale_factor / h_Hubble
    dpos_over_R200 = dpos_phys_mpc / host_R200
    return dpos_vec_ckpch, dpos_phys_mpc, dpos_over_R200


def get_subhalo_host_distance_over_R200(data):
    """Return d_sub-host/R200 for processed TNG subhalos."""
    return get_subhalo_host_distance(data)[2]


def _format_profile_value_tag(value):
    """Format a float for compact filename tags, e.g. 0.05 -> 0p05."""
    text = f'{value:g}'
    return text.replace('-', 'm').replace('.', 'p')


def get_profile_tag(psi_range=None, profile_tag=None):
    """Return a filename tag, preferring a manually supplied tag."""
    if profile_tag is not None:
        return profile_tag
    if psi_range is None:
        return ''
    left = _format_profile_value_tag(psi_range[0])
    right = _format_profile_value_tag(psi_range[1])
    return f'psi{left}_{right}'


PSI_THRESHOLDS_FOR_EXPORT = (1.0e-3, 1.0e-2, 5.0e-2)
POP2PRIME_TNG_COMPARISON_DIR = Path(
    '/home/zwu/21cm_project/unified_model/Pop2prime_results/Pop2Prime_TNG_comparison'
)


def _build_tng_radial_profile_context(
    data,
    host_mass_key='GroupMass',
    psi_mass_key='GroupMass',
    num_M_bins=5,
    x_min=1.0e-2,
    x_max=3.0,
    num_x_bins=30,
    log_x_bins=True,
):
    """Precompute reusable arrays for TNG radial-profile exports."""
    host_indices_for_subs = data.subhalo_data['host_index'].value.astype(int)
    host_masses_all = data.halo_data[host_mass_key].value
    psi_host_masses_all = data.halo_data[psi_mass_key].value
    sub_masses = data.subhalo_data['SubMass'].value
    psi_host_masses_for_subs = psi_host_masses_all[host_indices_for_subs]
    dpos_over_R200 = get_subhalo_host_distance_over_R200(data)

    valid_subs_base = (
        np.isfinite(dpos_over_R200)
        & (dpos_over_R200 > 0)
        & np.isfinite(psi_host_masses_for_subs)
        & (psi_host_masses_for_subs > 0)
    )
    psi_all = sub_masses / psi_host_masses_for_subs

    valid_hosts = np.isfinite(host_masses_all) & (host_masses_all > 0)
    host_logM_all = np.log10(host_masses_all)
    logM_min = np.min(host_logM_all[valid_hosts])
    logM_max = np.max(host_logM_all[valid_hosts])
    host_mass_bins = np.linspace(logM_min, logM_max, num_M_bins + 1)

    if log_x_bins:
        x_edges = np.logspace(np.log10(x_min), np.log10(x_max), num_x_bins + 1)
    else:
        x_edges = np.linspace(x_min, x_max, num_x_bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    dx3 = x_edges[1:]**3 - x_edges[:-1]**3

    host_bin_masks = []
    host_ids_by_bin = []
    host_local_index_by_bin = []
    for i in range(num_M_bins):
        is_last = i == num_M_bins - 1
        if is_last:
            host_mask = (
                valid_hosts
                & (host_logM_all >= host_mass_bins[i])
                & (host_logM_all <= host_mass_bins[i + 1])
            )
        else:
            host_mask = (
                valid_hosts
                & (host_logM_all >= host_mass_bins[i])
                & (host_logM_all < host_mass_bins[i + 1])
            )
        host_ids = np.where(host_mask)[0]
        local_index = np.full(host_masses_all.shape[0], -1, dtype=int)
        local_index[host_ids] = np.arange(host_ids.size)
        host_bin_masks.append(host_mask)
        host_ids_by_bin.append(host_ids)
        host_local_index_by_bin.append(local_index)

    return {
        'host_indices_for_subs': host_indices_for_subs,
        'host_masses_all': host_masses_all,
        'valid_hosts': valid_hosts,
        'host_logM_all': host_logM_all,
        'host_mass_bins': host_mass_bins,
        'valid_subs_base': valid_subs_base,
        'psi_all': psi_all,
        'dpos_over_R200': dpos_over_R200,
        'x_edges': x_edges,
        'x_centers': x_centers,
        'dx3': dx3,
        'num_x_bins': num_x_bins,
        'num_M_bins': num_M_bins,
        'host_mass_key': host_mass_key,
        'psi_mass_key': psi_mass_key,
        'host_bin_masks': host_bin_masks,
        'host_ids_by_bin': host_ids_by_bin,
        'host_local_index_by_bin': host_local_index_by_bin,
    }


def _compute_host_profile_statistics(
    host_ids,
    host_local_index,
    sub_host_indices,
    sub_x_values,
    x_edges,
    dx3,
):
    """Compute per-host radial-profile summary statistics without per-host histograms."""
    n_hosts_total = len(host_ids)
    num_x_bins = len(dx3)
    if n_hosts_total == 0:
        zeros = np.zeros(num_x_bins)
        return {
            'n_hosts_with_subhalos': 0,
            'n_subhalos': 0,
            'mean_profile': zeros,
            'median_profile': zeros,
            'p16_profile': zeros,
            'p84_profile': zeros,
        }

    profile_matrix = np.zeros((n_hosts_total, num_x_bins), dtype=float)
    if sub_x_values.size > 0:
        x_bin_indices = np.searchsorted(x_edges, sub_x_values, side='right') - 1
        valid_x = (x_bin_indices >= 0) & (x_bin_indices < num_x_bins)
        if np.any(valid_x):
            local_host_indices = host_local_index[sub_host_indices[valid_x]]
            valid_host = local_host_indices >= 0
            if np.any(valid_host):
                np.add.at(
                    profile_matrix,
                    (local_host_indices[valid_host], x_bin_indices[valid_x][valid_host]),
                    1.0,
                )

    hosts_with_subhalos = int(np.count_nonzero(np.any(profile_matrix > 0, axis=1)))
    profile_matrix /= dx3[None, :]
    return {
        'n_hosts_with_subhalos': hosts_with_subhalos,
        'n_subhalos': int(sub_x_values.size),
        'mean_profile': np.mean(profile_matrix, axis=0),
        'median_profile': np.median(profile_matrix, axis=0),
        'p16_profile': np.percentile(profile_matrix, 16, axis=0),
        'p84_profile': np.percentile(profile_matrix, 84, axis=0),
    }


def export_tng_radial_profiles_txt(
    snapNum,
    simulation_set='TNG50-1',
    base_dir='/home/zwu/21cm_project/unified_model/TNG_results/',
    psi_thresholds=PSI_THRESHOLDS_FOR_EXPORT,
    host_mass_key='GroupMass',
    psi_mass_key='GroupMass',
    num_M_bins=5,
    x_min=1.0e-2,
    x_max=3.0,
    num_x_bins=30,
    log_x_bins=True,
    output_filename=None,
):
    """
    Export TNG radial subhalo profiles to one block-structured txt file.

    The file is organized as:
    snapshot header -> psi_min block -> mass-bin block -> binned profile rows.
    """
    processed_file = os.path.join(
        base_dir,
        simulation_set,
        f'snap_{snapNum}',
        f'processed_halos_snap_{snapNum}.h5'
    )
    output_dir = os.path.join(base_dir, simulation_set, f'snap_{snapNum}', 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    data = load_processed_data(processed_file)
    context = _build_tng_radial_profile_context(
        data,
        host_mass_key=host_mass_key,
        psi_mass_key=psi_mass_key,
        num_M_bins=num_M_bins,
        x_min=x_min,
        x_max=x_max,
        num_x_bins=num_x_bins,
        log_x_bins=log_x_bins,
    )
    _, dark_matter_resolution = get_simulation_resolution(simulation_set)

    if output_filename is None:
        output_filename = f'tng_radial_profiles_allpsi_snap_{snapNum}.txt'
    output_path = os.path.join(output_dir, output_filename)

    print(f'Calculating individual radial profile and Exporting TNG radial profiles to txt for snap {snapNum} ...')
    with open(output_path, 'w') as f:
        f.write(f'# snapshot {snapNum}\n')
        f.write(f'# redshift {data.header.get("Redshift", np.nan):.6f}\n')
        f.write(f'# simulation_set {simulation_set}\n')
        f.write('# subhalo_definition SUBFIND\n')
        f.write('# profile_definition dN/dx^3 with x=d_sub-host/R200\n')
        f.write(f'# host_mass_key {host_mass_key}\n')
        f.write(f'# psi_mass_key {psi_mass_key}\n')
        f.write(f'# num_host_mass_bins {num_M_bins}\n')
        f.write(f'# dark_matter_resolution_Msunh {dark_matter_resolution:.8e}\n')
        f.write('\n')

        for psi_min in psi_thresholds:
            f.write(f'# psi_min {psi_min:.8e}\n')
            valid_subs = (
                context['valid_subs_base']
                & np.isfinite(context['psi_all'])
                & (context['psi_all'] >= psi_min)
            )
            selected_host_indices = context['host_indices_for_subs'][valid_subs]
            selected_x_values = context['dpos_over_R200'][valid_subs]

            for i in range(context['num_M_bins']):
                host_ids = context['host_ids_by_bin'][i]
                n_hosts_total = len(host_ids)
                host_mass_min = 10**context['host_mass_bins'][i]
                host_mass_max = 10**context['host_mass_bins'][i + 1]
                crit_psi = 50.0 * dark_matter_resolution / host_mass_min

                stats = _compute_host_profile_statistics(
                    host_ids=host_ids,
                    host_local_index=context['host_local_index_by_bin'][i],
                    sub_host_indices=selected_host_indices,
                    sub_x_values=selected_x_values,
                    x_edges=context['x_edges'],
                    dx3=context['dx3'],
                )
                n_hosts_with_subhalos = stats['n_hosts_with_subhalos']
                n_subhalos = stats['n_subhalos']
                mean_profile = stats['mean_profile']
                median_profile = stats['median_profile']
                p16_profile = stats['p16_profile']
                p84_profile = stats['p84_profile']

                f.write(f'# mass_bin_index {i}\n')
                f.write(f'# host_mass_min_Msunh {host_mass_min:.8e}\n')
                f.write(f'# host_mass_max_Msunh {host_mass_max:.8e}\n')
                f.write(f'# crit_psi {crit_psi:.8e}\n')
                f.write(f'# psi_min_over_crit_psi {psi_min / crit_psi:.8e}\n')
                f.write(f'# n_hosts_total {n_hosts_total}\n')
                f.write(f'# n_hosts_with_subhalos {n_hosts_with_subhalos}\n')
                f.write(f'# n_subhalos {n_subhalos}\n')
                f.write(
                    '# columns: x_left x_right x_center '
                    'mean_dN_dx3 median_dN_dx3 p16_dN_dx3 p84_dN_dx3\n'
                )
                for k in range(context['num_x_bins']):
                    f.write(
                        f'{context["x_edges"][k]:.8e} '
                        f'{context["x_edges"][k + 1]:.8e} '
                        f'{context["x_centers"][k]:.8e} '
                        f'{mean_profile[k]:.8e} '
                        f'{median_profile[k]:.8e} '
                        f'{p16_profile[k]:.8e} '
                        f'{p84_profile[k]:.8e}\n'
                    )
                f.write('\n')
            f.write('\n')

    print(f'Saved TNG radial profile table: {output_path}')
    return output_path


def _is_data_row(line):
    stripped = line.strip()
    return stripped != '' and not stripped.startswith('#')


def _format_log_mass_range_label(prefix, mass_min, mass_max):
    return f'{prefix} [{np.log10(mass_min):.1f}, {np.log10(mass_max):.1f}]'


def _append_count_label(
    label,
    n_hosts_total=None,
    n_hosts_with_subhalos=None,
    n_subhalos=None,
):
    """Append host/subhalo counts to a legend label when available."""
    suffix_parts = []
    if n_hosts_total is not None:
        suffix_parts.append(f'Nhost,total={n_hosts_total}')
    if n_hosts_with_subhalos is not None:
        suffix_parts.append(f'Nhost,sub={n_hosts_with_subhalos}')
    if n_subhalos is not None:
        suffix_parts.append(f'Nsub={n_subhalos}')
    if not suffix_parts:
        return label
    return f'{label}, ' + ', '.join(suffix_parts)


def load_pop2prime_profile_blocks(filepath):
    """Parse a Pop2Prime radial-profile export txt file."""
    filepath = Path(filepath)
    metadata = {}
    blocks = {}
    current_psi = None

    with filepath.open('r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == '':
                continue
            if line.startswith('#'):
                content = line[1:].strip()
                if content.startswith('psi_min '):
                    current_psi = float(content.split()[1])
                    blocks[current_psi] = {'metadata': {'psi_min': current_psi}, 'rows': []}
                elif current_psi is not None:
                    if content.startswith('n_hosts_total '):
                        blocks[current_psi]['metadata']['n_hosts_total'] = int(content.split()[1])
                    elif content.startswith('n_hosts_with_subhalos '):
                        blocks[current_psi]['metadata']['n_hosts_with_subhalos'] = int(content.split()[1])
                    elif content.startswith('n_subhalos '):
                        blocks[current_psi]['metadata']['n_subhalos'] = int(content.split()[1])
                elif not content.startswith('columns:'):
                    parts = content.split(maxsplit=1)
                    if len(parts) == 2:
                        metadata[parts[0]] = parts[1]
                continue

            if current_psi is not None and _is_data_row(line):
                blocks[current_psi]['rows'].append([float(x) for x in line.split()])

    for psi_min, block in blocks.items():
        rows = np.array(block['rows'], dtype=float)
        block['table'] = {
            'x_left': rows[:, 0],
            'x_right': rows[:, 1],
            'x_center': rows[:, 2],
            'mean_dN_dx3': rows[:, 3],
            'median_dN_dx3': rows[:, 4],
            'p16_dN_dx3': rows[:, 5],
            'p84_dN_dx3': rows[:, 6],
        }
    return {'metadata': metadata, 'blocks': blocks}


def load_tng_profile_blocks(filepath):
    """Parse a TNG radial-profile export txt file."""
    filepath = Path(filepath)
    metadata = {}
    psi_blocks = {}
    current_psi = None
    current_mass_bin = None

    with filepath.open('r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == '':
                continue
            if line.startswith('#'):
                content = line[1:].strip()
                if content.startswith('psi_min '):
                    current_psi = float(content.split()[1])
                    psi_blocks[current_psi] = {'mass_bins': {}}
                    current_mass_bin = None
                elif content.startswith('mass_bin_index ') and current_psi is not None:
                    current_mass_bin = int(content.split()[1])
                    psi_blocks[current_psi]['mass_bins'][current_mass_bin] = {
                        'metadata': {'mass_bin_index': current_mass_bin},
                        'rows': [],
                    }
                elif current_psi is not None and current_mass_bin is not None:
                    block_meta = psi_blocks[current_psi]['mass_bins'][current_mass_bin]['metadata']
                    if content.startswith('host_mass_min_Msunh '):
                        block_meta['host_mass_min_Msunh'] = float(content.split()[1])
                    elif content.startswith('host_mass_max_Msunh '):
                        block_meta['host_mass_max_Msunh'] = float(content.split()[1])
                    elif content.startswith('crit_psi '):
                        block_meta['crit_psi'] = float(content.split()[1])
                    elif content.startswith('psi_min_over_crit_psi '):
                        block_meta['psi_min_over_crit_psi'] = float(content.split()[1])
                    elif content.startswith('n_hosts_total '):
                        block_meta['n_hosts_total'] = int(content.split()[1])
                    elif content.startswith('n_hosts_with_subhalos '):
                        block_meta['n_hosts_with_subhalos'] = int(content.split()[1])
                    elif content.startswith('n_subhalos '):
                        block_meta['n_subhalos'] = int(content.split()[1])
                elif current_psi is None and not content.startswith('columns:'):
                    parts = content.split(maxsplit=1)
                    if len(parts) == 2:
                        metadata[parts[0]] = parts[1]
                continue

            if current_psi is not None and current_mass_bin is not None and _is_data_row(line):
                psi_blocks[current_psi]['mass_bins'][current_mass_bin]['rows'].append(
                    [float(x) for x in line.split()]
                )

    for psi_min, psi_block in psi_blocks.items():
        for mass_bin_index, mass_bin_block in psi_block['mass_bins'].items():
            rows = np.array(mass_bin_block['rows'], dtype=float)
            mass_bin_block['table'] = {
                'x_left': rows[:, 0],
                'x_right': rows[:, 1],
                'x_center': rows[:, 2],
                'mean_dN_dx3': rows[:, 3],
                'median_dN_dx3': rows[:, 4],
                'p16_dN_dx3': rows[:, 5],
                'p84_dN_dx3': rows[:, 6],
            }
    return {'metadata': metadata, 'blocks': psi_blocks}


def plot_pop2prime_tng_radial_profile_comparison(
    # pop2prime_file='/home/zwu/21cm_project/unified_model/Pop2prime_results/pop2prime_radial_profiles_geometric_allpsi_DD0525.txt',
    pop2prime_file=None,
    tng_file='/home/zwu/21cm_project/unified_model/TNG_results/TNG50-1/snap_13/analysis/tng_radial_profiles_allpsi_snap_13.txt',
    output_dir=POP2PRIME_TNG_COMPARISON_DIR,
    z_label=6,
):
    """
    Plot Pop2Prime and TNG radial-profile means in three psi panels.

    TNG curves are shown only for resolved host-mass bins, defined by
    psi_min >= crit_psi.  Pop2Prime is shown as one mean curve per psi_min.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pop_data = None
    if pop2prime_file is not None:
        pop2prime_path = Path(pop2prime_file)
        if pop2prime_path.exists():
            pop_data = load_pop2prime_profile_blocks(pop2prime_path)
        else:
            print(f'Pop2Prime radial-profile file not found, skipping Pop2Prime curves: {pop2prime_path}')
    tng_data = load_tng_profile_blocks(tng_file)

    if pop_data is not None:
        pop_mass_min = float(pop_data['metadata'].get('host_mass_min_Msunh', 'nan'))
        pop_mass_max = float(pop_data['metadata'].get('host_mass_max_Msunh', 'nan'))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True, facecolor='white')
    tng_colors = plt.cm.rainbow(np.linspace(0, 1, 5))

    for ax, psi_min in zip(axes, PSI_THRESHOLDS_FOR_EXPORT):
        pop_block = None if pop_data is None else pop_data['blocks'].get(psi_min)
        if pop_block is not None:
            pop_table = pop_block['table']
            ax.plot(
                pop_table['x_center'],
                pop_table['mean_dN_dx3'],
                color='black',
                linewidth=2.4,
                label=_append_count_label(
                    _format_log_mass_range_label('Pop2Prime', pop_mass_min, pop_mass_max),
                    pop_block['metadata'].get('n_hosts_total'),
                    pop_block['metadata'].get('n_hosts_with_subhalos'),
                    pop_block['metadata'].get('n_subhalos'),
                ),
            )

        tng_mass_bins = tng_data['blocks'].get(psi_min, {}).get('mass_bins', {})
        for mass_bin_index in sorted(tng_mass_bins.keys()):
            mass_bin_block = tng_mass_bins[mass_bin_index]
            metadata = mass_bin_block['metadata']
            if psi_min < metadata['crit_psi']:
                continue
            table = mass_bin_block['table']
            ax.plot(
                table['x_center'],
                table['mean_dN_dx3'],
                color=tng_colors[mass_bin_index],
                linewidth=1.8,
                label=_append_count_label(
                    _format_log_mass_range_label(
                        'TNG',
                        metadata['host_mass_min_Msunh'],
                        metadata['host_mass_max_Msunh'],
                    ),
                    metadata.get('n_hosts_total'),
                    metadata.get('n_hosts_with_subhalos'),
                    metadata.get('n_subhalos'),
                ),
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.axvline(1.0, color='black', linestyle=':', linewidth=1.2)
        ax.set_xlabel(r'$x=d_{\mathrm{sub-host}}/R$', fontsize=13)
        ax.set_title(rf'$\psi > {psi_min:.0e}$', fontsize=13)
        ax.tick_params(direction='in', which='both', labelsize=11)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    axes[0].set_ylabel(r'$\mathrm{d}N_{\mathrm{sub}}/\mathrm{d}x^3$', fontsize=14)
    fig.suptitle(
        f'Pop2Prime vs TNG radial subhalo profile, z={z_label}',
        fontsize=13,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))

    output_path = output_dir / f'pop2prime_tng_radial_profile_comparison_mean_z{z_label}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved Pop2Prime-TNG comparison plot: {output_path}')
    return output_path


def plot_host_averaged_radial_subhalo_profile(
    data,
    snapNum,
    output_dir,
    host_mass_key='GroupMass',
    psi_mass_key='GroupMass',
    num_M_bins=5,
    x_min=1.0e-2,
    x_max=3.0,
    num_x_bins=30,
    log_x_bins=True,
    psi_range=None,
    profile_tag=None,
    save_prefix='radial_subhalo_profile'
):
    """
    Plot host-averaged subhalo radial profile dN/dx^3 with x=d_sub-host/R200.

    Each host is histogrammed separately, including hosts with zero subhalos in a
    given host-mass bin. The plotted mean/median are then taken across hosts.
    """
    print(f"Plotting host-averaged radial subhalo profile for snap {snapNum} ...")
    os.makedirs(output_dir, exist_ok=True)

    host_indices_for_subs = data.subhalo_data['host_index'].value.astype(int)
    host_masses_all = data.halo_data[host_mass_key].value
    psi_host_masses_all = data.halo_data[psi_mass_key].value
    sub_masses = data.subhalo_data['SubMass'].value
    psi_host_masses_for_subs = psi_host_masses_all[host_indices_for_subs]
    dpos_over_R200 = get_subhalo_host_distance_over_R200(data)

    valid_subs = (
        np.isfinite(dpos_over_R200)
        & (dpos_over_R200 > 0)
        & np.isfinite(psi_host_masses_for_subs)
        & (psi_host_masses_for_subs > 0)
    )
    if psi_range is not None:
        psi = sub_masses / psi_host_masses_for_subs
        valid_subs &= np.isfinite(psi) & (psi >= psi_range[0]) & (psi < psi_range[1])

    valid_hosts = np.isfinite(host_masses_all) & (host_masses_all > 0)
    host_logM_all = np.log10(host_masses_all)
    logM_min = np.min(host_logM_all[valid_hosts])
    logM_max = np.max(host_logM_all[valid_hosts])
    host_mass_bins = np.linspace(logM_min, logM_max, num_M_bins + 1)

    if log_x_bins:
        x_edges = np.logspace(np.log10(x_min), np.log10(x_max), num_x_bins + 1)
    else:
        x_edges = np.linspace(x_min, x_max, num_x_bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    dx3 = x_edges[1:]**3 - x_edges[:-1]**3

    colors = plt.cm.rainbow(np.linspace(0, 1, num_M_bins))
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')

    summary_rows = []
    artificial_small = 1.0e-12

    for i in range(num_M_bins):
        is_last = i == num_M_bins - 1
        if is_last:
            host_mask = (
                valid_hosts
                & (host_logM_all >= host_mass_bins[i])
                & (host_logM_all <= host_mass_bins[i + 1])
            )
        else:
            host_mask = (
                valid_hosts
                & (host_logM_all >= host_mass_bins[i])
                & (host_logM_all < host_mass_bins[i + 1])
            )
        host_ids = np.where(host_mask)[0]
        n_hosts = len(host_ids)
        if n_hosts == 0:
            continue

        profile_matrix = np.zeros((n_hosts, num_x_bins))
        for j, host_id in enumerate(host_ids):
            sub_mask = valid_subs & (host_indices_for_subs == host_id)
            counts, _ = np.histogram(dpos_over_R200[sub_mask], bins=x_edges)
            profile_matrix[j, :] = counts / dx3

        mean_profile = np.mean(profile_matrix, axis=0)
        median_profile = np.median(profile_matrix, axis=0)
        p16_profile = np.percentile(profile_matrix, 16, axis=0)
        p84_profile = np.percentile(profile_matrix, 84, axis=0)

        plot_mean = np.where(mean_profile > 0, mean_profile, artificial_small)
        plot_median = np.where(median_profile > 0, median_profile, artificial_small)
        label = rf'${host_mass_bins[i]:.1f}<\log_{{10}}(M_{{200}}/M_\odot h^{{-1}})<{host_mass_bins[i+1]:.1f}$'

        ax.plot(x_centers, plot_mean, color=colors[i], linewidth=2.0, label=label + ' mean')
        ax.plot(x_centers, plot_median, color=colors[i], linewidth=1.6, linestyle='--', label=label + ' median')
        ax.fill_between(
            x_centers,
            np.where(p16_profile > 0, p16_profile, artificial_small),
            np.where(p84_profile > 0, p84_profile, artificial_small),
            color=colors[i],
            alpha=0.15,
            linewidth=0,
        )

        n_subs = int(np.sum(valid_subs & np.isin(host_indices_for_subs, host_ids)))
        for k in range(num_x_bins):
            summary_rows.append([
                i, host_mass_bins[i], host_mass_bins[i + 1], n_hosts, n_subs,
                x_edges[k], x_edges[k + 1], x_centers[k],
                mean_profile[k], median_profile[k], p16_profile[k], p84_profile[k]
            ])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$x=d_{\mathrm{sub-host}}/R_{200}$', fontsize=14)
    ax.set_ylabel(r'$\mathrm{d}N_{\mathrm{sub}}/\mathrm{d}x^3$', fontsize=14)
    ax.axvline(1.0, color='black', linestyle=':', linewidth=1.5)
    redshift = data.header.get('Redshift', np.nan)
    title = f'snap {snapNum}, z={redshift:.2f}'
    if psi_range is not None:
        title += rf', ${psi_range[0]:.1e}<\psi<{psi_range[1]:.1e}$'
    ax.set_title(title, fontsize=13)
    ax.tick_params(direction='in', which='both', labelsize=12)
    ax.legend(fontsize=8, ncol=1)
    plt.tight_layout()

    tag = get_profile_tag(psi_range=psi_range, profile_tag=profile_tag)
    tag_suffix = '' if tag == '' else f'_{tag}'
    filename = os.path.join(output_dir, f'{save_prefix}{tag_suffix}_snap_{snapNum}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved radial subhalo profile: {filename}")

    summary_file = filename.replace('.png', '.txt')
    header = (
        'mass_bin_index logM_left logM_right n_hosts n_subs '
        'x_left x_right x_center mean_dN_dx3 median_dN_dx3 p16_dN_dx3 p84_dN_dx3'
        f' host_mass_key={host_mass_key} psi_mass_key={psi_mass_key}'
    )
    np.savetxt(summary_file, np.array(summary_rows), header=header)
    print(f"Saved radial subhalo profile data: {summary_file}")


def run_subhalo_number_profile(
    snapNum_list=None,
    simulation_set='TNG50-1',
    base_dir='/home/zwu/21cm_project/unified_model/TNG_results/',
    psi_range=(0.05, 1.0),
    profile_tag='psi0p05_1',
    profile_kwargs=None,
):
    """
    Main driver for host-averaged subhalo number radial profiles.

    This intentionally handles only the subhalo number profile for now; future
    Mach, velocity, angular-momentum, and heating-weighted profile drivers can
    be added as separate functions.
    """
    if snapNum_list is None:
        snapNum_list = [99, 13, 2, 1]
    if profile_kwargs is None:
        profile_kwargs = {}
    profile_kwargs = dict(profile_kwargs)
    profile_kwargs.setdefault('psi_range', psi_range)
    profile_kwargs.setdefault('profile_tag', profile_tag)

    for snapNum in snapNum_list:
        print(f"Processing snapshot {snapNum} ...")
        processed_file = os.path.join(
            base_dir,
            simulation_set,
            f'snap_{snapNum}',
            f'processed_halos_snap_{snapNum}.h5'
        )
        output_dir = os.path.join(base_dir, simulation_set, f'snap_{snapNum}', 'analysis')
        data = load_processed_data(processed_file)
        plot_host_averaged_radial_subhalo_profile(
            data,
            snapNum,
            output_dir,
            **profile_kwargs
        )


if __name__ == '__main__':
    # run_subhalo_number_profile()
    # export_tng_radial_profiles_txt(snapNum=99)
    plot_pop2prime_tng_radial_profile_comparison()
