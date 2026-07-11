import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from TNGDataHandler import get_simulation_resolution
from TNGDataHandler import load_processed_data
from physical_constants import h_Hubble

try:
    from colossus.halo import concentration as colossus_concentration
except ModuleNotFoundError:
    colossus_concentration = None


def normalize_tng_halo_definition(radius_definition):
    """Return a canonical TNG halo-definition tag: '200c' or '200m'."""
    value = str(radius_definition).strip().lower()
    aliases = {
        '200c': '200c',
        'crit200': '200c',
        'm200c': '200c',
        'r200c': '200c',
        '200m': '200m',
        'mean200': '200m',
        'm200m': '200m',
        'r200m': '200m',
    }
    if value not in aliases:
        raise ValueError(f'Unknown radius_definition: {radius_definition}')
    return aliases[value]


def get_tng_halo_definition_keys(radius_definition='200c'):
    """Return the processed-data mass/radius keys for one TNG halo definition."""
    normalized = normalize_tng_halo_definition(radius_definition)
    if normalized == '200c':
        return 'Group_M_Crit200', 'Group_R_Crit200'
    return 'Group_M_Mean200', 'Group_R_Mean200'


def get_tng_halo_definition_metadata(radius_definition='200c'):
    """Return reusable metadata for a TNG halo definition."""
    normalized = normalize_tng_halo_definition(radius_definition)
    mass_key, radius_key = get_tng_halo_definition_keys(normalized)
    if normalized == '200c':
        return {
            'definition': normalized,
            'mass_key': mass_key,
            'radius_key': radius_key,
            'mass_label': 'M200c',
            'radius_label': 'R200c',
            'v_label': 'V200c',
            'tag': '200c',
        }
    return {
        'definition': normalized,
        'mass_key': mass_key,
        'radius_key': radius_key,
        'mass_label': 'M200m',
        'radius_label': 'R200m',
        'v_label': 'V200m',
        'tag': '200m',
    }


def minimum_image_displacement(pos, center, box_size):
    """Return displacement vectors in ckpc/h with periodic minimum-image wrapping."""
    delta = pos - center
    if box_size is not None and box_size > 0:
        delta -= box_size * np.round(delta / box_size)
    return delta


def get_subhalo_host_distance(data, radius_definition='200c'):
    """
    Return host-centric subhalo distance vectors and normalized radii.

    SubhaloPos and GroupPos are comoving ckpc/h. The selected host radius is
    stored as a physical Mpc quantity in processed TNG data.
    """
    halo_meta = get_tng_halo_definition_metadata(radius_definition)
    host_indices = data.subhalo_data['host_index'].value
    subhalo_pos = data.subhalo_data['SubPos'].value
    host_pos = data.halo_data['GroupPos'].value[host_indices]
    host_R200 = data.halo_data[halo_meta['radius_key']].value[host_indices]

    box_size = data.header.get('BoxSize', None)
    scale_factor = data.header.get('Time', 1.0)
    dpos_vec_ckpch = minimum_image_displacement(subhalo_pos, host_pos, box_size)
    dpos_ckpch = np.sqrt(np.sum(dpos_vec_ckpch**2, axis=1))
    dpos_phys_mpc = dpos_ckpch / 1.0e3 * scale_factor / h_Hubble
    dpos_over_R200 = dpos_phys_mpc / host_R200
    return dpos_vec_ckpch, dpos_phys_mpc, dpos_over_R200


def get_subhalo_host_distance_over_R200(data, radius_definition='200c'):
    """Return d_sub-host/R200 for processed TNG subhalos."""
    return get_subhalo_host_distance(data, radius_definition=radius_definition)[2]


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


def _get_statistic_metadata(statistic):
    """Return normalization metadata for a supported radial-profile statistic."""
    metadata = {
        'count_dx3': {
            'ylabel': r'$\mathrm{d}N_{\mathrm{sub}}/\mathrm{d}\xi^3$',
            'column_label': 'dN_dx3',
            'profile_definition': 'dN/dxi^3 with xi=d_sub-host/R200',
            'title_label': 'count-per-volume-weighted',
        },
        'count_dx': {
            'ylabel': r'$\mathrm{d}N_{\mathrm{sub}}/\mathrm{d}x$',
            'column_label': 'dN_dx',
            'profile_definition': 'dN/dx with x=d_sub-host/R200',
            'title_label': 'count-weighted',
        },
        'mass_dx': {
            'ylabel': r'$\mathrm{d}\sum (m_{\mathrm{sub}}/M_{\mathrm{host}})/\mathrm{d}x$',
            'column_label': 'mass_dx',
            'profile_definition': 'd/dx sum(m_sub/M_host) with x=d_sub-host/R200',
            'title_label': 'mass-weighted',
        },
        'mass2_dx': {
            'ylabel': r'$\mathrm{d}\sum (m_{\mathrm{sub}}/M_{\mathrm{host}})^2/\mathrm{d}x$',
            'column_label': 'mass2_dx',
            'profile_definition': 'd/dx sum((m_sub/M_host)^2) with x=d_sub-host/R200',
            'title_label': 'mass-squared-weighted',
        },
    }
    if statistic not in metadata:
        raise ValueError(f'Unknown statistic: {statistic}')
    return metadata[statistic]


def _interpolate_profile_at_x(x_centers, profile, x_target=1.0):
    """Estimate one profile value at x_target from neighboring radial bins."""
    x_centers = np.asarray(x_centers, dtype=float)
    profile = np.asarray(profile, dtype=float)
    valid = np.isfinite(x_centers) & np.isfinite(profile) & (profile > 0)
    if np.count_nonzero(valid) == 0:
        return np.nan

    x_valid = x_centers[valid]
    profile_valid = profile[valid]
    if x_target <= x_valid[0]:
        return profile_valid[0]
    if x_target >= x_valid[-1]:
        return profile_valid[-1]

    return np.interp(x_target, x_valid, profile_valid)


def _normalize_profile_at_x(x_centers, profile, x_target=1.0):
    """Return profile/profile(x_target), keeping invalid cases as NaN."""
    normalization = _interpolate_profile_at_x(x_centers, profile, x_target=x_target)
    if not np.isfinite(normalization) or normalization <= 0:
        return np.full_like(profile, np.nan, dtype=float), normalization
    return np.asarray(profile, dtype=float) / normalization, normalization


def _f_nfw(x):
    """Return the standard NFW helper f(x)=ln(1+x)-x/(1+x)."""
    x = np.asarray(x, dtype=float)
    return np.log1p(x) - x / (1.0 + x)


def _get_subhalo_reference_concentration(M_mid_msunh, redshift, radius_definition, model='diemer19'):
    """Return c(M,z) for the requested halo definition using Colossus."""
    if colossus_concentration is None:
        return np.nan
    halo_meta = get_tng_halo_definition_metadata(radius_definition)
    try:
        return float(
            colossus_concentration.concentration(
                M_mid_msunh,
                halo_meta['tag'],
                redshift,
                model=model,
                range_return=False,
            )
        )
    except Exception:
        return np.nan


def _get_normalized_nfw_subhalo_reference(
    x_values,
    concentration,
    normalize_at_vir=True,
    jiang_eta=2.0,
    jiang_mu=4.0,
    han16_gamma=1.33,
):
    r"""
    Return normalized NFW, Jiang+vdB, and Han16 subhalo number-density shapes.

    The baseline dN/dx^3_NFW is taken to follow the NFW density shape,
    rho(x) \propto 1 / [(c x) (1 + c x)^2], with x=r/Rvir.
    """
    x_values = np.asarray(x_values, dtype=float)
    if not np.isfinite(concentration) or concentration <= 0:
        nan_profile = np.full_like(x_values, np.nan, dtype=float)
        return nan_profile, nan_profile, nan_profile

    cx = concentration * x_values
    nfw_profile = concentration**3 / (3.0 * _f_nfw(concentration))
    nfw_profile /= cx * (1.0 + cx) ** 2

    jiang_factor = (2.0 ** jiang_mu) * x_values ** jiang_eta / (1.0 + x_values) ** jiang_mu
    jiang_modified_profile = nfw_profile * jiang_factor
    han16_modified_profile = nfw_profile * x_values ** han16_gamma

    if normalize_at_vir:
        nfw_profile, _ = _normalize_profile_at_x(x_values, nfw_profile, x_target=1.0)
        jiang_modified_profile, _ = _normalize_profile_at_x(
            x_values,
            jiang_modified_profile,
            x_target=1.0,
        )
        han16_modified_profile, _ = _normalize_profile_at_x(
            x_values,
            han16_modified_profile,
            x_target=1.0,
        )

    return nfw_profile, jiang_modified_profile, han16_modified_profile


PSI_THRESHOLDS_FOR_EXPORT = (1.0e-3, 1.0e-2, 5.0e-2)
POP2PRIME_TNG_COMPARISON_DIR = Path(
    '/home/zwu/21cm_project/unified_model/Pop2prime_results/Pop2Prime_TNG_comparison'
)


def _build_tng_radial_profile_context(
    data,
    host_mass_key='GroupMass',
    psi_mass_key='GroupMass',
    radius_definition='200c',
    num_M_bins=5,
    x_min=1.0e-2,
    x_max=3.0,
    num_x_bins=30,
    log_x_bins=True,
):
    """Precompute reusable arrays for TNG radial-profile exports."""
    halo_meta = get_tng_halo_definition_metadata(radius_definition)
    host_indices_for_subs = data.subhalo_data['host_index'].value.astype(int)
    host_masses_all = data.halo_data[host_mass_key].value
    psi_host_masses_all = data.halo_data[psi_mass_key].value
    sub_masses = data.subhalo_data['SubMass'].value
    psi_host_masses_for_subs = psi_host_masses_all[host_indices_for_subs]
    dpos_over_R200 = get_subhalo_host_distance_over_R200(
        data,
        radius_definition=radius_definition,
    )

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
        'sub_masses': sub_masses,
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
        'radius_definition': halo_meta['definition'],
        'radius_key': halo_meta['radius_key'],
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
    statistic='count_dx3',
    sub_weights=None,
    percentile_mode='all_hosts',
):
    """Compute per-host radial-profile summary statistics without per-host histograms."""
    n_hosts_total = len(host_ids)
    num_x_bins = len(dx3)
    if n_hosts_total == 0:
        zeros = np.zeros(num_x_bins)
        return {
            'profile_matrix': np.zeros((0, num_x_bins), dtype=float),
            'percentile_matrix': np.zeros((0, num_x_bins), dtype=float),
            'n_hosts_with_subhalos': 0,
            'n_subhalos': 0,
            'mean_profile': zeros,
            'median_profile': zeros,
            'p16_profile': zeros,
            'p84_profile': zeros,
        }

    profile_matrix = np.zeros((n_hosts_total, num_x_bins), dtype=float)
    if statistic == 'count_dx3':
        normalization = dx3
    elif statistic in {'count_dx', 'mass_dx', 'mass2_dx'}:
        normalization = x_edges[1:] - x_edges[:-1]
    else:
        raise ValueError(f'Unknown statistic: {statistic}')
    if percentile_mode not in {'all_hosts', 'occupied_hosts'}:
        raise ValueError(f'Unknown percentile_mode: {percentile_mode}')

    if sub_x_values.size > 0:
        x_bin_indices = np.searchsorted(x_edges, sub_x_values, side='right') - 1
        valid_x = (x_bin_indices >= 0) & (x_bin_indices < num_x_bins)
        if np.any(valid_x):
            local_host_indices = host_local_index[sub_host_indices[valid_x]]
            valid_host = local_host_indices >= 0
            if np.any(valid_host):
                weights = 1.0 if sub_weights is None else sub_weights[valid_x][valid_host]
                #sum weights to the appropriate host and x-bin in the profile matrix
                np.add.at(
                    profile_matrix,
                    (local_host_indices[valid_host], x_bin_indices[valid_x][valid_host]),
                    weights,
                )

    hosts_with_subhalos = int(np.count_nonzero(np.any(profile_matrix > 0, axis=1)))
    profile_matrix /= normalization[None, :]
    percentile_matrix = profile_matrix
    if percentile_mode == 'occupied_hosts':
        occupied_mask = np.any(profile_matrix > 0, axis=1)
        percentile_matrix = profile_matrix[occupied_mask]
    if percentile_matrix.shape[0] == 0:
        percentile_matrix = np.zeros((1, num_x_bins), dtype=float)
    return {
        'profile_matrix': profile_matrix,
        'percentile_matrix': percentile_matrix,
        'n_hosts_with_subhalos': hosts_with_subhalos,
        'n_subhalos': int(sub_x_values.size),
        'mean_profile': np.mean(profile_matrix, axis=0),
        'median_profile': np.median(percentile_matrix, axis=0),
        'p16_profile': np.percentile(percentile_matrix, 16, axis=0),
        'p84_profile': np.percentile(percentile_matrix, 84, axis=0),
    }


def export_tng_radial_profiles_txt(
    snapNum,
    simulation_set='TNG50-1',
    base_dir='/home/zwu/21cm_project/unified_model/TNG_results/',
    psi_thresholds=PSI_THRESHOLDS_FOR_EXPORT,
    host_mass_key='GroupMass',
    psi_mass_key='GroupMass',
    radius_definition='200c',
    num_M_bins=5,
    x_min=1.0e-2,
    x_max=3.0,
    num_x_bins=30,
    log_x_bins=True,
    output_filename=None,
    statistic='count_dx3',
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
    halo_meta = get_tng_halo_definition_metadata(radius_definition)
    context = _build_tng_radial_profile_context(
        data,
        host_mass_key=host_mass_key,
        psi_mass_key=psi_mass_key,
        radius_definition=radius_definition,
        num_M_bins=num_M_bins,
        x_min=x_min,
        x_max=x_max,
        num_x_bins=num_x_bins,
        log_x_bins=log_x_bins,
    )
    _, dark_matter_resolution = get_simulation_resolution(simulation_set)
    statistic_meta = _get_statistic_metadata(statistic)

    if output_filename is None:
        output_filename = f'tng_radial_profiles_allpsi_{halo_meta["tag"]}_snap_{snapNum}.txt'
    output_path = os.path.join(output_dir, output_filename)

    print(f'Calculating individual radial profile and Exporting TNG radial profiles to txt for snap {snapNum} ...')
    with open(output_path, 'w') as f:
        f.write(f'# snapshot {snapNum}\n')
        f.write(f'# redshift {data.header.get("Redshift", np.nan):.6f}\n')
        f.write(f'# simulation_set {simulation_set}\n')
        f.write('# subhalo_definition SUBFIND\n')
        f.write(f'# profile_definition {statistic_meta["profile_definition"]}\n')
        f.write(f'# statistic {statistic}\n')
        f.write(f'# host_mass_key {host_mass_key}\n')
        f.write(f'# psi_mass_key {psi_mass_key}\n')
        f.write(f'# radius_definition {halo_meta["definition"]}\n')
        f.write(f'# radius_key {halo_meta["radius_key"]}\n')
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
            selected_sub_weights = None
            if statistic == 'mass_dx':
                selected_sub_weights = (
                    context['sub_masses'][valid_subs]
                    / context['host_masses_all'][selected_host_indices]
                )
            elif statistic == 'mass2_dx':
                selected_sub_weights = (
                    context['sub_masses'][valid_subs]
                    / context['host_masses_all'][selected_host_indices]
                ) ** 2

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
                    statistic=statistic,
                    sub_weights=selected_sub_weights,
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
                    f'mean_{statistic_meta["column_label"]} '
                    f'median_{statistic_meta["column_label"]} '
                    f'p16_{statistic_meta["column_label"]} '
                    f'p84_{statistic_meta["column_label"]}\n'
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


def _format_paper_radial_profile_label(prefix, mass_min, mass_max, psi_min):
    return (
        rf'{prefix}: $10^{{{np.log10(mass_min):.1f}}}<M_{{\rm host}}/(M_\odot/h)<'
        rf'10^{{{np.log10(mass_max):.1f}}}$, $\psi\in[{psi_min:.0e},1]$'
    )



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
        statistic = psi_block.get('metadata', {}).get('statistic', metadata.get('statistic', 'count_dx3'))
        column_label = _get_statistic_metadata(statistic)['column_label']
        for mass_bin_index, mass_bin_block in psi_block['mass_bins'].items():
            rows = np.array(mass_bin_block['rows'], dtype=float)
            mass_bin_block['table'] = {
                'x_left': rows[:, 0],
                'x_right': rows[:, 1],
                'x_center': rows[:, 2],
                f'mean_{column_label}': rows[:, 3],
                f'median_{column_label}': rows[:, 4],
                f'p16_{column_label}': rows[:, 5],
                f'p84_{column_label}': rows[:, 6],
            }
    return {'metadata': metadata, 'blocks': psi_blocks}


def plot_pop2prime_tng_radial_profile_comparison(
    # pop2prime_file='/home/zwu/21cm_project/unified_model/Pop2prime_results/pop2prime_radial_profiles_geometric_allpsi_DD0525.txt',
    pop2prime_file=None,
    tng_file='/home/zwu/21cm_project/unified_model/TNG_results/TNG50-1/snap_13/analysis/tng_radial_profiles_allpsi_snap_13.txt',
    output_dir=POP2PRIME_TNG_COMPARISON_DIR,
    z_label=6,
    paper_style=False,
    output_suffix='',
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
                label=(
                    _format_paper_radial_profile_label('Pop2Prime', pop_mass_min, pop_mass_max, psi_min)
                    if paper_style
                    else _append_count_label(
                        _format_log_mass_range_label('Pop2Prime', pop_mass_min, pop_mass_max),
                        pop_block['metadata'].get('n_hosts_total'),
                        pop_block['metadata'].get('n_hosts_with_subhalos'),
                        pop_block['metadata'].get('n_subhalos'),
                    )
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
                label=(
                    _format_paper_radial_profile_label(
                        'TNG50',
                        metadata['host_mass_min_Msunh'],
                        metadata['host_mass_max_Msunh'],
                        psi_min,
                    )
                    if paper_style
                    else _append_count_label(
                        _format_log_mass_range_label(
                            'TNG',
                            metadata['host_mass_min_Msunh'],
                            metadata['host_mass_max_Msunh'],
                        ),
                        metadata.get('n_hosts_total'),
                        metadata.get('n_hosts_with_subhalos'),
                        metadata.get('n_subhalos'),
                    )
                ),
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.axvline(1.0, color='black', linestyle=':', linewidth=1.2)
        ax.set_xlabel(r'$\xi = r/R_{\rm vir}$', fontsize=13)
        ax.set_title(rf'$\psi > {psi_min:.0e}$', fontsize=13)
        ax.tick_params(direction='in', which='both', labelsize=11)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8.2 if paper_style else 8)

    axes[0].set_ylabel(r'$\mathrm{d}N_{\mathrm{sub}}/\mathrm{d}\xi^3$', fontsize=14)
    if not paper_style:
        fig.suptitle(
            f'Pop2Prime vs TNG radial subhalo profile, z={z_label}',
            fontsize=13,
        )
        layout_rect = (0, 0, 1, 0.94)
    else:
        layout_rect = (0, 0, 1, 1)
    plt.tight_layout(rect=layout_rect)

    output_path = output_dir / f'pop2prime_tng_radial_profile_comparison_mean_z{z_label}{output_suffix}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved Pop2Prime-TNG comparison plot: {output_path}')
    return output_path


def _finite_profile_points(x_values, y_values):
    """Return finite plotting points while preserving zero-valued bins."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[finite], y_values[finite]


def _normalize_profile_at_x_loglog(x_centers, profile, x_target=1.0):
    """Normalize a positive profile using log-log interpolation at x_target."""
    x_centers = np.asarray(x_centers, dtype=float)
    profile = np.asarray(profile, dtype=float)
    valid = (
        np.isfinite(x_centers)
        & np.isfinite(profile)
        & (x_centers > 0.0)
        & (profile > 0.0)
    )
    if np.count_nonzero(valid) == 0:
        return np.full_like(profile, np.nan, dtype=float), np.nan

    x_valid = x_centers[valid]
    profile_valid = profile[valid]
    if x_target <= x_valid[0]:
        normalization = profile_valid[0]
    elif x_target >= x_valid[-1]:
        normalization = profile_valid[-1]
    else:
        log_norm = np.interp(
            np.log10(x_target),
            np.log10(x_valid),
            np.log10(profile_valid),
        )
        normalization = 10.0**log_norm

    return profile / normalization, normalization


def _compute_tng_selected_count_dx3_profile(
    data,
    host_mass_min_select=1.0e10,
    psi_range=(2.0e-3, 1.0),
    radius_definition='200c',
    host_mass_key='GroupMass',
    psi_mass_key='GroupMass',
    x_min=1.0e-2,
    x_max=3.0,
    num_x_bins=30,
):
    """Compute the host-averaged TNG count/dx^3 profile for one selected sample."""
    host_indices = data.subhalo_data['host_index'].value.astype(int)
    host_masses = data.halo_data[host_mass_key].value
    psi_host_masses = data.halo_data[psi_mass_key].value
    sub_masses = data.subhalo_data['SubMass'].value
    host_masses_for_subs = host_masses[host_indices]
    psi_host_masses_for_subs = psi_host_masses[host_indices]
    dpos_over_R200 = get_subhalo_host_distance_over_R200(
        data,
        radius_definition=radius_definition,
    )

    valid_hosts = np.isfinite(host_masses) & (host_masses >= host_mass_min_select)
    host_ids = np.where(valid_hosts)[0]
    if host_ids.size == 0:
        raise ValueError('No TNG hosts pass the selected host-mass cut.')

    local_index = np.full(host_masses.shape[0], -1, dtype=int)
    local_index[host_ids] = np.arange(host_ids.size)
    psi = np.divide(
        sub_masses,
        psi_host_masses_for_subs,
        out=np.full_like(sub_masses, np.nan, dtype=float),
        where=np.isfinite(psi_host_masses_for_subs) & (psi_host_masses_for_subs > 0.0),
    )
    valid_subs = (
        np.isfinite(dpos_over_R200)
        & (dpos_over_R200 > 0.0)
        & np.isfinite(host_masses_for_subs)
        & (host_masses_for_subs > 0.0)
        & np.isfinite(psi_host_masses_for_subs)
        & (psi_host_masses_for_subs > 0.0)
        & valid_hosts[host_indices]
        & np.isfinite(psi)
        & (psi >= psi_range[0])
        & (psi < psi_range[1])
    )

    x_edges = np.logspace(np.log10(x_min), np.log10(x_max), num_x_bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    dx3 = x_edges[1:]**3 - x_edges[:-1]**3
    stats = _compute_host_profile_statistics(
        host_ids=host_ids,
        host_local_index=local_index,
        sub_host_indices=host_indices[valid_subs],
        sub_x_values=dpos_over_R200[valid_subs],
        x_edges=x_edges,
        dx3=dx3,
        statistic='count_dx3',
        percentile_mode='all_hosts',
    )

    return {
        'x_edges': x_edges,
        'x_centers': x_centers,
        'dx3': dx3,
        'stats': stats,
        'host_ids': host_ids,
        'host_masses': host_masses[host_ids],
        'selected_subhalo_count': int(np.count_nonzero(valid_subs)),
        'redshift': data.header.get('Redshift', np.nan),
    }


def plot_pop2prime_tng_radial_profile_comparison_selected_z12_paper(
    pop2prime_file='/home/zwu/21cm_project/unified_model/Pop2prime_results/pop2prime_radial_profiles_geometric_allpsi_DD0525.txt',
    tng_processed_file='/home/zwu/21cm_project/unified_model/TNG_results/TNG50-1/snap_2/processed_halos_snap_2.h5',
    output_dir='/home/zwu/21cm_project/unified_model/Profile_results_for_paper',
    output_filename='pop2prime_tng_radial_profile_comparison_z12.png',
    pop_psi_min=1.0e-3,
    tng_psi_range=(2.0e-3, 1.0),
    tng_host_mass_min_select=1.0e10,
    radius_definition='200c',
    tng_host_mass_key='GroupMass',
    tng_psi_mass_key='GroupMass',
    concentration_model='ludlow16',
    normalize_at_vir=True,
):
    """Plot the final z=12 Pop2Prime/TNG normalized dN_sub/dx^3 comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pop_data = load_pop2prime_profile_blocks(pop2prime_file)
    pop_psi_key = min(pop_data['blocks'], key=lambda value: abs(value - pop_psi_min))
    if not np.isclose(pop_psi_key, pop_psi_min):
        raise ValueError(f'No Pop2Prime block close to psi_min={pop_psi_min:g}.')
    pop_block = pop_data['blocks'][pop_psi_key]
    pop_table = pop_block['table']
    pop_x = pop_table['x_center']
    pop_mean = pop_table['mean_dN_dx3']
    pop_median = pop_table['median_dN_dx3']

    tng_data = load_processed_data(tng_processed_file)
    tng_profile = _compute_tng_selected_count_dx3_profile(
        tng_data,
        host_mass_min_select=tng_host_mass_min_select,
        psi_range=tng_psi_range,
        radius_definition=radius_definition,
        host_mass_key=tng_host_mass_key,
        psi_mass_key=tng_psi_mass_key,
    )
    tng_x = tng_profile['x_centers']
    tng_stats = tng_profile['stats']
    tng_mean = tng_stats['mean_profile']
    tng_median = tng_stats['median_profile']

    pop_mass_min = float(pop_data['metadata'].get('host_mass_min_Msunh', np.nan))
    pop_mass_max = float(pop_data['metadata'].get('host_mass_max_Msunh', np.nan))
    tng_mass_max = float(np.max(tng_profile['host_masses']))
    pop_mean_label = (
        rf'Pop2Prime mean: $N_{{\rm host}}={pop_block["metadata"].get("n_hosts_total")}$, '
        rf'$10^{{{np.log10(pop_mass_min):.1f}}}<M_{{\rm host}}/(M_\odot/h)'
        rf'<10^{{{np.log10(pop_mass_max):.1f}}}$, '
        rf'$\psi\in[{pop_psi_key:.0e},1]$'
    )
    tng_mean_label = (
        rf'TNG50 mean: $N_{{\rm host}}={len(tng_profile["host_ids"])}$, '
        rf'$10^{{{np.log10(tng_host_mass_min_select):.1f}}}<M_{{\rm host}}/(M_\odot/h)'
        rf'<10^{{{np.log10(tng_mass_max):.1f}}}$, '
        rf'$\psi\in[{tng_psi_range[0]:.0e},1]$'
    )

    if normalize_at_vir:
        pop_mean, pop_mean_norm = _normalize_profile_at_x(pop_x, pop_mean, x_target=1.0)
        pop_median, pop_median_norm = _normalize_profile_at_x(pop_x, pop_median, x_target=1.0)
        tng_mean, tng_mean_norm = _normalize_profile_at_x(tng_x, tng_mean, x_target=1.0)
        tng_median, tng_median_norm = _normalize_profile_at_x(tng_x, tng_median, x_target=1.0)
    else:
        pop_mean_norm = pop_median_norm = tng_mean_norm = tng_median_norm = np.nan

    reference_concentration = _get_subhalo_reference_concentration(
        np.mean(tng_profile['host_masses']),
        tng_profile['redshift'],
        radius_definition=radius_definition,
        model=concentration_model,
    )
    reference_nfw, _, _ = _get_normalized_nfw_subhalo_reference(
        tng_x,
        reference_concentration,
        normalize_at_vir=False,
    )
    if normalize_at_vir:
        reference_nfw, _ = _normalize_profile_at_x(tng_x, reference_nfw, x_target=1.0)

    fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor='white')
    plot_items = [
        (pop_x, pop_mean, {'color': '#1f77b4', 'linewidth': 2.3, 'label': pop_mean_label}),
        (tng_x, tng_mean, {'color': 'black', 'linewidth': 2.3, 'label': tng_mean_label}),
        (tng_x, reference_nfw, {'color': '#d62728', 'linewidth': 1.9, 'linestyle': '-.', 'label': rf'NFW, $c={reference_concentration:.2f}$'}),
    ]
    for x_values, y_values, style in plot_items:
        x_plot, y_plot = _finite_profile_points(x_values, y_values)
        if x_plot.size > 0:
            ax.plot(x_plot, y_plot, **style)

    scatter_items = [
        (pop_x, pop_median, {'edgecolors': '#1f77b4', 'label': 'Pop2Prime median'}),
        (tng_x, tng_median, {'edgecolors': 'black', 'label': 'TNG50 median'}),
    ]
    for x_values, y_values, style in scatter_items:
        x_plot, y_plot = _finite_profile_points(x_values, y_values)
        if x_plot.size == 0:
            continue
        ax.scatter(
            x_plot,
            y_plot,
            s=32,
            facecolors='white',
            linewidths=1.1,
            zorder=4,
            **style,
        )

    ax.axvline(1.0, color='0.45', linestyle=':', linewidth=1.0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1.0e-2, 3.0)
    ax.set_ylim(bottom=1.0e-3)
    ax.set_xlabel(r'$\xi = r/R_{200c}$', fontsize=13)
    ax.set_ylabel(r'Normalized $\mathrm{d}N_{\rm sub}/\mathrm{d}\xi^3$', fontsize=13)
    ax.tick_params(direction='in', which='both', labelsize=11)
    ax.legend(fontsize=8.5, frameon=True)
    plt.tight_layout()

    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved final z=12 Pop2Prime/TNG comparison plot: {output_path}')
    print(
        'Normalization at x=1: '
        f'Pop2Prime mean={pop_mean_norm:.6e}, median={pop_median_norm:.6e}; '
        f'TNG50 mean={tng_mean_norm:.6e}, median={tng_median_norm:.6e}'
    )
    print(
        'Selection: '
        f'Pop2Prime psi_min={pop_psi_key:g}; '
        f'TNG Mhost({tng_host_mass_key})>={tng_host_mass_min_select:.2e} Msun/h, '
        f'psi=SubMass/{tng_psi_mass_key} in [{tng_psi_range[0]:g}, {tng_psi_range[1]:g}), '
        f'Nhost={len(tng_profile["host_ids"])}, Nsub={tng_profile["selected_subhalo_count"]}'
    )
    print(
        f'NFW reference: c={reference_concentration:.3f}, '
        f'model={concentration_model}, z={tng_profile["redshift"]:.3f}'
    )
    return output_path


def plot_host_averaged_radial_subhalo_profile(
    data,
    snapNum,
    output_dir,
    host_mass_key='GroupMass',
    psi_mass_key='GroupMass',
    radius_definition='200c',
    num_M_bins=5,
    x_min=1.0e-2,
    x_max=3.0,
    num_x_bins=30,
    log_x_bins=True,
    psi_range=None,
    profile_tag=None,
    save_prefix='radial_subhalo_profile',
    apply_resolution_cut=None,
    show_average=None,
    show_percentile=None,
    show_median=None,
    show_individual=None,
    statistic=None,
    percentile_mode=None,
    normalize_at_vir=None,
    show_analytic_reference=None,
    analytic_concentration_model=None,
    jiang_eta=None,
    jiang_mu=None,
    han16_gamma=None,
    paper_style=False,
    han16_extra_gammas=(),
    output_suffix='',
):
    """
    Plot host-averaged subhalo radial profile with x=d_sub-host/R200.

    Each host is histogrammed separately, including hosts with zero subhalos in a
    given host-mass bin. The mean is always taken across all hosts. Analysis
    strategy parameters must be passed explicitly by the caller.
    """
    print(f"Plotting host-averaged radial subhalo profile for snap {snapNum} ...")
    os.makedirs(output_dir, exist_ok=True)
    required_args = {
        'psi_range': psi_range,
        'profile_tag': profile_tag,
        'apply_resolution_cut': apply_resolution_cut,
        'show_average': show_average,
        'show_percentile': show_percentile,
        'show_median': show_median,
        'show_individual': show_individual,
        'statistic': statistic,
        'percentile_mode': percentile_mode,
        'normalize_at_vir': normalize_at_vir,
        'show_analytic_reference': show_analytic_reference,
        'analytic_concentration_model': analytic_concentration_model,
        'jiang_eta': jiang_eta,
        'jiang_mu': jiang_mu,
        'han16_gamma': han16_gamma,
    }
    missing_args = [name for name, value in required_args.items() if value is None]
    if missing_args:
        raise ValueError(
            'plot_host_averaged_radial_subhalo_profile requires explicit values for: '
            + ', '.join(missing_args)
        )
    statistic_meta = _get_statistic_metadata(statistic)
    halo_meta = get_tng_halo_definition_metadata(radius_definition)

    host_indices_for_subs = data.subhalo_data['host_index'].value.astype(int)
    host_masses_all = data.halo_data[host_mass_key].value
    psi_host_masses_all = data.halo_data[psi_mass_key].value
    sub_masses = data.subhalo_data['SubMass'].value
    psi_host_masses_for_subs = psi_host_masses_all[host_indices_for_subs]
    dpos_over_R200 = get_subhalo_host_distance_over_R200(
        data,
        radius_definition=radius_definition,
    )

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

    _, dark_matter_resolution = get_simulation_resolution(
        data.header.get('SimulationName', 'TNG50-1')
    )
    colors = plt.cm.rainbow(np.linspace(0, 1, num_M_bins))
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')


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
        host_mass_min = 10**host_mass_bins[i]
        host_mass_max = 10**host_mass_bins[i + 1]
        host_mass_mid = 10**(0.5 * (host_mass_bins[i] + host_mass_bins[i + 1]))
        crit_psi = 50.0 * dark_matter_resolution / host_mass_min
        is_resolved = psi_range is None or psi_range[0] >= crit_psi
        print(
            f'mass bin {i}: logM=[{host_mass_bins[i]:.2f}, {host_mass_bins[i + 1]:.2f}], '
            f'crit_psi={crit_psi:.3e}, resolved={is_resolved}'
        )
        if apply_resolution_cut and not is_resolved:
            continue

        host_local_index = np.full(host_masses_all.shape[0], -1, dtype=int)
        host_local_index[host_ids] = np.arange(host_ids.size)
        sub_mask = valid_subs & np.isin(host_indices_for_subs, host_ids)
        sub_weights = None
        if statistic == 'mass_dx':
            sub_weights = sub_masses[sub_mask] / host_masses_all[host_indices_for_subs[sub_mask]]
        elif statistic == 'mass2_dx':
            sub_weights = (sub_masses[sub_mask] / host_masses_all[host_indices_for_subs[sub_mask]]) ** 2
        stats = _compute_host_profile_statistics(
            host_ids=host_ids,
            host_local_index=host_local_index,
            sub_host_indices=host_indices_for_subs[sub_mask],
            sub_x_values=dpos_over_R200[sub_mask],
            x_edges=x_edges,
            dx3=dx3,
            statistic=statistic,
            sub_weights=sub_weights,
            percentile_mode=percentile_mode,
        )
        profile_matrix = stats['profile_matrix']
        mean_profile = stats['mean_profile']
        median_profile = stats['median_profile']
        p16_profile = stats['p16_profile']
        p84_profile = stats['p84_profile']

        if normalize_at_vir:
            normalized_rows = []
            for row in profile_matrix:
                normalized_row, _ = _normalize_profile_at_x(x_centers, row, x_target=1.0)
                normalized_rows.append(normalized_row)
            if normalized_rows:
                profile_matrix = np.asarray(normalized_rows, dtype=float)
            mean_profile, mean_norm = _normalize_profile_at_x(x_centers, mean_profile, x_target=1.0)
            median_profile, median_norm = _normalize_profile_at_x(x_centers, median_profile, x_target=1.0)
            p16_profile, p16_norm = _normalize_profile_at_x(x_centers, p16_profile, x_target=1.0)
            p84_profile, p84_norm = _normalize_profile_at_x(x_centers, p84_profile, x_target=1.0)
            print(
                f'mass bin {i} normalization at x=1: '
                f'mean={mean_norm:.3e}, median={median_norm:.3e}, '
                f'p16={p16_norm:.3e}, p84={p84_norm:.3e}'
            )

        nonzero_width = np.where(p84_profile > p16_profile)[0]
        sample_bins = [0, num_x_bins // 2, num_x_bins - 1]
        sample_summary = ', '.join(
            (
                f'bin{k}: mean={mean_profile[k]:.3e}, median={median_profile[k]:.3e}, '
                f'p16={p16_profile[k]:.3e}, p84={p84_profile[k]:.3e}'
            )
            for k in sample_bins
        )
        print(
            f'mass bin {i} percentile summary: '
            f'n_hosts={n_hosts}, n_subs={stats["n_subhalos"]}, '
            f'n_hosts_with_subhalos={stats["n_hosts_with_subhalos"]}, '
            f'percentile_mode={percentile_mode}, '
            f'nonzero_width_bins={nonzero_width.size}/{num_x_bins}; {sample_summary}'
        )

        plot_mean_x, plot_mean = _finite_profile_points(x_centers, mean_profile)
        plot_median_x, plot_median = _finite_profile_points(x_centers, median_profile)
        base_label = (
            rf'${host_mass_bins[i]:.1f}<\log_{{10}}({halo_meta["mass_label"]}/M_\odot h^{{-1}})'
            rf'<{host_mass_bins[i+1]:.1f}$'
        )
        label = _append_count_label(
            base_label,
            n_hosts_total=n_hosts,
            n_hosts_with_subhalos=stats['n_hosts_with_subhalos'],
            n_subhalos=stats['n_subhalos'],
        )
        reference_concentration = np.nan
        reference_nfw = None
        reference_jiang = None
        if show_analytic_reference and statistic == 'count_dx3':
            reference_concentration = _get_subhalo_reference_concentration(
                host_mass_mid,
                data.header.get('Redshift', np.nan),
                radius_definition=radius_definition,
                model=analytic_concentration_model,
            )
            reference_nfw, reference_jiang, reference_han16 = _get_normalized_nfw_subhalo_reference(
                x_centers,
                reference_concentration,
                normalize_at_vir=normalize_at_vir,
                jiang_eta=jiang_eta,
                jiang_mu=jiang_mu,
                han16_gamma=han16_gamma,
            )
            print(
                f'mass bin {i} analytic reference: '
                f'Mmid={host_mass_mid:.3e} Msun/h, c={reference_concentration:.3f}, '
                f'model={analytic_concentration_model}, eta={jiang_eta:.1f}, mu={jiang_mu:.1f}, '
                f'Han16 gamma={han16_gamma:.2f}'
            )

        if show_individual:
            for row in profile_matrix:
                row_x, row_y = _finite_profile_points(x_centers, row)
                ax.plot(
                    row_x,
                    row_y,
                    color=colors[i],
                    linewidth=0.8,
                    alpha=0.08,
                )
        if show_average:
            ax.plot(plot_mean_x, plot_mean, color=colors[i], linewidth=2.0, label=label + ' mean')
        if show_analytic_reference and statistic == 'count_dx3':
            reference_nfw_x, plot_reference_nfw = _finite_profile_points(x_centers, reference_nfw)
            reference_jiang_x, plot_reference_jiang = _finite_profile_points(x_centers, reference_jiang)
            reference_han16_x, plot_reference_han16 = _finite_profile_points(x_centers, reference_han16)
            ax.plot(
                reference_nfw_x,
                plot_reference_nfw,
                color=colors[i],
                linewidth=1.4,
                linestyle='-.',
                alpha=0.95,
                label=base_label + rf' NFW, $c={reference_concentration:.2f}$',
            )
            ax.plot(
                reference_jiang_x,
                plot_reference_jiang,
                color=colors[i],
                linewidth=1.4,
                linestyle=':',
                alpha=0.95,
                label=base_label + ' NFW x JvBIII',
            )
            ax.plot(
                reference_han16_x,
                plot_reference_han16,
                color=colors[i],
                linewidth=1.4,
                linestyle=(0, (3, 1, 1, 1)),
                alpha=0.95,
                label=base_label + rf' NFW x Han16, $\gamma={han16_gamma:.2f}$',
            )
        if show_percentile:
            percentile_valid = np.isfinite(x_centers) & np.isfinite(p16_profile) & np.isfinite(p84_profile)
            plot_percentile_x = x_centers[percentile_valid]
            plot_p16 = p16_profile[percentile_valid]
            plot_p84 = p84_profile[percentile_valid]
            ax.fill_between(
                plot_percentile_x,
                plot_p16,
                plot_p84,
                color=colors[i],
                alpha=0.25,
                linewidth=0,
            )
            ax.plot(
                plot_percentile_x,
                plot_p16,
                color=colors[i],
                linewidth=0.9,
                linestyle=':',
                alpha=0.9,
            )
            ax.plot(
                plot_percentile_x,
                plot_p84,
                color=colors[i],
                linewidth=0.9,
                linestyle=':',
                alpha=0.9,
            )
            ax.plot(
                plot_median_x,
                plot_median,
                color=colors[i],
                linewidth=1.6,
                linestyle='--',
                label=label + ' median',
            )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1.0e-3)
    if paper_style:
        ax.set_xlabel(rf'$\xi = r/{halo_meta["radius_label"]}$', fontsize=14)
    else:
        ax.set_xlabel(rf'$\xi = d_{{\mathrm{{sub-host}}}}/{halo_meta["radius_label"]}$', fontsize=14)
    ylabel = statistic_meta['ylabel']
    if normalize_at_vir:
        ylabel = 'Normalized ' + ylabel
    ax.set_ylabel(ylabel, fontsize=14)
    ax.axvline(1.0, color='black', linestyle=':', linewidth=1.5)
    redshift = data.header.get('Redshift', np.nan)
    title = f'snap {snapNum}, z={redshift:.2f}, {statistic_meta["title_label"]}'
    if psi_range is not None:
        title += rf', ${psi_range[0]:.1e}<\psi<{psi_range[1]:.1e}$'
    if apply_resolution_cut:
        title += ', resolved bins only'
    if show_percentile:
        title += f', pct={percentile_mode}'
    if normalize_at_vir:
        title += ', normalized at Rvir'
    if show_analytic_reference and statistic == 'count_dx3':
        title += f', analytic c(M,z)={analytic_concentration_model}'
    if not paper_style:
        ax.set_title(title, fontsize=13)
    ax.tick_params(direction='in', which='both', labelsize=12)
    ax.legend(fontsize=10 if paper_style else 8, ncol=1)
    plt.tight_layout()

    tag = get_profile_tag(psi_range=psi_range, profile_tag=profile_tag)
    tag_suffix = '' if tag == '' else f'_{tag}'
    normalization_tag = '_normRvir' if normalize_at_vir else ''
    analytic_tag = '_withAnalyticRef' if show_analytic_reference and statistic == 'count_dx3' else ''
    filename = os.path.join(
        output_dir,
        f'{save_prefix}_{halo_meta["tag"]}_{statistic}{normalization_tag}{analytic_tag}{tag_suffix}_snap_{snapNum}.png'
    )
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved radial subhalo profile: {filename}")


def _format_log_host_mass_selection_label(host_mass_min_select, host_mass_max_select, mass_label):
    """Return a compact legend/title label for one direct host-mass selection."""
    if host_mass_min_select is not None and host_mass_max_select is not None:
        return (
            rf'${np.log10(host_mass_min_select):.1f}<\log_{{10}}({mass_label}/M_\odot h^{{-1}})'
            rf'<{np.log10(host_mass_max_select):.1f}$'
        )
    if host_mass_min_select is not None:
        return rf'$\log_{{10}}({mass_label}/M_\odot h^{{-1}})>{np.log10(host_mass_min_select):.1f}$'
    if host_mass_max_select is not None:
        return rf'$\log_{{10}}({mass_label}/M_\odot h^{{-1}})<{np.log10(host_mass_max_select):.1f}$'
    return rf'all {mass_label} hosts'


def _format_host_mass_selection_tag(host_mass_min_select, host_mass_max_select):
    """Return a filename tag for one direct host-mass selection."""
    if host_mass_min_select is not None and host_mass_max_select is not None:
        return f'lgM{np.log10(host_mass_min_select):.1f}_{np.log10(host_mass_max_select):.1f}'.replace('.', 'p')
    if host_mass_min_select is not None:
        return f'lgMgt{np.log10(host_mass_min_select):.1f}'.replace('.', 'p')
    if host_mass_max_select is not None:
        return f'lgMlt{np.log10(host_mass_max_select):.1f}'.replace('.', 'p')
    return 'allhosts'


def plot_host_selected_radial_subhalo_profile(
    data,
    snapNum,
    output_dir,
    host_mass_key='GroupMass',
    psi_mass_key='GroupMass',
    radius_definition='200c',
    host_mass_min_select=None,
    host_mass_max_select=None,
    x_min=1.0e-2,
    x_max=3.0,
    num_x_bins=30,
    log_x_bins=True,
    psi_range=None,
    profile_tag=None,
    save_prefix='radial_subhalo_profile_threshold',
    apply_resolution_cut=None,
    show_average=None,
    show_percentile=None,
    show_median=None,
    show_individual=None,
    statistic=None,
    percentile_mode=None,
    normalize_at_vir=None,
    show_analytic_reference=None,
    analytic_concentration_model=None,
    jiang_eta=None,
    jiang_mu=None,
    han16_gamma=None,
    paper_style=False,
    han16_extra_gammas=(),
    output_suffix='',
):
    """Plot one host-mass-threshold-selected radial profile without mass bins."""
    print(f'Plotting threshold-selected radial subhalo profile for snap {snapNum} ...')
    os.makedirs(output_dir, exist_ok=True)
    required_args = {
        'psi_range': psi_range,
        'profile_tag': profile_tag,
        'apply_resolution_cut': apply_resolution_cut,
        'show_average': show_average,
        'show_percentile': show_percentile,
        'show_median': show_median,
        'show_individual': show_individual,
        'statistic': statistic,
        'percentile_mode': percentile_mode,
        'normalize_at_vir': normalize_at_vir,
        'show_analytic_reference': show_analytic_reference,
        'analytic_concentration_model': analytic_concentration_model,
        'jiang_eta': jiang_eta,
        'jiang_mu': jiang_mu,
        'han16_gamma': han16_gamma,
    }
    missing_args = [name for name, value in required_args.items() if value is None]
    if missing_args:
        raise ValueError(
            'plot_host_selected_radial_subhalo_profile requires explicit values for: ' + ', '.join(missing_args)
        )

    statistic_meta = _get_statistic_metadata(statistic)
    halo_meta = get_tng_halo_definition_metadata(radius_definition)

    host_indices_for_subs = data.subhalo_data['host_index'].value.astype(int)
    host_masses_all = data.halo_data[host_mass_key].value
    psi_host_masses_all = data.halo_data[psi_mass_key].value
    sub_masses = data.subhalo_data['SubMass'].value
    psi_host_masses_for_subs = psi_host_masses_all[host_indices_for_subs]
    dpos_over_R200 = get_subhalo_host_distance_over_R200(data, radius_definition=radius_definition)

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
    host_selection_mask = valid_hosts.copy()
    if host_mass_min_select is not None:
        host_selection_mask &= host_masses_all >= host_mass_min_select
    if host_mass_max_select is not None:
        host_selection_mask &= host_masses_all <= host_mass_max_select

    host_ids = np.where(host_selection_mask)[0]
    n_hosts = len(host_ids)
    if n_hosts == 0:
        raise RuntimeError('No hosts satisfy the requested host-mass selection.')

    selected_host_masses = host_masses_all[host_ids]
    actual_host_mass_min = float(np.min(selected_host_masses))
    actual_host_mass_max = float(np.max(selected_host_masses))
    mean_host_mass = float(np.mean(selected_host_masses))
    crit_psi = 50.0 * get_simulation_resolution(data.header.get('SimulationName', 'TNG50-1'))[1] / actual_host_mass_min
    is_resolved = psi_range is None or psi_range[0] >= crit_psi
    print(
        'host selection: '
        f'Nhost={n_hosts}, logM=[{np.log10(actual_host_mass_min):.2f}, {np.log10(actual_host_mass_max):.2f}], '
        f'mean_logM={np.log10(mean_host_mass):.2f}, crit_psi={crit_psi:.3e}, resolved={is_resolved}'
    )
    if apply_resolution_cut and not is_resolved:
        print('Skipping threshold-selected profile because it does not satisfy the resolution cut.')
        return None

    if log_x_bins:
        x_edges = np.logspace(np.log10(x_min), np.log10(x_max), num_x_bins + 1)
    else:
        x_edges = np.linspace(x_min, x_max, num_x_bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    dx3 = x_edges[1:]**3 - x_edges[:-1]**3

    host_local_index = np.full(host_masses_all.shape[0], -1, dtype=int)
    host_local_index[host_ids] = np.arange(host_ids.size)
    sub_mask = valid_subs & np.isin(host_indices_for_subs, host_ids)
    sub_weights = None
    if statistic == 'mass_dx':
        sub_weights = sub_masses[sub_mask] / host_masses_all[host_indices_for_subs[sub_mask]]
    elif statistic == 'mass2_dx':
        sub_weights = (sub_masses[sub_mask] / host_masses_all[host_indices_for_subs[sub_mask]]) ** 2

    stats = _compute_host_profile_statistics(
        host_ids=host_ids,
        host_local_index=host_local_index,
        sub_host_indices=host_indices_for_subs[sub_mask],
        sub_x_values=dpos_over_R200[sub_mask],
        x_edges=x_edges,
        dx3=dx3,
        statistic=statistic,
        sub_weights=sub_weights,
        percentile_mode=percentile_mode,
    )

    profile_matrix = stats['profile_matrix']
    mean_profile = stats['mean_profile']
    median_profile = stats['median_profile']
    p16_profile = stats['p16_profile']
    p84_profile = stats['p84_profile']

    if normalize_at_vir:
        normalized_rows = []
        for row in profile_matrix:
            normalized_row, _ = _normalize_profile_at_x(x_centers, row, x_target=1.0)
            normalized_rows.append(normalized_row)
        if normalized_rows:
            profile_matrix = np.asarray(normalized_rows, dtype=float)
        mean_profile, mean_norm = _normalize_profile_at_x(x_centers, mean_profile, x_target=1.0)
        median_profile, median_norm = _normalize_profile_at_x(x_centers, median_profile, x_target=1.0)
        p16_profile, p16_norm = _normalize_profile_at_x(x_centers, p16_profile, x_target=1.0)
        p84_profile, p84_norm = _normalize_profile_at_x(x_centers, p84_profile, x_target=1.0)
        print(
            f'threshold selection normalization at x=1: mean={mean_norm:.3e}, median={median_norm:.3e}, '
            f'p16={p16_norm:.3e}, p84={p84_norm:.3e}'
        )

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    selection_label = _format_log_host_mass_selection_label(
        host_mass_min_select,
        host_mass_max_select,
        halo_meta['mass_label'],
    )
    label = _append_count_label(
        selection_label,
        n_hosts_total=n_hosts,
        n_hosts_with_subhalos=stats['n_hosts_with_subhalos'],
        n_subhalos=stats['n_subhalos'],
    )
    if paper_style:
        mass_min_for_label = host_mass_min_select if host_mass_min_select is not None else actual_host_mass_min
        mass_max_for_label = host_mass_max_select if host_mass_max_select is not None else actual_host_mass_max
        psi_label = ''
        if psi_range is not None:
            psi_left = f'{psi_range[0]:.0e}'.replace('e-0', 'e-').replace('e+0', 'e')
            if np.isclose(psi_range[1], 1.0):
                psi_right = '1'
            else:
                psi_right = f'{psi_range[1]:.0e}'.replace('e-0', 'e-').replace('e+0', 'e')
            psi_label = rf', $\psi\in[{psi_left},{psi_right}]$'
        mean_label = (
            rf'TNG50 mean: $N_{{\rm host}}={n_hosts}$, '
            rf'$10^{{{np.log10(mass_min_for_label):.1f}}}<M_{{\rm host}}/(M_\odot/h)'
            rf'<10^{{{np.log10(mass_max_for_label):.1f}}}$'
            + psi_label
        )
        median_label = 'TNG50 median'
    else:
        mean_label = label + ' mean'
        median_label = label + ' median'

    if show_individual:
        for row in profile_matrix:
            row_x, row_y = _finite_profile_points(x_centers, row)
            ax.plot(
                row_x,
                row_y,
                color='tab:blue',
                linewidth=0.8,
                alpha=0.08,
            )
    if show_average:
        mean_x, plot_mean = _finite_profile_points(x_centers, mean_profile)
        ax.plot(
            mean_x,
            plot_mean,
            color='black' if paper_style else 'tab:blue',
            linewidth=2.2,
            label=mean_label,
        )
    if show_analytic_reference and statistic == 'count_dx3':
        reference_concentration = _get_subhalo_reference_concentration(
            mean_host_mass,
            data.header.get('Redshift', np.nan),
            radius_definition=radius_definition,
            model=analytic_concentration_model,
        )
        reference_nfw, reference_jiang, reference_han16 = _get_normalized_nfw_subhalo_reference(
            x_centers,
            reference_concentration,
            normalize_at_vir=normalize_at_vir,
            jiang_eta=jiang_eta,
            jiang_mu=jiang_mu,
            han16_gamma=han16_gamma,
        )
        print(
            f'threshold analytic reference: Mmean={mean_host_mass:.3e} Msun/h, c={reference_concentration:.3f}, '
            f'model={analytic_concentration_model}, eta={jiang_eta:.1f}, mu={jiang_mu:.1f}, '
            f'Han16 gamma={han16_gamma:.2f}'
        )
        reference_nfw_x, plot_reference_nfw = _finite_profile_points(x_centers, reference_nfw)
        reference_jiang_x, plot_reference_jiang = _finite_profile_points(x_centers, reference_jiang)
        reference_han16_x, plot_reference_han16 = _finite_profile_points(x_centers, reference_han16)
        ax.plot(
            reference_nfw_x,
            plot_reference_nfw,
            color='tab:red',
            linewidth=1.8,
            linestyle='-.',
            label=rf'NFW, $c={reference_concentration:.2f}$',
        )
        ax.plot(
            reference_jiang_x,
            plot_reference_jiang,
            color='tab:green',
            linewidth=1.8,
            linestyle=':',
            label='NFW x JvBIII',
        )
        ax.plot(
            reference_han16_x,
            plot_reference_han16,
            color='tab:purple',
            linewidth=1.8,
            linestyle=(0, (3, 1, 1, 1)),
            label=rf'NFW x Han16, $\gamma={han16_gamma:.2f}$',
        )
        for extra_gamma in han16_extra_gammas:
            if np.isclose(extra_gamma, han16_gamma):
                continue
            _, _, reference_han16_extra = _get_normalized_nfw_subhalo_reference(
                x_centers,
                reference_concentration,
                normalize_at_vir=normalize_at_vir,
                jiang_eta=jiang_eta,
                jiang_mu=jiang_mu,
                han16_gamma=extra_gamma,
            )
            reference_han16_extra_x, plot_reference_han16_extra = _finite_profile_points(
                x_centers,
                reference_han16_extra,
            )
            ax.plot(
                reference_han16_extra_x,
                plot_reference_han16_extra,
                color='tab:purple',
                linewidth=1.8,
                linestyle='--',
                label=rf'NFW x Han16, $\gamma={extra_gamma:.2f}$',
            )
    median_x, plot_median = _finite_profile_points(x_centers, median_profile)
    if show_percentile:
        percentile_valid = np.isfinite(x_centers) & np.isfinite(p16_profile) & np.isfinite(p84_profile)
        plot_percentile_x = x_centers[percentile_valid]
        plot_p16 = p16_profile[percentile_valid]
        plot_p84 = p84_profile[percentile_valid]
        ax.fill_between(plot_percentile_x, plot_p16, plot_p84, color='tab:blue', alpha=0.22, linewidth=0)
        ax.plot(plot_percentile_x, plot_p16, color='tab:blue', linewidth=0.9, linestyle=':', alpha=0.9)
        ax.plot(plot_percentile_x, plot_p84, color='tab:blue', linewidth=0.9, linestyle=':', alpha=0.9)
    if show_median:
        if paper_style:
            ax.scatter(
                median_x,
                plot_median,
                facecolors='none',
                edgecolors='black',
                linewidths=1.5,
                s=34,
                label=median_label,
                zorder=4,
            )
        else:
            ax.plot(median_x, plot_median, color='tab:blue', linewidth=1.6, linestyle='--', label=median_label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1.0e-3)
    if paper_style:
        ax.set_xlabel(rf'$\xi = r/{halo_meta["radius_label"]}$', fontsize=14)
    else:
        ax.set_xlabel(rf'$\xi = d_{{\mathrm{{sub-host}}}}/{halo_meta["radius_label"]}$', fontsize=14)
    ylabel = statistic_meta['ylabel']
    if normalize_at_vir:
        ylabel = 'Normalized ' + ylabel
    ax.set_ylabel(ylabel, fontsize=14)
    ax.axvline(1.0, color='black', linestyle=':', linewidth=1.5)
    redshift = data.header.get('Redshift', np.nan)
    title = f'snap {snapNum}, z={redshift:.2f}, {statistic_meta["title_label"]}'
    title += ', host threshold selection'
    if psi_range is not None:
        title += rf', ${psi_range[0]:.1e}<\psi<{psi_range[1]:.1e}$'
    if apply_resolution_cut:
        title += ', resolved selection only'
    if normalize_at_vir:
        title += ', normalized at Rvir'
    if not paper_style:
        ax.set_title(title, fontsize=13)
    ax.tick_params(direction='in', which='both', labelsize=12)
    ax.legend(fontsize=10, ncol=1)
    plt.tight_layout()

    selection_tag = _format_host_mass_selection_tag(host_mass_min_select, host_mass_max_select)
    tag = get_profile_tag(psi_range=psi_range, profile_tag=profile_tag)
    tag_suffix = '' if tag == '' else f'_{tag}'
    normalization_tag = '_normRvir' if normalize_at_vir else ''
    analytic_tag = '_withAnalyticRef' if show_analytic_reference and statistic == 'count_dx3' else ''
    suffix_tag = output_suffix or ''
    filename = os.path.join(
        output_dir,
        f'{save_prefix}_{halo_meta["tag"]}_{selection_tag}_{statistic}{normalization_tag}{analytic_tag}{tag_suffix}{suffix_tag}_snap_{snapNum}.png'
    )
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved threshold-selected radial subhalo profile: {filename}')
    return filename


def run_subhalo_number_profile_mass_thresholds(
    snapNum_list=None,
    simulation_set='TNG50-1',
    base_dir='/home/zwu/21cm_project/unified_model/TNG_results/',
    psi_range=(0.001, 1.0),
    profile_tag='psi0p001_1',
    radius_definition=('200c', '200m'),
    host_mass_min_selects=(1.0e13, 1.0e14),
    profile_kwargs=None,
):
    """Driver for direct host-mass-threshold TNG radial subhalo profiles."""
    if snapNum_list is None:
        snapNum_list = [99]
    driver_defaults = {
        'psi_range': psi_range,
        'profile_tag': profile_tag,
        'apply_resolution_cut': True,
        'show_average': True,
        'show_percentile': False,
        'show_median': True,
        'show_individual': False,
        'statistic': 'count_dx3',
        'percentile_mode': 'all_hosts',
        'normalize_at_vir': True,
        'show_analytic_reference': True,
        'analytic_concentration_model': 'diemer19',
        'jiang_eta': 2.0,
        'jiang_mu': 4.0,
        'han16_gamma': 1.33,
        'han16_extra_gammas': (0.95,),
    }
    if profile_kwargs is not None:
        driver_defaults.update(profile_kwargs)

    if isinstance(radius_definition, str):
        radius_definitions = [radius_definition]
    else:
        radius_definitions = list(radius_definition)

    for snapNum in snapNum_list:
        print(f'Processing threshold-selected snapshot {snapNum} ...')
        processed_file = os.path.join(
            base_dir,
            simulation_set,
            f'snap_{snapNum}',
            f'processed_halos_snap_{snapNum}.h5'
        )
        output_dir = os.path.join(base_dir, simulation_set, f'snap_{snapNum}', 'analysis')
        data = load_processed_data(processed_file)
        for radius_definition_value in radius_definitions:
            for host_mass_min_select in host_mass_min_selects:
                plot_host_selected_radial_subhalo_profile(
                    data,
                    snapNum,
                    output_dir,
                    radius_definition=radius_definition_value,
                    host_mass_min_select=host_mass_min_select,
                    **driver_defaults,
                )


def run_subhalo_number_profile(
    snapNum_list=None,
    simulation_set='TNG50-1',
    base_dir='/home/zwu/21cm_project/unified_model/TNG_results/',
    psi_range=(0.001, 1.0),
    profile_tag='psi0p001_1',
    radius_definition=('200c', '200m'),
    profile_kwargs=None,
):
    """
    Main driver for host-averaged subhalo number radial profiles.

    This intentionally handles only the dN/dx^3 subhalo number profile for now.
    The default plotting mode emphasizes the host-averaged mean plus percentile
    shading, while optionally discarding host-mass bins that fail the subhalo
    resolution cut for the selected psi range.
    """
    if snapNum_list is None:
        snapNum_list = [99, 50, 13]
    driver_defaults = {
        'psi_range': psi_range,
        'profile_tag': profile_tag,
        'apply_resolution_cut': True,
        'show_average': True,
        'show_percentile': False,
        'show_median': True,
        'show_individual': False,
        'statistic': 'count_dx3',   #count_dx3, count_dx, mass_dx, or mass2_dx
        'percentile_mode': 'all_hosts',  #'all_hosts' or 'occupied_hosts'
        'normalize_at_vir': True,
        'show_analytic_reference': True,
        'analytic_concentration_model': 'diemer19',
        'jiang_eta': 2.0,
        'jiang_mu': 4.0,
        'han16_gamma': 1.33,
    }
    if profile_kwargs is not None:
        driver_defaults.update(profile_kwargs)

    if isinstance(radius_definition, str):
        radius_definitions = [radius_definition]
    else:
        radius_definitions = list(radius_definition)

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
        for radius_definition_value in radius_definitions:
            plot_host_averaged_radial_subhalo_profile(
                data,
                snapNum,
                output_dir,
                radius_definition=radius_definition_value,
                **driver_defaults
            )



def plot_tng_z0_radial_profile_for_paper(
    processed_file='/home/zwu/21cm_project/unified_model/TNG_results/TNG50-1/snap_99/processed_halos_snap_99.h5',
    output_dir='/home/zwu/21cm_project/unified_model/Profile_results_for_paper',
):
    """Generate the final z=0 TNG radial profile figure used by the paper."""
    data = load_processed_data(processed_file)
    return plot_host_selected_radial_subhalo_profile(
        data,
        snapNum=99,
        output_dir=output_dir,
        radius_definition='200c',
        host_mass_min_select=1.0e13,
        psi_range=(1.0e-3, 1.0),
        profile_tag='psi0p001_1',
        apply_resolution_cut=True,
        show_average=True,
        show_percentile=False,
        show_median=True,
        show_individual=False,
        statistic='count_dx3',
        percentile_mode='all_hosts',
        normalize_at_vir=True,
        show_analytic_reference=True,
        analytic_concentration_model='diemer19',
        jiang_eta=2.0,
        jiang_mu=4.0,
        han16_gamma=1.33,
        han16_extra_gammas=(0.95,),
        paper_style=True,
        output_suffix='_paper',
    )


def run_profile_figures_for_paper():
    """Generate the current final radial-profile figures in Profile_results_for_paper."""
    output_paths = []
    output_paths.append(plot_pop2prime_tng_radial_profile_comparison_selected_z12_paper())
    output_paths.append(plot_tng_z0_radial_profile_for_paper())
    print('Generated paper profile figures:')
    for output_path in output_paths:
        print(f'  {output_path}')
    return output_paths


if __name__ == '__main__':
    run_profile_figures_for_paper()
