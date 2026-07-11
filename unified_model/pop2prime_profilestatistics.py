import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from Analytic_halo_profile import (
    density_NFW_profile,
    gasdensity_arbitrary_profile,
    get_concentration,
)
from HaloProperties import Temperature_Virial_analytic
from physical_constants import Omega_b, Omega_k, Omega_lambda, Omega_m, Omega_r, h_Hubble, rho_crit_z0
from read_pop2prime import (
    POP2PRIME_BASE_DIR,
    POP2PRIME_ROCKSTAR_DIR,
    create_parent_dict,
    find_nearest_snapshots,
    load_snapshot_redshifts,
)


POP2PRIME_RESULTS_DIR = Path("/home/zwu/21cm_project/unified_model/Pop2prime_results")
TARGET_REDSHIFT = 12.0
HOST_MASS_MIN = 10**5.5  # Msun/h
CONCENTRATION_MODEL = "ludlow16"
CONCENTRATION_LABEL = "Ludlow16"
PSI_MIN = 0.05   #Note: this is not the psi_min for plot_pop2prime_radial_subhalo_weighted_profile()
PSI_THRESHOLDS_FOR_EXPORT = (1.0e-3, 1.0e-2, 5.0e-2)
GAS_PROFILE_HOST_MASS_MIN = HOST_MASS_MIN
PAPER_GAS_FRACTION_LOW = 0.03
PAPER_GAS_PROFILE_ALPHAS = (-0.5, 1.5)
PAPER_DENSITY_YMIN = 1.0e-3
JB17_ETA = 2.0
JB17_MU = 4.0
HAN16_GAMMA = 1.33


def build_x_bins(x_min=2.0e-2, x_max=2.0, num_x_bins=20, log_x_bins=True):
    """Return radial bin edges, centers, and x^3 bin widths."""
    if log_x_bins:
        x_edges = np.logspace(np.log10(x_min), np.log10(x_max), num_x_bins + 1)
    else:
        x_edges = np.linspace(x_min, x_max, num_x_bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    dx3 = x_edges[1:] ** 3 - x_edges[:-1] ** 3
    return x_edges, x_centers, dx3


def interpolate_profile_at_x(x_centers, profile, x_target=1.0):
    """Estimate a positive profile value at x_target from neighboring bins."""
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


def normalize_profile_at_x(x_centers, profile, x_target=1.0):
    """Return profile/profile(x_target), preserving invalid normalizations as NaN."""
    normalization = interpolate_profile_at_x(x_centers, profile, x_target=x_target)
    if not np.isfinite(normalization) or normalization <= 0:
        return np.full_like(profile, np.nan, dtype=float), normalization
    return np.asarray(profile, dtype=float) / normalization, normalization


def jb17_radial_bias(x_values, eta=JB17_ETA, mu=JB17_MU):
    """Return the JB17 radial bias factor with the corrected parameter order."""
    x_values = np.asarray(x_values, dtype=float)
    return (2.0 ** mu) * x_values ** eta / (1.0 + x_values) ** mu


def han16_radial_bias(x_values, gamma=HAN16_GAMMA):
    """Return the Han16 radial bias factor phi(x) = x^gamma."""
    x_values = np.asarray(x_values, dtype=float)
    return x_values ** gamma


def nfw_count_dx3_shape(x_values, concentration):
    """Return the NFW dN/dx^3 shape for x=r/Rvir up to an overall constant."""
    x_values = np.asarray(x_values, dtype=float)
    cx = concentration * x_values
    return 1.0 / (cx * (1.0 + cx) ** 2)


def build_jb17_reference_profiles(x_centers, host_mass_msunh, redshift, han16_gamma=HAN16_GAMMA):
    """Return normalized NFW, JB17-modified, and Han16-modified profiles."""
    concentration = get_concentration(host_mass_msunh / h_Hubble, redshift, CONCENTRATION_MODEL)
    nfw_profile = nfw_count_dx3_shape(x_centers, concentration)
    jb17_profile = nfw_profile * jb17_radial_bias(x_centers)
    han16_profile = nfw_profile * han16_radial_bias(x_centers, gamma=han16_gamma)
    nfw_profile, _ = normalize_profile_at_x(x_centers, nfw_profile, x_target=1.0)
    jb17_profile, _ = normalize_profile_at_x(x_centers, jb17_profile, x_target=1.0)
    han16_profile, _ = normalize_profile_at_x(x_centers, han16_profile, x_target=1.0)
    return nfw_profile, jb17_profile, han16_profile, concentration


def format_host_mass_min_tag(host_mass_min):
    """Return a compact filename tag for the host-mass threshold."""
    return f"lgMmin{np.log10(host_mass_min):.1f}"


def infer_host_mass_min_for_label(input_path, metadata, host_profiles):
    """
    Return the host-mass threshold to display in plot labels.

    Preference order:
    1. Explicit metadata saved in the table header.
    2. Filename tag like ``lgMmin5.5``.
    3. The actual minimum host mass in the loaded sample.
    """
    if "host_mass_min_Msunh" in metadata:
        return float(metadata["host_mass_min_Msunh"])

    match = re.search(r"lgMmin([0-9]+(?:\.[0-9]+)?)", Path(input_path).name)
    if match is not None:
        return 10 ** float(match.group(1))

    return min(profile["host_mass"] for profile in host_profiles)


def minimum_image_displacement(pos, center, box_size):
    """Return displacement vectors with periodic minimum-image wrapping."""
    delta = pos - center
    if box_size is not None and np.all(box_size > 0):
        delta -= box_size * np.round(delta / box_size)
    return delta


def _append_count_label(
    label,
    n_hosts_total=None,
    n_hosts_with_subhalos=None,
    n_subhalos=None,
):
    """Append host/subhalo counts to a legend label when available."""
    suffix_parts = []
    if n_hosts_total is not None:
        suffix_parts.append(f"Nhost,total={n_hosts_total}")
    if n_hosts_with_subhalos is not None:
        suffix_parts.append(f"Nhost,sub={n_hosts_with_subhalos}")
    if n_subhalos is not None:
        suffix_parts.append(f"Nsub={n_subhalos}")
    if not suffix_parts:
        return label
    return f"{label}, " + ", ".join(suffix_parts)


def load_pop2prime_halo_catalog(snapshot=None, target_redshift=TARGET_REDSHIFT, print_keys = False):
    """
    Load one Pop2Prime Rockstar halo catalog and return a compact data dict.

    Parameters
    ----------
    snapshot : int, optional
        Explicit snapshot number.  If omitted, the nearest snapshot to
        ``target_redshift`` is used.
    target_redshift : float, optional
        Target redshift used when ``snapshot`` is not supplied.
    """
    if snapshot is None:
        snapshot = find_nearest_snapshots([target_redshift])[0]

    halo_path = POP2PRIME_ROCKSTAR_DIR / f"halos_DD{snapshot:04d}.0.bin"
    ds = yt.load(str(halo_path))
    ad = ds.all_data()

    if print_keys:
        print("Available halo fields:")
        for field in sorted(ds.field_list):
            if field[0] == "halos":
                print(f"  {field[1]}")
        

    positions = np.column_stack(
        [
            ad["halos", "particle_position_x"].in_units("Mpc").to_ndarray(),
            ad["halos", "particle_position_y"].in_units("Mpc").to_ndarray(),
            ad["halos", "particle_position_z"].in_units("Mpc").to_ndarray(),
        ]
    )
    velocities = np.column_stack(
        [
            ad["halos", "particle_velocity_x"].in_units("km/s").to_ndarray(),
            ad["halos", "particle_velocity_y"].in_units("km/s").to_ndarray(),
            ad["halos", "particle_velocity_z"].in_units("km/s").to_ndarray(),
        ]
    )
    catalog = {
        "snapshot": snapshot,
        "redshift": float(ds.current_redshift),
        "scale_factor": float(ds.scale_factor),
        "box_size_mpc": ds.domain_width.in_units("Mpc").to_ndarray(),
        "dataset": ds,
        "all_data": ad,
        "halo_ids": ad["halos", "particle_identifier"].to_ndarray().astype(int),
        "masses": ad["halos", "particle_mass"].to("Msun/h").to_ndarray(),
        "rvir": ad["halos", "virial_radius"].in_units("Mpc").to_ndarray(),
        "positions": positions,
        "velocities": velocities,
        "field_list": sorted([field for field in ds.field_list if field[0] == "halos"]),
    }
    return catalog


def load_pop2prime_snapshot(snapshot):
    """Load the Pop2Prime hydro snapshot matching one halo catalog snapshot."""
    snapshot_tag = f"DD{snapshot:04d}"
    snapshot_path = POP2PRIME_BASE_DIR / snapshot_tag / snapshot_tag
    return yt.load(str(snapshot_path))


def find_snapshot_metadata(snapshot):
    """Return the saved scale factor and redshift metadata for one snapshot."""
    for entry in load_snapshot_redshifts():
        if entry["snapshot"] == snapshot:
            return entry
    return None


def get_geometric_parent_ids(catalog):
    """Return parent halo IDs using the current geometric overlap rule."""
    parent_dict = create_parent_dict(catalog["all_data"])
    return np.array([parent_dict[halo_id] for halo_id in catalog["halo_ids"]], dtype=int)


def get_rockstar_native_parent_ids(catalog):
    """
    Return parent halo IDs from native Rockstar fields if they exist.

    This checks several common parent-label names.  The current Pop2Prime
    halo catalogs inspected so far do not expose ``pid`` or ``upid`` through
    yt, so this may raise until those fields are available in the source files.
    """
    candidate_fields = [
        ("halos", "upid"),
        ("halos", "pid"),
        ("halos", "parent_id"),
        ("halos", "host_id"),
    ]
    available_fields = set(catalog["field_list"])

    for field in candidate_fields:
        if field in available_fields:
            parent_ids = catalog["all_data"][field].to_ndarray().astype(int)
            return parent_ids

    raise RuntimeError(
        "No native Rockstar parent-ID field was found in this halo catalog. "
        "Checked fields: upid, pid, parent_id, host_id."
    )


def compute_subhalo_count_profile_sample(
    catalog,
    parent_ids,
    host_mass_min=HOST_MASS_MIN,
    psi_min=PSI_MIN,
):
    """
    Build a host-averaged subhalo radial sample for one parent-label definition.

    Returns
    -------
    dict
        Contains per-subhalo radii, host properties, and per-host summary arrays.
        Only subhalos with psi > psi_min are retained.
    """
    halo_ids = catalog["halo_ids"]
    masses = catalog["masses"]
    rvir = catalog["rvir"]
    positions = catalog["positions"]
    box_size = catalog["box_size_mpc"]

    id_to_index = {halo_id: i for i, halo_id in enumerate(halo_ids)}
    host_indices = np.where((parent_ids == -1) & (masses >= host_mass_min))[0]

    rows = []
    host_subhalo_counts = np.zeros(host_indices.size, dtype=int)

    for host_counter, host_index in enumerate(host_indices):
        host_id = halo_ids[host_index]
        subhalo_indices = np.where(parent_ids == host_id)[0]
        if subhalo_indices.size > 0:
            psi_all = masses[subhalo_indices] / masses[host_index]
            subhalo_indices = subhalo_indices[psi_all > psi_min]
        host_subhalo_counts[host_counter] = subhalo_indices.size

        for sub_index in subhalo_indices:
            dr_vec = minimum_image_displacement(
                positions[sub_index],
                positions[host_index],
                box_size,
            )
            dr = np.sqrt(np.sum(dr_vec**2))
            dr_over_rvir = dr / rvir[host_index] if rvir[host_index] > 0 else np.nan
            rows.append(
                (
                    host_index,
                    host_id,
                    masses[host_index],
                    rvir[host_index],
                    sub_index,
                    halo_ids[sub_index],
                    masses[sub_index],
                    dr,
                    dr_over_rvir,
                    masses[sub_index] / masses[host_index],
                )
            )

    dtype = [
        ("host_index", "i8"),
        ("host_id", "i8"),
        ("host_mass", "f8"),
        ("host_rvir_mpc", "f8"),
        ("subhalo_index", "i8"),
        ("subhalo_id", "i8"),
        ("subhalo_mass", "f8"),
        ("distance_mpc", "f8"),
        ("distance_over_rvir", "f8"),
        ("psi", "f8"),
    ]
    sample = np.array(rows, dtype=dtype) if rows else np.array([], dtype=dtype)

    return {
        "sample": sample,
        "host_indices": host_indices,
        "host_ids": halo_ids[host_indices],
        "host_masses": masses[host_indices],
        "host_rvir": rvir[host_indices],
        "host_subhalo_counts": host_subhalo_counts,
        "n_hosts": host_indices.size,
        "n_subhalos": sample.size,
        "id_to_index": id_to_index,
        "psi_min": psi_min,
    }


def compute_host_averaged_subhalo_count_profile(
    subhalo_count_data,
    x_min=2.0e-2,
    x_max=2.0,
    num_x_bins=20,
    log_x_bins=True,
):
    """
    Compute the host-averaged radial profile dN/dx^3 with x = r_sub-host / Rvir.
    """
    x_edges, x_centers, dx3 = build_x_bins(
        x_min=x_min,
        x_max=x_max,
        num_x_bins=num_x_bins,
        log_x_bins=log_x_bins,
    )

    host_ids = subhalo_count_data["host_ids"]
    sample = subhalo_count_data["sample"]
    n_hosts = subhalo_count_data["n_hosts"]
    profile_matrix = np.zeros((n_hosts, num_x_bins))

    if n_hosts > 0 and sample.size > 0:
        for i, host_id in enumerate(host_ids):
            host_mask = sample["host_id"] == host_id
            counts, _ = np.histogram(sample["distance_over_rvir"][host_mask], bins=x_edges)
            profile_matrix[i, :] = counts / dx3

    return {
        "x_edges": x_edges,
        "x_centers": x_centers,
        "profile_matrix": profile_matrix,
        "host_ids": host_ids,
        "host_masses": subhalo_count_data["host_masses"],
        "mean_profile": np.mean(profile_matrix, axis=0) if n_hosts > 0 else np.zeros(num_x_bins),
        "median_profile": np.median(profile_matrix, axis=0) if n_hosts > 0 else np.zeros(num_x_bins),
        "p16_profile": np.percentile(profile_matrix, 16, axis=0) if n_hosts > 0 else np.zeros(num_x_bins),
        "p84_profile": np.percentile(profile_matrix, 84, axis=0) if n_hosts > 0 else np.zeros(num_x_bins),
        "n_hosts": n_hosts,
        "n_subhalos": subhalo_count_data["n_subhalos"],
    }


def compute_host_averaged_subhalo_weighted_profile(
    subhalo_count_data,
    x_min=2.0e-2,
    x_max=2.0,
    num_x_bins=20,
    log_x_bins=True,
    statistic="count_dx",
):
    """
    Compute host-averaged radial subhalo profiles with d/dx weighting.

    statistic:
      - "count_dx": dN_sub / dx
      - "mass_dx": d/dx sum(m_sub / M_host)
      - "mass2_dx": d/dx sum((m_sub / M_host)^2)
    """
    x_edges, x_centers, _ = build_x_bins(
        x_min=x_min,
        x_max=x_max,
        num_x_bins=num_x_bins,
        log_x_bins=log_x_bins,
    )
    dx = x_edges[1:] - x_edges[:-1]

    host_ids = subhalo_count_data["host_ids"]
    host_masses = subhalo_count_data["host_masses"]
    sample = subhalo_count_data["sample"]
    n_hosts = subhalo_count_data["n_hosts"]
    profile_matrix = np.zeros((n_hosts, num_x_bins))

    if n_hosts > 0 and sample.size > 0:
        for i, (host_id, host_mass) in enumerate(zip(host_ids, host_masses)):
            host_mask = sample["host_id"] == host_id
            x_values = sample["distance_over_rvir"][host_mask]

            if statistic == "count_dx":
                weights = None
            elif statistic == "mass_dx":
                weights = sample["subhalo_mass"][host_mask] / host_mass
            elif statistic == "mass2_dx":
                weights = (sample["subhalo_mass"][host_mask] / host_mass) ** 2
            else:
                raise ValueError(f"Unknown statistic: {statistic}")

            histogram, _ = np.histogram(x_values, bins=x_edges, weights=weights)
            profile_matrix[i, :] = histogram / dx

    return {
        "x_edges": x_edges,
        "x_centers": x_centers,
        "profile_matrix": profile_matrix,
        "host_ids": host_ids,
        "host_masses": host_masses,
        "mean_profile": np.mean(profile_matrix, axis=0) if n_hosts > 0 else np.zeros(num_x_bins),
        "median_profile": np.median(profile_matrix, axis=0) if n_hosts > 0 else np.zeros(num_x_bins),
        "p16_profile": np.percentile(profile_matrix, 16, axis=0) if n_hosts > 0 else np.zeros(num_x_bins),
        "p84_profile": np.percentile(profile_matrix, 84, axis=0) if n_hosts > 0 else np.zeros(num_x_bins),
        "n_hosts": n_hosts,
        "n_subhalos": subhalo_count_data["n_subhalos"],
        "statistic": statistic,
    }


def select_hosts_with_subhalos(
    catalog,
    parent_ids,
    host_mass_min=HOST_MASS_MIN,
):
    """
    Return host-halo indices that retain at least one subhalo.

    This is a lightweight selector for workflows that need the host list but do
    not need the full subhalo-count radial-profile sample.  The selection
    follows the read_pop2prime.py convention: a host is kept if it has at
    least one subhalo, without applying a psi threshold.
    """
    halo_ids = catalog["halo_ids"]
    masses = catalog["masses"]
    all_host_indices = np.where(parent_ids == -1)[0]
    host_indices = np.where((parent_ids == -1) & (masses >= host_mass_min))[0]

    selected_host_indices = []
    selected_host_subhalo_counts = []
    for host_index in host_indices:
        host_id = halo_ids[host_index]
        subhalo_indices = np.where(parent_ids == host_id)[0]
        if subhalo_indices.size > 0:
            selected_host_indices.append(host_index)
            selected_host_subhalo_counts.append(subhalo_indices.size)

    print(f"Total host halos: {all_host_indices.size}")
    print(f"Host halos with Mhost >= {host_mass_min:.2e} Msun/h: {host_indices.size}")
    print(f"Host halos with at least one subhalo: {len(selected_host_indices)}")

    return {
        "all_host_indices": all_host_indices,
        "host_indices": np.array(selected_host_indices, dtype=int),
        "host_subhalo_counts": np.array(selected_host_subhalo_counts, dtype=int),
        "n_hosts_total": all_host_indices.size,
        "n_hosts_above_mass_min": host_indices.size,
        "n_hosts_with_subhalos": len(selected_host_indices),
        "host_mass_min": host_mass_min,
    }




def get_host_count_with_subhalos(radial_data):
    """Return the number of hosts that retain at least one selected subhalo."""
    return int(np.sum(radial_data["host_subhalo_counts"] > 0))


def summarize_subhalo_count_profile_sample(radial_data):
    """Print a compact summary for one radial sample."""
    sample = radial_data["sample"]
    print(f"Hosts above threshold: {radial_data['n_hosts']}")
    print(f"Hosts with at least one selected subhalo: {get_host_count_with_subhalos(radial_data)}")
    print(f"Subhalos linked to those hosts with psi > {radial_data['psi_min']:.2f}: {radial_data['n_subhalos']}")
    if sample.size == 0:
        print("No subhalos in sample.")
        return
    ratio = sample["distance_over_rvir"]
    print(f"Median r/Rvir: {np.median(ratio):.3f}")
    print(f"Max r/Rvir: {np.max(ratio):.3f}")
    print(f"Fraction with r/Rvir > 1: {np.mean(ratio > 1.0):.4f}")


def save_subhalo_count_profile_sample(radial_data, output_path):
    """Save the per-subhalo radial sample to a text table."""
    sample = radial_data["sample"]
    header = (
        "host_index host_id host_mass_Msunh host_rvir_Mpc "
        "subhalo_index subhalo_id subhalo_mass_Msunh "
        f"distance_Mpc distance_over_rvir psi psi_min_gt={radial_data['psi_min']:.2f}"
    )
    np.savetxt(
        output_path,
        sample,
        header=header,
        fmt=[
            "%d",
            "%d",
            "%.8e",
            "%.8e",
            "%d",
            "%d",
            "%.8e",
            "%.8e",
            "%.8e",
            "%.8e",
        ],
    )
    print(f"Saved radial sample to {output_path}")


def plot_subhalo_count_profiles_comparison(
    profile_by_method,
    snapshot,
    redshift,
    output_path,
    host_mass_min=HOST_MASS_MIN,
    psi_min=PSI_MIN,
):
    """Plot host-averaged radial profiles for each available subhalo definition."""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    artificial_small = 1.0e-12
    colors = {
        "geometric": "tab:blue",
        "rockstar_native": "tab:orange",
    }

    for method_name, profile in profile_by_method.items():
        x_centers = profile["x_centers"]
        mean_profile = np.where(profile["mean_profile"] > 0, profile["mean_profile"], artificial_small)
        p16_profile = np.where(profile["p16_profile"] > 0, profile["p16_profile"], artificial_small)
        p84_profile = np.where(profile["p84_profile"] > 0, profile["p84_profile"], artificial_small)
        label = (
            f"{method_name} "
            f"(N_host={profile['n_hosts']}, N_sub={profile['n_subhalos']})"
        )
        ax.plot(
            x_centers,
            mean_profile,
            color=colors.get(method_name, None),
            linewidth=2.0,
            label=label,
        )
        ax.fill_between(
            x_centers,
            p16_profile,
            p84_profile,
            color=colors.get(method_name, None),
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(1.0, color="black", linestyle=":", linewidth=1.4)
    ax.set_xlabel(r"$x=d_{\mathrm{sub-host}}/R_{\mathrm{vir}}$", fontsize=14)
    ax.set_ylabel(r"$\mathrm{d}N_{\mathrm{sub}}/\mathrm{d}x^3$", fontsize=14)
    ax.set_title(
        rf"Pop2Prime radial subhalo profile, z={redshift:.2f}, "
        rf"$M_{{\mathrm{{host}}}} \geq 10^{{{np.log10(host_mass_min):.1f}}}\,M_\odot/h$, "
        rf"$\psi > {psi_min:.2f}$",
        fontsize=12,
    )
    ax.tick_params(direction="in", which="both", labelsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved radial profile plot to {output_path}")



def export_subhalo_count_profiles_txt(
    snapshot=None,
    target_redshift=TARGET_REDSHIFT,
    host_mass_min=HOST_MASS_MIN,
    psi_thresholds=PSI_THRESHOLDS_FOR_EXPORT,
    output_dir=POP2PRIME_RESULTS_DIR,
    subhalo_definition="geometric_allpsi",
):
    """
    Export Pop2Prime radial subhalo profiles to one text file per redshift.

    The file contains one profile block for each psi threshold.  Each block
    stores host/subhalo counts and binned radial profile statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_pop2prime_halo_catalog(snapshot=snapshot, target_redshift=target_redshift)
    snapshot = catalog["snapshot"]
    metadata = find_snapshot_metadata(snapshot)
    geometric_parent_ids = get_geometric_parent_ids(catalog)
    host_mask = (geometric_parent_ids == -1) & (catalog["masses"] >= host_mass_min)
    host_masses = catalog["masses"][host_mask]
    host_mass_max = np.max(host_masses) if host_masses.size > 0 else np.nan

    output_path = output_dir / f"pop2prime_radial_profiles_geometric_allpsi_DD{snapshot:04d}.txt"
    with output_path.open("w") as f:
        f.write(f"# snapshot {snapshot}\n")
        if metadata is not None:
            f.write(f"# redshift {metadata['z']:.6f}\n")
            f.write(f"# scale_factor {metadata['a']:.6f}\n")
        else:
            f.write(f"# redshift {catalog['redshift']:.6f}\n")
            f.write(f"# scale_factor {catalog['scale_factor']:.6f}\n")
        f.write(f"# host_mass_min_Msunh {host_mass_min:.8e}\n")
        f.write(f"# host_mass_max_Msunh {host_mass_max:.8e}\n")
        f.write(f"# subhalo_definition {subhalo_definition}\n")
        f.write("# profile_definition dN/dx^3 with x=r_sub-host/Rvir\n")
        f.write("\n")

        for psi_min in psi_thresholds:
            radial_data = compute_subhalo_count_profile_sample(
                catalog,
                geometric_parent_ids,
                host_mass_min=host_mass_min,
                psi_min=psi_min,
            )
            profile = compute_host_averaged_subhalo_count_profile(radial_data)
            n_hosts_with_subhalos = get_host_count_with_subhalos(radial_data)

            f.write(f"# psi_min {psi_min:.8e}\n")
            f.write(f"# n_hosts_total {radial_data['n_hosts']}\n")
            f.write(f"# n_hosts_with_subhalos {n_hosts_with_subhalos}\n")
            f.write(f"# n_subhalos {radial_data['n_subhalos']}\n")
            f.write(
                "# columns: x_left x_right x_center "
                "mean_dN_dx3 median_dN_dx3 p16_dN_dx3 p84_dN_dx3\n"
            )

            for i in range(len(profile["x_centers"])):
                f.write(
                    f"{profile['x_edges'][i]:.8e} "
                    f"{profile['x_edges'][i + 1]:.8e} "
                    f"{profile['x_centers'][i]:.8e} "
                    f"{profile['mean_profile'][i]:.8e} "
                    f"{profile['median_profile'][i]:.8e} "
                    f"{profile['p16_profile'][i]:.8e} "
                    f"{profile['p84_profile'][i]:.8e}\n"
                )
            f.write("\n")

    print(f"Saved radial profile table to {output_path}")
    return output_path


def load_exported_subhalo_count_profiles_txt(input_path):
    """Load an exported Pop2Prime dN/dx^3 radial-profile table."""
    input_path = Path(input_path)
    metadata = {}
    profiles = []
    current = None
    rows = []

    def finish_current_profile():
        if current is None:
            return
        data = np.asarray(rows, dtype=float)
        if data.size == 0:
            data = np.zeros((0, 7), dtype=float)
        current.update(
            {
                "x_left": data[:, 0],
                "x_right": data[:, 1],
                "x_centers": data[:, 2],
                "mean_profile": data[:, 3],
                "median_profile": data[:, 4],
                "p16_profile": data[:, 5],
                "p84_profile": data[:, 6],
            }
        )
        profiles.append(current)

    with input_path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line[1:].strip().split()
                if not parts:
                    continue
                key = parts[0]
                if key == "psi_min":
                    finish_current_profile()
                    current = {"psi_min": float(parts[1])}
                    rows = []
                elif current is not None and key in {"n_hosts_total", "n_hosts_with_subhalos", "n_subhalos"}:
                    current[key] = int(parts[1])
                elif current is None and len(parts) >= 2:
                    value = parts[1]
                    try:
                        metadata[key] = float(value)
                    except ValueError:
                        metadata[key] = " ".join(parts[1:])
                continue
            rows.append([float(value) for value in line.split()])

    finish_current_profile()
    return {"metadata": metadata, "profiles": profiles, "input_path": input_path}


def infer_mean_host_mass_for_exported_profiles(input_path, snapshot):
    """Return mean host mass from the matching per-host profile table if present."""
    input_path = Path(input_path)
    candidate_paths = [
        input_path.parent / f"pop2prime_host_gas_profiles_DD{snapshot:04d}" / f"all_host_profiles_DD{snapshot:04d}.txt",
        input_path.parent / f"pop2prime_host_gas_profiles_DD{snapshot:04d}" / f"all_host_profiles_withoutT_DD{snapshot:04d}.txt",
    ]
    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue
        data = np.loadtxt(candidate_path)
        if data.ndim == 1:
            data = data[None, :]
        host_ids = data[:, 1].astype(int)
        host_masses = data[:, 2]
        _, first_indices = np.unique(host_ids, return_index=True)
        return float(np.mean(host_masses[first_indices])), candidate_path
    return np.nan, None


def plot_exported_pop2prime_radial_profiles_with_jb17(
    input_path,
    output_dir=None,
    normalize_at_vir=True,
    show_percentile=False,
    show_median=True,
):
    """Plot exported Pop2Prime dN/dx^3 profiles against NFW and corrected JB17."""
    loaded = load_exported_subhalo_count_profiles_txt(input_path)
    metadata = loaded["metadata"]
    profiles = loaded["profiles"]
    if len(profiles) == 0:
        raise RuntimeError(f"No profile blocks were found in {input_path}")

    input_path = loaded["input_path"]
    output_dir = Path(output_dir) if output_dir is not None else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    redshift = float(metadata.get("redshift", TARGET_REDSHIFT))
    host_mass_min = float(metadata.get("host_mass_min_Msunh", HOST_MASS_MIN))
    host_mass_max = float(metadata.get("host_mass_max_Msunh", host_mass_min))
    snapshot = int(metadata.get("snapshot", -1))
    mean_host_mass, mean_host_mass_source = infer_mean_host_mass_for_exported_profiles(
        input_path,
        snapshot,
    )
    if np.isfinite(mean_host_mass) and mean_host_mass > 0:
        reference_host_mass = mean_host_mass
        reference_mass_label = r"mean host mass"
        print(f"Using mean host mass for analytic reference: {reference_host_mass:.8e} Msun/h")
        print(f"Mean host mass source: {mean_host_mass_source}")
    else:
        reference_host_mass = np.sqrt(host_mass_min * host_mass_max)
        reference_mass_label = r"$\sqrt{M_{\rm min}M_{\rm max}}$ fallback"
        print(f"Using fallback reference host mass: {reference_host_mass:.8e} Msun/h")

    fig, axes = plt.subplots(1, len(profiles), figsize=(17, 5.5), sharey=True, facecolor="white")
    if len(profiles) == 1:
        axes = [axes]

    artificial_small = 1.0e-12
    reference_concentration = np.nan
    for ax, profile in zip(axes, profiles):
        x_centers = profile["x_centers"]
        mean_profile = profile["mean_profile"]
        if normalize_at_vir:
            mean_profile, _ = normalize_profile_at_x(x_centers, mean_profile, x_target=1.0)
        mean_profile = np.where(np.isfinite(mean_profile) & (mean_profile > 0), mean_profile, artificial_small)
        ax.plot(
            x_centers,
            mean_profile,
            color="black",
            linewidth=2.8,
            label=_append_count_label(
                "Pop2Prime mean",
                profile.get("n_hosts_total"),
                profile.get("n_hosts_with_subhalos"),
                profile.get("n_subhalos"),
            ),
        )

        p16_profile = profile["p16_profile"]
        median_profile = profile["median_profile"]
        p84_profile = profile["p84_profile"]
        if normalize_at_vir:
            p16_profile, _ = normalize_profile_at_x(x_centers, p16_profile, x_target=1.0)
            median_profile, _ = normalize_profile_at_x(x_centers, median_profile, x_target=1.0)
            p84_profile, _ = normalize_profile_at_x(x_centers, p84_profile, x_target=1.0)
        p16_profile = np.where(np.isfinite(p16_profile) & (p16_profile > 0), p16_profile, artificial_small)
        median_profile = np.where(np.isfinite(median_profile) & (median_profile > 0), median_profile, artificial_small)
        p84_profile = np.where(np.isfinite(p84_profile) & (p84_profile > 0), p84_profile, artificial_small)
        if show_percentile:
            ax.fill_between(x_centers, p16_profile, p84_profile, color="black", alpha=0.18, linewidth=0)
        if show_median:
            ax.plot(x_centers, median_profile, color="black", linestyle="--", linewidth=2.0, label="median")

        nfw_reference, jb17_reference, han16_reference, reference_concentration = build_jb17_reference_profiles(
            x_centers,
            reference_host_mass,
            redshift,
        )
        nfw_reference = np.where(np.isfinite(nfw_reference) & (nfw_reference > 0), nfw_reference, artificial_small)
        jb17_reference = np.where(np.isfinite(jb17_reference) & (jb17_reference > 0), jb17_reference, artificial_small)
        han16_reference = np.where(np.isfinite(han16_reference) & (han16_reference > 0), han16_reference, artificial_small)
        ax.plot(
            x_centers,
            nfw_reference,
            color="#D85A30",
            linewidth=2.0,
            linestyle="-.",
            label=rf"NFW, $c={reference_concentration:.2f}$",
        )
        ax.plot(
            x_centers,
            jb17_reference,
            color="#1D9E75",
            linewidth=2.0,
            linestyle="--",
            label=rf"NFW $\times$ JB17, $(\eta,\mu)=({JB17_ETA:.0f},{JB17_MU:.0f})$",
        )
        ax.plot(
            x_centers,
            han16_reference,
            color="#7B61B7",
            linewidth=2.0,
            linestyle=":",
            label=rf"NFW $\times$ Han16, $\gamma={HAN16_GAMMA:.2f}$",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1.0e-3)
        ax.axvline(1.0, color="black", linestyle=":", linewidth=1.2)
        ax.set_xlabel(r"$x=d_{\mathrm{sub-host}}/R_{\mathrm{vir}}$", fontsize=13)
        ax.set_title(rf"$\psi > {profile['psi_min']:.0e}$", fontsize=13)
        ax.tick_params(direction="in", which="both", labelsize=11)

    ylabel = r"$\mathrm{d}N_{\mathrm{sub}}/\mathrm{d}x^3$"
    if normalize_at_vir:
        ylabel = "Normalized " + ylabel
    axes[0].set_ylabel(ylabel, fontsize=14)
    fig.suptitle(
        rf"Pop2Prime radial subhalo profiles, z={redshift:.2f}, "
        rf"analytic ref ({reference_mass_label}): "
        rf"$M_{{\mathrm{{host}}}}=10^{{{np.log10(reference_host_mass):.1f}}}\,M_\odot/h$, "
        rf"{CONCENTRATION_LABEL} $c={reference_concentration:.2f}$",
        fontsize=13,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(fontsize=7, ncol=2)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    norm_tag = "_normRvir" if normalize_at_vir else ""
    output_path = output_dir / f"{input_path.stem}{norm_tag}_JB17eta{JB17_ETA:.0f}mu{JB17_MU:.0f}_Han16gamma{HAN16_GAMMA:.2f}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved exported Pop2Prime JB17 comparison plot: {output_path}")
    return output_path


def plot_pop2prime_radial_subhalo_profiles_allpsi(
    snapshot=None,
    target_redshift=TARGET_REDSHIFT,
    host_mass_min=HOST_MASS_MIN,
    psi_thresholds=PSI_THRESHOLDS_FOR_EXPORT,
    output_dir=POP2PRIME_RESULTS_DIR,
    show_average=True,
    show_individual=False,
    show_percentile=False,
    show_jb17_reference=True,
    normalize_at_vir=True,
):
    """
    Plot Pop2Prime radial subhalo profiles for all psi thresholds in one figure.

    The figure can show the host-averaged dN/dx^3 profile, rainbow-colored
    individual host profiles, percentile shading, or any combination of them.
    Individual-host curves use a fixed host-ID color mapping across the psi
    panels for easier visual comparison.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_pop2prime_halo_catalog(snapshot=snapshot, target_redshift=target_redshift)
    snapshot = catalog["snapshot"]
    metadata = find_snapshot_metadata(snapshot)
    redshift = metadata["z"] if metadata is not None else catalog["redshift"]
    geometric_parent_ids = get_geometric_parent_ids(catalog)
    host_mass_tag = format_host_mass_min_tag(host_mass_min)

    host_dir = output_dir / f"pop2prime_host_gas_profiles_DD{snapshot:04d}"
    host_dir.mkdir(parents=True, exist_ok=True)

    profile_by_psi = {}
    all_host_ids = set()
    all_host_masses = []
    for psi_min in psi_thresholds:
        radial_data = compute_subhalo_count_profile_sample(
            catalog,
            geometric_parent_ids,
            host_mass_min=host_mass_min,
            psi_min=psi_min,
        )
        profile = compute_host_averaged_subhalo_count_profile(radial_data)
        profile["n_hosts_with_subhalos"] = get_host_count_with_subhalos(radial_data)
        profile["n_subhalos"] = radial_data["n_subhalos"]
        profile_by_psi[psi_min] = profile
        all_host_ids.update(profile["host_ids"].tolist())
        all_host_masses.extend(profile["host_masses"].tolist())

    sorted_host_ids = np.array(sorted(all_host_ids), dtype=int)
    host_color_map = {}
    if sorted_host_ids.size > 0:
        host_colors = plt.cm.rainbow(np.linspace(0.0, 1.0, sorted_host_ids.size))
        host_color_map = {host_id: color for host_id, color in zip(sorted_host_ids, host_colors)}

    host_mass_min_used = min(all_host_masses) if all_host_masses else host_mass_min
    reference_host_mass = np.median(all_host_masses) if all_host_masses else host_mass_min
    fig, axes = plt.subplots(1, len(psi_thresholds), figsize=(17, 5.5), sharey=True, facecolor="white")
    if len(psi_thresholds) == 1:
        axes = [axes]

    artificial_small = 1.0e-12
    for ax, psi_min in zip(axes, psi_thresholds):
        profile = profile_by_psi[psi_min]
        x_centers = profile["x_centers"]

        if show_individual and profile["host_ids"].size > 0:
            host_order = np.argsort(profile["host_ids"])
            for row_index in host_order:
                host_id = int(profile["host_ids"][row_index])
                y = profile["profile_matrix"][row_index]
                if normalize_at_vir:
                    y, _ = normalize_profile_at_x(x_centers, y, x_target=1.0)
                y_plot = np.where(np.isfinite(y) & (y > 0), y, artificial_small)
                ax.plot(
                    x_centers,
                    y_plot,
                    color=host_color_map[host_id],
                    linewidth=1.2,
                    alpha=0.8,
                    label=f"host {host_id}",
                )

        if show_average:
            mean_profile = profile["mean_profile"]
            if normalize_at_vir:
                mean_profile, mean_norm = normalize_profile_at_x(x_centers, mean_profile, x_target=1.0)
            mean_profile = np.where(np.isfinite(mean_profile) & (mean_profile > 0), mean_profile, artificial_small)
            ax.plot(
                x_centers,
                mean_profile,
                color="black",
                linewidth=2.8,
                label=_append_count_label(
                    "Pop2Prime mean",
                    profile["n_hosts"],
                    profile["n_hosts_with_subhalos"],
                    profile["n_subhalos"],
                ),
            )
        if show_percentile:
            p16_profile = profile["p16_profile"]
            median_profile = profile["median_profile"]
            p84_profile = profile["p84_profile"]
            if normalize_at_vir:
                p16_profile, _ = normalize_profile_at_x(x_centers, p16_profile, x_target=1.0)
                median_profile, _ = normalize_profile_at_x(x_centers, median_profile, x_target=1.0)
                p84_profile, _ = normalize_profile_at_x(x_centers, p84_profile, x_target=1.0)
            p16_profile = np.where(np.isfinite(p16_profile) & (p16_profile > 0), p16_profile, artificial_small)
            median_profile = np.where(np.isfinite(median_profile) & (median_profile > 0), median_profile, artificial_small)
            p84_profile = np.where(np.isfinite(p84_profile) & (p84_profile > 0), p84_profile, artificial_small)
            ax.fill_between(
                x_centers,
                p16_profile,
                p84_profile,
                color="black",
                alpha=0.18,
                linewidth=0,
                label="16-84th percentile",
            )
            ax.plot(
                x_centers,
                median_profile,
                color="black",
                linestyle="--",
                linewidth=2.0,
                label="median",
            )

        if show_jb17_reference:
            nfw_reference, jb17_reference, han16_reference, reference_concentration = build_jb17_reference_profiles(
                x_centers,
                reference_host_mass,
                redshift,
            )
            nfw_reference = np.where(
                np.isfinite(nfw_reference) & (nfw_reference > 0),
                nfw_reference,
                artificial_small,
            )
            jb17_reference = np.where(
                np.isfinite(jb17_reference) & (jb17_reference > 0),
                jb17_reference,
                artificial_small,
            )
            han16_reference = np.where(
                np.isfinite(han16_reference) & (han16_reference > 0),
                han16_reference,
                artificial_small,
            )
            ax.plot(
                x_centers,
                nfw_reference,
                color="#D85A30",
                linewidth=2.0,
                linestyle="-.",
                label=rf"NFW, $c={reference_concentration:.2f}$",
            )
            ax.plot(
                x_centers,
                jb17_reference,
                color="#1D9E75",
                linewidth=2.0,
                linestyle="--",
                label=rf"NFW $\times$ JB17, $(\eta,\mu)=({JB17_ETA:.0f},{JB17_MU:.0f})$",
            )
            ax.plot(
                x_centers,
                han16_reference,
                color="#7B61B7",
                linewidth=2.0,
                linestyle=":",
                label=rf"NFW $\times$ Han16, $\gamma={HAN16_GAMMA:.2f}$",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(1.0, color="black", linestyle=":", linewidth=1.2)
        ax.set_xlabel(r"$x=d_{\mathrm{sub-host}}/R_{\mathrm{vir}}$", fontsize=13)
        ax.set_title(rf"$\psi > {psi_min:.0e}$", fontsize=13)
        ax.tick_params(direction="in", which="both", labelsize=11)

    ylabel = r"$\mathrm{d}N_{\mathrm{sub}}/\mathrm{d}x^3$"
    if normalize_at_vir:
        ylabel = "Normalized " + ylabel
    axes[0].set_ylabel(ylabel, fontsize=14)
    fig.suptitle(
        rf"Pop2Prime radial subhalo profiles, z={redshift:.2f}, "
        rf"$M_{{\mathrm{{host}}}} \geq 10^{{{np.log10(host_mass_min_used):.1f}}}\,M_\odot/h$",
        fontsize=13,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(fontsize=7, ncol=2)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    mode_tag = f"avg{int(show_average)}_ind{int(show_individual)}_pct{int(show_percentile)}"
    ref_tag = f"_JB17eta{JB17_ETA:.0f}mu{JB17_MU:.0f}_Han16gamma{HAN16_GAMMA:.2f}" if show_jb17_reference else ""
    norm_tag = "_normRvir" if normalize_at_vir else ""
    output_path = host_dir / f"pop2prime_radial_profiles_allpsi_{mode_tag}{norm_tag}{ref_tag}_{host_mass_tag}_DD{snapshot:04d}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Pop2Prime all-psi radial subhalo profile plot: {output_path}")
    return output_path


def plot_pop2prime_radial_subhalo_weighted_profile(
    snapshot=None,
    target_redshift=TARGET_REDSHIFT,
    host_mass_min=HOST_MASS_MIN,
    psi_min=1.0e-3,
    output_dir=POP2PRIME_RESULTS_DIR,
    show_average=True,
    show_individual=False,
    show_percentile=False,
    statistic="mass2_dx",
):
    """
    Plot one Pop2Prime radial subhalo profile using a low psi threshold and a
    weighted d/dx statistic suited for DF-heating-oriented analysis.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_pop2prime_halo_catalog(snapshot=snapshot, target_redshift=target_redshift)
    snapshot = catalog["snapshot"]
    metadata = find_snapshot_metadata(snapshot)
    redshift = metadata["z"] if metadata is not None else catalog["redshift"]
    geometric_parent_ids = get_geometric_parent_ids(catalog)
    host_mass_tag = format_host_mass_min_tag(host_mass_min)

    host_dir = output_dir / f"pop2prime_host_gas_profiles_DD{snapshot:04d}"
    host_dir.mkdir(parents=True, exist_ok=True)

    radial_data = compute_subhalo_count_profile_sample(
        catalog,
        geometric_parent_ids,
        host_mass_min=host_mass_min,
        psi_min=psi_min,
    )
    profile = compute_host_averaged_subhalo_weighted_profile(
        radial_data,
        statistic=statistic,
    )
    profile["n_hosts_with_subhalos"] = get_host_count_with_subhalos(radial_data)
    profile["n_subhalos"] = radial_data["n_subhalos"]

    sorted_host_ids = np.array(sorted(profile["host_ids"].tolist()), dtype=int)
    host_color_map = {}
    if sorted_host_ids.size > 0:
        host_colors = plt.cm.rainbow(np.linspace(0.0, 1.0, sorted_host_ids.size))
        host_color_map = {host_id: color for host_id, color in zip(sorted_host_ids, host_colors)}

    ylabel_map = {
        "count_dx": r"$\mathrm{d}N_{\mathrm{sub}}/\mathrm{d}x$",
        "mass_dx": r"$\mathrm{d}\sum (m_{\mathrm{sub}}/M_{\mathrm{host}})/\mathrm{d}x$",
        "mass2_dx": r"$\mathrm{d}\sum (m_{\mathrm{sub}}/M_{\mathrm{host}})^2/\mathrm{d}x$",
    }
    title_map = {
        "count_dx": "count-weighted",
        "mass_dx": "mass-weighted",
        "mass2_dx": "mass-squared-weighted",
    }
    if statistic not in ylabel_map:
        raise ValueError(f"Unknown statistic: {statistic}")

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    artificial_small = 1.0e-12
    x_centers = profile["x_centers"]

    if show_individual and profile["host_ids"].size > 0:
        host_order = np.argsort(profile["host_ids"])
        for row_index in host_order:
            host_id = int(profile["host_ids"][row_index])
            y = profile["profile_matrix"][row_index]
            y_plot = np.where(y > 0, y, artificial_small)
            ax.plot(
                x_centers,
                y_plot,
                color=host_color_map[host_id],
                linewidth=1.2,
                alpha=0.8,
                label=f"host {host_id}",
            )

    if show_average:
        mean_profile = np.where(profile["mean_profile"] > 0, profile["mean_profile"], artificial_small)
        ax.plot(
            x_centers,
            mean_profile,
            color="black",
            linewidth=2.8,
            label=_append_count_label(
                "Pop2Prime mean",
                profile["n_hosts"],
                profile["n_hosts_with_subhalos"],
                profile["n_subhalos"],
            ),
        )
    if show_percentile:
        p16_profile = np.where(profile["p16_profile"] > 0, profile["p16_profile"], artificial_small)
        median_profile = np.where(profile["median_profile"] > 0, profile["median_profile"], artificial_small)
        p84_profile = np.where(profile["p84_profile"] > 0, profile["p84_profile"], artificial_small)
        ax.fill_between(
            x_centers,
            p16_profile,
            p84_profile,
            color="black",
            alpha=0.18,
            linewidth=0,
            label="16-84th percentile",
        )
        ax.plot(
            x_centers,
            median_profile,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="median",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1.0e-3)
    ax.axvline(1.0, color="black", linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$x=d_{\mathrm{sub-host}}/R_{\mathrm{vir}}$", fontsize=14)
    ax.set_ylabel(ylabel_map[statistic], fontsize=14)
    ax.set_title(
        rf"Pop2Prime {title_map[statistic]} profile, z={redshift:.2f}, "
        rf"$M_{{\mathrm{{host}}}} \geq 10^{{{np.log10(host_mass_min):.1f}}}\,M_\odot/h$, "
        rf"$\psi > {psi_min:.0e}$",
        fontsize=12,
    )
    ax.tick_params(direction="in", which="both", labelsize=12)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    mode_tag = f"avg{int(show_average)}_ind{int(show_individual)}_pct{int(show_percentile)}"
    output_path = host_dir / (
        f"pop2prime_radial_weighted_{statistic}_{mode_tag}_psi_gt_{psi_min:.0e}_{host_mass_tag}_DD{snapshot:04d}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Pop2Prime weighted radial subhalo profile plot: {output_path}")
    return output_path

def run_pop2prime_subhalo_count_profile_analysis(
    snapshot=None,
    target_redshift=TARGET_REDSHIFT,
    host_mass_min=HOST_MASS_MIN,
    psi_min=PSI_MIN,
    output_dir=POP2PRIME_RESULTS_DIR,
):
    """
    Run Pop2Prime radial subhalo statistics for z~12 host halos.

    The geometric parent definition is always evaluated.  The native Rockstar
    definition is attempted only if a parent-ID field is present in the halo
    catalog.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_pop2prime_halo_catalog(snapshot=snapshot, target_redshift=target_redshift)
    snapshot = catalog["snapshot"]
    metadata = find_snapshot_metadata(snapshot)
    if metadata is not None:
        print(
            f"Using snapshot DD{snapshot:04d} for target z={target_redshift:.2f}: "
            f"a={metadata['a']:.6f}, z={metadata['z']:.6f}"
        )
    else:
        print(
            f"Using snapshot DD{snapshot:04d} for target z={target_redshift:.2f}: "
            f"yt redshift z={catalog['redshift']:.6f}"
        )

    radial_data_by_method = {}
    profile_by_method = {}

    geometric_parent_ids = get_geometric_parent_ids(catalog)
    geometric_data = compute_subhalo_count_profile_sample(
        catalog,
        geometric_parent_ids,
        host_mass_min=host_mass_min,
        psi_min=psi_min,
    )
    print("\nGeometric parent definition:")
    summarize_subhalo_count_profile_sample(geometric_data)
    radial_data_by_method["geometric"] = geometric_data
    profile_by_method["geometric"] = compute_host_averaged_subhalo_count_profile(geometric_data)

    try:
        native_parent_ids = get_rockstar_native_parent_ids(catalog)
    except RuntimeError as exc:
        print("\nRockstar native parent definition unavailable:")
        print(str(exc))
    else:
        native_data = compute_subhalo_count_profile_sample(
            catalog,
            native_parent_ids,
            host_mass_min=host_mass_min,
            psi_min=psi_min,
        )
        print("\nRockstar native parent definition:")
        summarize_subhalo_count_profile_sample(native_data)
        radial_data_by_method["rockstar_native"] = native_data
        profile_by_method["rockstar_native"] = compute_host_averaged_subhalo_count_profile(native_data)

    for method_name, radial_data in radial_data_by_method.items():
        sample_path = output_dir / (
            f"pop2prime_radial_sample_{method_name}_psi_gt_{psi_min:.2f}_DD{snapshot:04d}.txt"
        )
        save_subhalo_count_profile_sample(radial_data, sample_path)

    if profile_by_method:
        plot_path = output_dir / (
            f"pop2prime_radial_profile_comparison_psi_gt_{psi_min:.2f}_DD{snapshot:04d}.png"
        )
        plot_subhalo_count_profiles_comparison(
            profile_by_method,
            snapshot=snapshot,
            redshift=catalog["redshift"],
            output_path=plot_path,
            host_mass_min=host_mass_min,
            psi_min=psi_min,
        )

    return {
        "catalog": catalog,
        "radial_data_by_method": radial_data_by_method,
        "profile_by_method": profile_by_method,
    }


#--------------------------------------------------------------
#------------------- gas density profile -----------------------
#--------------------------------------------------------------

def get_host_sphere(hydro_ds, halo_data, host_index, radius_scale=1.0):
    """Return a yt sphere centered on one halo with radius_scale * Rvir."""
    center = hydro_ds.arr(
        halo_data["halos", "particle_position"][host_index].to("unitary"),
        "unitary",
    )
    radius = hydro_ds.quan(
        halo_data["halos", "virial_radius"][host_index].to("unitary"),
        "unitary",
    )
    return hydro_ds.sphere(center, radius_scale * radius)


def compute_single_host_gas_and_total_profiles(
    catalog,
    hydro_ds,
    host_index,
    x_min=1.0e-2,
    x_max=2.0,
    num_x_bins=20,
    log_x_bins=True,
    density_unit="g/cm**3",
    metadata=None,
):
    """Compute one host halo's gas and total density profiles in spherical shells."""
    def accumulate_shell_mass_from_chunks(data_source, radius_field, mass_field, field_label):
        """Accumulate shell mass from yt chunks without materializing full-field arrays."""
        shell_mass_msun = np.zeros(num_x_bins, dtype=float)
        valid_count = 0
        radius_min = np.inf
        radius_max = -np.inf

        for chunk in data_source.chunks([radius_field, mass_field], "io"):
            radius_kpc = chunk[radius_field].to("kpc").to_ndarray()
            mass_msun = chunk[mass_field].to("Msun").to_ndarray()
            x = radius_kpc / host_rvir_kpc

            valid = (
                np.isfinite(x)
                & np.isfinite(mass_msun)
                & (x >= x_edges[0])
                & (x <= x_edges[-1])
                & (mass_msun >= 0)
            )
            if not np.any(valid):
                continue

            valid_x = x[valid]
            shell_mass_msun += np.histogram(
                valid_x,
                bins=x_edges,
                weights=mass_msun[valid],
            )[0]
            valid_count += valid_x.size
            radius_min = min(radius_min, float(np.min(valid_x)))
            radius_max = max(radius_max, float(np.max(valid_x)))

        if valid_count == 0:
            raise RuntimeError(
                f"No {field_label} elements were found in the requested radial range for this host halo."
            )

        return shell_mass_msun, valid_count, radius_min, radius_max

    def accumulate_mass_weighted_scalar_from_chunks(data_source, radius_field, mass_field, scalar_field, field_label):
        """Accumulate a mass-weighted scalar profile from yt chunks."""
        shell_weighted_scalar = np.zeros(num_x_bins, dtype=float)
        shell_mass_msun = np.zeros(num_x_bins, dtype=float)
        valid_count = 0

        for chunk in data_source.chunks([radius_field, mass_field, scalar_field], "io"):
            radius_kpc = chunk[radius_field].to("kpc").to_ndarray()
            mass_msun = chunk[mass_field].to("Msun").to_ndarray()
            scalar = chunk[scalar_field].to_ndarray()
            x = radius_kpc / host_rvir_kpc

            valid = (
                np.isfinite(x)
                & np.isfinite(mass_msun)
                & np.isfinite(scalar)
                & (x >= x_edges[0])
                & (x <= x_edges[-1])
                & (mass_msun > 0)
            )
            if not np.any(valid):
                continue

            valid_x = x[valid]
            valid_mass = mass_msun[valid]
            valid_scalar = scalar[valid]
            shell_weighted_scalar += np.histogram(
                valid_x,
                bins=x_edges,
                weights=valid_mass * valid_scalar,
            )[0]
            shell_mass_msun += np.histogram(
                valid_x,
                bins=x_edges,
                weights=valid_mass,
            )[0]
            valid_count += valid_x.size

        if valid_count == 0:
            raise RuntimeError(
                f"No valid {field_label} elements were found in the requested radial range for this host halo."
            )

        shell_scalar_mw = np.divide(
            shell_weighted_scalar,
            shell_mass_msun,
            out=np.zeros_like(shell_weighted_scalar),
            where=shell_mass_msun > 0,
        )
        return shell_scalar_mw, shell_mass_msun, valid_count

    host_id = int(catalog["halo_ids"][host_index])
    host_mass = float(catalog["masses"][host_index])
    host_rvir_kpc = float(catalog["all_data"]["halos", "virial_radius"][host_index].to("kpc"))
    x_edges, x_centers, dx3 = build_x_bins(
        x_min=x_min,
        x_max=x_max,
        num_x_bins=num_x_bins,
        log_x_bins=log_x_bins,
    )

    print(
        "Processing host halo index {host_index} with ID {host_id}, mass {host_mass:.2e} Msun/h, "
        "Rvir {host_rvir_kpc:.2f} kpc".format(
            host_index=host_index,
            host_id=host_id,
            host_mass=host_mass,
            host_rvir_kpc=host_rvir_kpc,
        )
    )
    host_sphere = get_host_sphere(
        hydro_ds,
        catalog["all_data"],
        host_index,
        radius_scale=x_max,
    )

    print("finished loading host sphere, now extracting gas and particle data for density profile calculation...")
    shell_gas_mass_msun, n_valid_gas, gas_x_min, gas_x_max = accumulate_shell_mass_from_chunks(
        host_sphere,
        ("index", "radius"),
        ("gas", "cell_mass"),
        "gas",
    )
    print(f"Found {n_valid_gas} valid gas cells for density profile calculation.")
    print("self-consistency check: gas_x min {:.3e}, max {:.3e}".format(gas_x_min, gas_x_max))

    shell_volume_kpc3 = (4.0 * np.pi / 3.0) * (host_rvir_kpc**3) * dx3
    shell_gas_density_msun_kpc3 = np.divide(
        shell_gas_mass_msun,
        shell_volume_kpc3,
        out=np.zeros_like(shell_gas_mass_msun, dtype=float),
        where=shell_volume_kpc3 > 0,
    )

    print("finished extracting gas mass data, now extracting gas temperature data for profile calculation...")
    shell_gas_temperature_mw, shell_gas_mass_for_temperature_msun, n_valid_temperature = (
        accumulate_mass_weighted_scalar_from_chunks(
            host_sphere,
            ("index", "radius"),
            ("gas", "cell_mass"),
            ("gas", "temperature"),
            "gas temperature",
        )
    )
    print(f"Found {n_valid_temperature} valid gas cells for temperature profile calculation.")

    print("finished extracting gas data, now extracting particle data for density profile calculation...")
    shell_particle_mass_msun, n_valid_particles, particle_x_min, particle_x_max = accumulate_shell_mass_from_chunks(
        host_sphere,
        ("nbody", "particle_radius"),
        ("nbody", "particle_mass"),
        "particle",
    )
    print(f"Found {n_valid_particles} valid particles for density profile calculation.")
    print("self-consistency check: particle_x min {:.3e}, max {:.3e}".format(particle_x_min, particle_x_max))

    shell_total_mass_msun = shell_gas_mass_msun + shell_particle_mass_msun
    shell_total_density_msun_kpc3 = np.divide(
        shell_total_mass_msun,
        shell_volume_kpc3,
        out=np.zeros_like(shell_total_mass_msun, dtype=float),
        where=shell_volume_kpc3 > 0,
    )

    density_factor = hydro_ds.quan(1.0, "Msun/kpc**3").to(density_unit).value
    shell_gas_density = shell_gas_density_msun_kpc3 * density_factor
    shell_total_density = shell_total_density_msun_kpc3 * density_factor

    rho_vir_msun_kpc3 = (host_mass / h_Hubble) / (4.0 / 3.0 * np.pi * host_rvir_kpc**3)
    rho_vir = rho_vir_msun_kpc3 * density_factor

    z_compare = metadata["z"] if metadata is not None else catalog["redshift"]
    ez2 = (
        Omega_m * (1.0 + z_compare) ** 3
        + Omega_r * (1.0 + z_compare) ** 4
        + Omega_k * (1.0 + z_compare) ** 2
        + Omega_lambda
    )
    rho_crit_z_compare_msun_mpc3 = rho_crit_z0 * ez2
    rho_crit_z_compare = hydro_ds.quan(rho_crit_z_compare_msun_mpc3, "Msun/Mpc**3").to(density_unit).value
    overdensity_200_rho_crit = 200.0 * rho_crit_z_compare
    overdensity_200_rho_m = 200.0 * hydro_ds.quan(
        Omega_m * rho_crit_z0 * (1.0 + z_compare) ** 3,
        "Msun/Mpc**3",
    ).to(density_unit).value
    overdensity_200_rho_b = (Omega_b / Omega_m) * overdensity_200_rho_m

    shell_gas_density_norm = shell_gas_density / rho_vir
    shell_total_density_norm = shell_total_density / rho_vir
    overdensity_200_rho_crit_norm = overdensity_200_rho_crit / rho_vir
    overdensity_200_rho_m_norm = overdensity_200_rho_m / rho_vir
    overdensity_200_rho_b_norm = overdensity_200_rho_b / rho_vir
    return {
        "snapshot": catalog["snapshot"],
        "host_index": host_index,
        "host_id": host_id,
        "host_mass": host_mass,
        "host_rvir_kpc": host_rvir_kpc,
        "x_edges": x_edges,
        "x_centers": x_centers,
        "shell_gas_mass_msun": shell_gas_mass_msun,
        "shell_particle_mass_msun": shell_particle_mass_msun,
        "shell_total_mass_msun": shell_total_mass_msun,
        "shell_volume_kpc3": shell_volume_kpc3,
        "shell_gas_density": shell_gas_density,
        "shell_total_density": shell_total_density,
        "shell_gas_temperature_mw": shell_gas_temperature_mw,
        "shell_gas_mass_for_temperature_msun": shell_gas_mass_for_temperature_msun,
        "shell_gas_density_norm": shell_gas_density_norm,
        "shell_total_density_norm": shell_total_density_norm,
        "shell_gas_to_total_density_ratio": np.divide(
            shell_gas_density,
            shell_total_density,
            out=np.zeros_like(shell_gas_density),
            where=shell_total_density > 0,
        ),
        "rho_vir": rho_vir,
        "density_unit": density_unit,
        "rho_200_crit": overdensity_200_rho_crit,
        "rho_200_m": overdensity_200_rho_m,
        "rho_200_b": overdensity_200_rho_b,
        "rho_200_crit_norm": overdensity_200_rho_crit_norm,
        "rho_200_m_norm": overdensity_200_rho_m_norm,
        "rho_200_b_norm": overdensity_200_rho_b_norm,
        "redshift": z_compare,
    }


def save_single_host_profile_table(profile, output_path):
    """Save one host halo's radial gas and total density profiles to a txt table."""
    header_lines = [
        f"snapshot {profile['snapshot']}",
        f"redshift {profile['redshift']:.8e}",
        f"host_index {profile['host_index']}",
        f"host_id {profile['host_id']}",
        f"host_mass_Msunh {profile['host_mass']:.8e}",
        f"host_rvir_kpc {profile['host_rvir_kpc']:.8e}",
        f"rho_vir_{profile['density_unit']} {profile['rho_vir']:.8e}",
        f"rho_200_crit_{profile['density_unit']} {profile['rho_200_crit']:.8e}",
        f"rho_200_m_{profile['density_unit']} {profile['rho_200_m']:.8e}",
        f"rho_200_b_{profile['density_unit']} {profile['rho_200_b']:.8e}",
        (
            "columns: x_left x_right x_center shell_gas_mass_Msun shell_particle_mass_Msun "
            "shell_total_mass_Msun shell_volume_kpc3 rho_gas rho_total Tgas_mw_K "
            "rho_gas_over_rhovir rho_total_over_rhovir rho_gas_over_rho_total"
        ),
    ]
    table = np.column_stack(
        [
            profile["x_edges"][:-1],
            profile["x_edges"][1:],
            profile["x_centers"],
            profile["shell_gas_mass_msun"],
            profile["shell_particle_mass_msun"],
            profile["shell_total_mass_msun"],
            profile["shell_volume_kpc3"],
            profile["shell_gas_density"],
            profile["shell_total_density"],
            profile["shell_gas_temperature_mw"],
            profile["shell_gas_density_norm"],
            profile["shell_total_density_norm"],
            profile["shell_gas_to_total_density_ratio"],
        ]
    )
    np.savetxt(output_path, table, header="\n".join(header_lines), fmt="%.8e")
    print(f"Saved host profile table to {output_path}")


def save_host_profile_collection_table(host_profiles, output_path, host_mass_min=None):
    """Save all selected host-halo profiles for one snapshot to a single txt table."""
    if len(host_profiles) == 0:
        raise ValueError("host_profiles is empty.")

    reference_profile = host_profiles[0]
    if host_mass_min is None:
        host_mass_min = min(profile["host_mass"] for profile in host_profiles)
    header_lines = [
        f"snapshot {reference_profile['snapshot']}",
        f"redshift {reference_profile['redshift']:.8e}",
        f"density_unit {reference_profile['density_unit']}",
        f"n_hosts {len(host_profiles)}",
        f"host_mass_min_Msunh {host_mass_min:.8e}",
        (
            "columns: host_index host_id host_mass_Msunh host_rvir_kpc "
            "rho_vir rho_200_crit rho_200_m rho_200_b "
            "x_bin x_left x_right x_center "
            "shell_gas_mass_Msun shell_particle_mass_Msun shell_total_mass_Msun "
            "shell_volume_kpc3 rho_gas rho_total Tgas_mw_K "
            "rho_gas_over_rhovir rho_total_over_rhovir rho_gas_over_rho_total"
        ),
    ]

    rows = []
    for profile in host_profiles:
        num_bins = len(profile["x_centers"])
        for i in range(num_bins):
            rows.append(
                [
                    profile["host_index"],
                    profile["host_id"],
                    profile["host_mass"],
                    profile["host_rvir_kpc"],
                    profile["rho_vir"],
                    profile["rho_200_crit"],
                    profile["rho_200_m"],
                    profile["rho_200_b"],
                    i,
                    profile["x_edges"][i],
                    profile["x_edges"][i + 1],
                    profile["x_centers"][i],
                    profile["shell_gas_mass_msun"][i],
                    profile["shell_particle_mass_msun"][i],
                    profile["shell_total_mass_msun"][i],
                    profile["shell_volume_kpc3"][i],
                    profile["shell_gas_density"][i],
                    profile["shell_total_density"][i],
                    profile["shell_gas_temperature_mw"][i],
                    profile["shell_gas_density_norm"][i],
                    profile["shell_total_density_norm"][i],
                    profile["shell_gas_to_total_density_ratio"][i],
                ]
            )

    np.savetxt(output_path, np.asarray(rows, dtype=float), header="\n".join(header_lines), fmt="%.8e")
    print(f"Saved combined host profile table to {output_path}")


def load_host_profile_collection_table(input_path):
    """Load one combined host-profile txt table and reconstruct per-host profiles."""
    input_path = Path(input_path)
    metadata = {}
    column_names = None
    with input_path.open() as f:
        for line in f:
            if not line.startswith("#"):
                break
            content = line[1:].strip()
            if not content:
                continue
            if content.startswith("columns:"):
                column_names = content.split(":", 1)[1].strip().split()
                continue
            key, value = content.split(maxsplit=1)
            metadata[key] = value

    table = np.loadtxt(input_path)
    if table.ndim == 1:
        table = table[np.newaxis, :]

    if column_names is None:
        raise ValueError(f"Missing '# columns:' header in {input_path}.")
    if len(column_names) != table.shape[1]:
        raise ValueError(
            f"Column count mismatch in {input_path}: header lists {len(column_names)} "
            f"columns but table has {table.shape[1]}."
        )

    column_index = {name: i for i, name in enumerate(column_names)}

    def get_column(rows, name):
        if name in column_index:
            return rows[:, column_index[name]]
        raise ValueError(f"Required column '{name}' not found in {input_path}.")

    metadata["snapshot"] = int(float(metadata["snapshot"]))
    metadata["redshift"] = float(metadata["redshift"])
    metadata["n_hosts"] = int(float(metadata["n_hosts"]))
    if "Tgas_mw_K" not in column_index:
        raise ValueError(
            f"Required column 'Tgas_mw_K' not found in {input_path}. "
            "This loader now expects the new Pop2Prime host-profile format."
        )
    has_temperature_column = True

    host_profiles = []
    host_ids = np.unique(table[:, column_index["host_id"]].astype(int))
    for host_id in host_ids:
        host_rows = table[table[:, column_index["host_id"]].astype(int) == host_id]
        order = np.argsort(get_column(host_rows, "x_bin").astype(int))
        host_rows = host_rows[order]
        host_profiles.append(
            {
                "snapshot": metadata["snapshot"],
                "redshift": metadata["redshift"],
                "density_unit": metadata.get("density_unit", ""),
                "host_index": int(host_rows[0, column_index["host_index"]]),
                "host_id": int(host_rows[0, column_index["host_id"]]),
                "host_mass": float(host_rows[0, column_index["host_mass_Msunh"]]),
                "host_rvir_kpc": float(host_rows[0, column_index["host_rvir_kpc"]]),
                "rho_vir": float(host_rows[0, column_index["rho_vir"]]),
                "rho_200_crit": float(host_rows[0, column_index["rho_200_crit"]]),
                "rho_200_m": float(host_rows[0, column_index["rho_200_m"]]),
                "rho_200_b": float(host_rows[0, column_index["rho_200_b"]]),
                "x_edges": np.concatenate(
                    [get_column(host_rows, "x_left"), [host_rows[-1, column_index["x_right"]]]]
                ),
                "x_centers": get_column(host_rows, "x_center"),
                "shell_gas_mass_msun": get_column(host_rows, "shell_gas_mass_Msun"),
                "shell_particle_mass_msun": get_column(host_rows, "shell_particle_mass_Msun"),
                "shell_total_mass_msun": get_column(host_rows, "shell_total_mass_Msun"),
                "shell_volume_kpc3": get_column(host_rows, "shell_volume_kpc3"),
                "shell_gas_density": get_column(host_rows, "rho_gas"),
                "shell_total_density": get_column(host_rows, "rho_total"),
                "shell_gas_temperature_mw": get_column(host_rows, "Tgas_mw_K"),
                "shell_gas_density_norm": get_column(host_rows, "rho_gas_over_rhovir"),
                "shell_total_density_norm": get_column(host_rows, "rho_total_over_rhovir"),
                "shell_gas_to_total_density_ratio": get_column(host_rows, "rho_gas_over_rho_total"),
            }
        )

    return {
        "metadata": metadata,
        "host_profiles": host_profiles,
        "has_temperature_column": has_temperature_column,
    }


def plot_all_host_profiles_overplot(input_path, output_dir=None, host_mass_min=None):
    """Read one combined host-profile table and generate standard overplot figures."""
    loaded = load_host_profile_collection_table(input_path)
    metadata = loaded["metadata"]
    host_profiles = loaded["host_profiles"]
    has_temperature_column = loaded["has_temperature_column"]
    if len(host_profiles) == 0:
        raise RuntimeError(f"No host profiles were loaded from {input_path}.")

    input_path = Path(input_path)
    output_dir = input_path.parent if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = metadata["snapshot"]
    redshift = metadata["redshift"]
    if host_mass_min is not None:
        host_profiles = [
            profile for profile in host_profiles if profile["host_mass"] >= host_mass_min
        ]
        if len(host_profiles) == 0:
            raise RuntimeError(
                f"No host profiles satisfy host_mass_min={host_mass_min:.3e} Msun/h "
                f"in {input_path}."
            )
    host_profiles = sorted(host_profiles, key=lambda profile: profile["host_id"])
    label_host_mass_min = (
        infer_host_mass_min_for_label(input_path, metadata, host_profiles)
        if host_mass_min is None
        else float(host_mass_min)
    )
    host_mass_tag = format_host_mass_min_tag(label_host_mass_min)
    representative_host_mass_msun = np.median(
        [profile["host_mass"] for profile in host_profiles]
    ) / h_Hubble
    mean_rho_vir = np.mean([profile["rho_vir"] for profile in host_profiles])
    host_colors = plt.cm.rainbow(np.linspace(0.0, 1.0, len(host_profiles)))
    reference_profile = host_profiles[0]
    if has_temperature_column:
        temperature_ratio_path = output_dir / f"all_hosts_gas_temperature_over_tvir_{host_mass_tag}_DD{snapshot:04d}.png"
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
        for profile, color in zip(host_profiles, host_colors):
            host_mvir_msun = profile["host_mass"] / h_Hubble
            host_tvir = Temperature_Virial_analytic(host_mvir_msun, redshift)
            y = profile["shell_gas_temperature_mw"] / host_tvir
            valid = (
                np.isfinite(y)
                & (y > 0.0)
                & np.isfinite(profile["shell_gas_mass_msun"])
                & (profile["shell_gas_mass_msun"] >= 1.0)
            )
            if not np.any(valid):
                continue
            ax.plot(
                profile["x_centers"][valid],
                y[valid],
                color=color,
                linewidth=1.2,
                alpha=0.75,
                label=f"host {profile['host_id']}",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1.0e-2)
        ax.axvline(1.0, color="black", linestyle=":", linewidth=1.2)
        ax.axhline(1.0, color="tab:red", linestyle="--", linewidth=1.5, label=r"$T_{\rm gas}=T_{\rm vir}$")
        ax.set_xlabel(r"$x=r/R_{\mathrm{vir}}$", fontsize=14)
        ax.set_ylabel(r"$T_{\rm gas} / T_{\rm vir}$", fontsize=14)
        ax.set_title(
            rf"Pop2Prime host temperature profiles, z={redshift:.2f}, "
            rf"$M_{{\mathrm{{host}}}} \geq 10^{{{np.log10(label_host_mass_min):.1f}}}\,M_\odot/h$",
            fontsize=12,
        )
        ax.tick_params(direction="in", which="both", labelsize=12)
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(temperature_ratio_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved overplot figure to {temperature_ratio_path}")

    reference_lines = [
        {
            "y": reference_profile["rho_200_crit"] / mean_rho_vir,
            "color": "tab:red",
            "linestyle": "--",
            "label": rf"$200\,\rho_{{\rm crit}}(z={redshift:.2f}) / \bar{{\rho}}_{{\rm vir}}$",
        },
        {
            "y": reference_profile["rho_200_b"] / mean_rho_vir,
            "color": "tab:blue",
            "linestyle": ":",
            "label": rf"$200\,\rho_{{\rm b}}(z={redshift:.2f}) / \bar{{\rho}}_{{\rm vir}}$",
        },
    ]

    combined_density_output_path = (
        output_dir / f"all_hosts_density_profiles_combined_{host_mass_tag}_DD{snapshot:04d}.png"
    )
    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor="white")
    artificial_small = 1.0e-10
    for profile, color in zip(host_profiles, host_colors):
        total_density_plot = np.where(
            profile["shell_total_density_norm"] > 0,
            profile["shell_total_density_norm"],
            artificial_small,
        )
        total_density_plot = np.maximum(total_density_plot, PAPER_DENSITY_YMIN)
        gas_density_valid = (
            np.isfinite(profile["shell_gas_density_norm"])
            & (profile["shell_gas_density_norm"] > 0.0)
            & np.isfinite(profile["shell_gas_mass_msun"])
            & (profile["shell_gas_mass_msun"] >= 1.0)
        )
        ax.plot(
            profile["x_centers"],
            total_density_plot,
            color=color,
            linewidth=1.5,
            alpha=0.8,
            label=f"host {profile['host_id']}",
        )
        if np.any(gas_density_valid):
            ax.plot(
                profile["x_centers"][gas_density_valid],
                profile["shell_gas_density_norm"][gas_density_valid],
                color=color,
                linewidth=1.3,
                linestyle="--",
                alpha=0.8,
            )

    nfw_concentration = get_concentration(
        representative_host_mass_msun,
        redshift,
        CONCENTRATION_MODEL,
    )
    profile_radii = reference_profile["x_centers"]
    scaled_radii = profile_radii * nfw_concentration
    total_nfw_profile = density_NFW_profile(
        scaled_radii,
        representative_host_mass_msun,
        redshift,
        CONCENTRATION_MODEL,
    )
    total_nfw_profile = np.maximum(total_nfw_profile, PAPER_DENSITY_YMIN)
    ax.plot(
        profile_radii,
        total_nfw_profile,
        color="black",
        linestyle="-",
        linewidth=3.2,
        label=rf"NFW total ({CONCENTRATION_LABEL})",
        zorder=5,
    )

    gas_profiles_cosmic = []
    for alpha in PAPER_GAS_PROFILE_ALPHAS:
        gas_profiles_cosmic.append(
            gasdensity_arbitrary_profile(
                scaled_radii,
                representative_host_mass_msun,
                redshift,
                CONCENTRATION_MODEL,
                alpha=alpha,
            )
        )
    gas_profiles_cosmic = np.asarray(gas_profiles_cosmic)
    gas_band_lower = np.maximum(np.min(gas_profiles_cosmic, axis=0), PAPER_DENSITY_YMIN)
    gas_band_upper = np.maximum(np.max(gas_profiles_cosmic, axis=0), PAPER_DENSITY_YMIN)
    ax.fill_between(
        profile_radii,
        gas_band_lower,
        gas_band_upper,
        color="0.45",
        alpha=0.38,
        linewidth=0.0,
        label=(
            r"Gas gNFW band, $\alpha \in [-0.5,1.5]$, "
            rf"$f_{{\rm g}}=\Omega_{{\rm b}}/\Omega_{{\rm m}}={Omega_b / Omega_m:.3f}$"
        ),
        zorder=2,
    )

    low_fgas_scale = PAPER_GAS_FRACTION_LOW / (Omega_b / Omega_m)
    ax.fill_between(
        profile_radii,
        np.maximum(gas_band_lower * low_fgas_scale, PAPER_DENSITY_YMIN),
        np.maximum(gas_band_upper * low_fgas_scale, PAPER_DENSITY_YMIN),
        color="0.75",
        alpha=0.24,
        linewidth=0.0,
        label=(
            r"Gas gNFW band, $\alpha \in [-0.5,1.5]$, "
            rf"$f_{{\rm g}}={PAPER_GAS_FRACTION_LOW:.2f}$"
        ),
        zorder=1,
    )

    for ref in reference_lines:
        ax.axhline(
            ref["y"],
            color=ref["color"],
            linestyle=ref["linestyle"],
            linewidth=3.0,
            label=ref["label"],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(1.0, color="black", linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$x=r/R_{\mathrm{vir}}$", fontsize=14)
    ax.set_ylabel(r"$\rho / \rho_{\mathrm{vir}}$", fontsize=14)
    ax.set_title(
        rf"Pop2Prime host density profiles, z={redshift:.2f}, "
        rf"$M_{{\mathrm{{host}}}} \geq 10^{{{np.log10(label_host_mass_min):.1f}}}\,M_\odot/h$",
        fontsize=12,
    )
    ax.tick_params(direction="in", which="both", labelsize=12)

    style_handles = [
        Line2D([0], [0], color="0.25", linewidth=1.6, linestyle="-", label="host total density"),
        Line2D([0], [0], color="0.25", linewidth=1.4, linestyle="--", label="host gas density"),
        Line2D([0], [0], color="black", linewidth=3.2, linestyle="-", label=rf"NFW total ({CONCENTRATION_LABEL})"),
        Patch(
            facecolor="0.45",
            alpha=0.38,
            label=(
                r"gas gNFW, $\alpha \in [-0.5,1.5]$, "
                rf"$f_{{\rm g}}=\Omega_{{\rm b}}/\Omega_{{\rm m}}={Omega_b / Omega_m:.3f}$"
            ),
        ),
        Patch(
            facecolor="0.75",
            alpha=0.24,
            label=(
                r"gas gNFW, $\alpha \in [-0.5,1.5]$, "
                rf"$f_{{\rm g}}={PAPER_GAS_FRACTION_LOW:.2f}$"
            ),
        ),
    ]
    style_legend = ax.legend(handles=style_handles, fontsize=8.5, loc="upper right")
    ax.add_artist(style_legend)

    host_handles = [
        Line2D([0], [0], color=color, linewidth=1.8, linestyle="-", label=f"host {profile['host_id']}")
        for profile, color in zip(host_profiles, host_colors)
    ]
    reference_handles = [
        Line2D(
            [0],
            [0],
            color=ref["color"],
            linewidth=1.8,
            linestyle=ref["linestyle"],
            label=ref["label"],
        )
        for ref in reference_lines
    ]
    ax.legend(
        handles=host_handles + reference_handles,
        fontsize=7.8,
        ncol=2,
        loc="lower left",
    )
    plt.tight_layout()
    plt.savefig(combined_density_output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined density overplot figure to {combined_density_output_path}")

    if has_temperature_column:
        density_temperature_output_path = (
            output_dir
            / f"all_hosts_density_temperature_profiles_{host_mass_tag}_DD{snapshot:04d}.png"
        )
        fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6), facecolor="white")
        ax_density, ax_temperature = axes

        for profile, color in zip(host_profiles, host_colors):
            total_density_plot = np.where(
                profile["shell_total_density_norm"] > 0,
                profile["shell_total_density_norm"],
                artificial_small,
            )
            total_density_plot = np.maximum(total_density_plot, PAPER_DENSITY_YMIN)
            gas_density_valid = (
                np.isfinite(profile["shell_gas_density_norm"])
                & (profile["shell_gas_density_norm"] > 0.0)
                & np.isfinite(profile["shell_gas_mass_msun"])
                & (profile["shell_gas_mass_msun"] >= 1.0)
            )
            ax_density.plot(
                profile["x_centers"],
                total_density_plot,
                color=color,
                linewidth=1.4,
                alpha=0.8,
            )
            if np.any(gas_density_valid):
                ax_density.plot(
                    profile["x_centers"][gas_density_valid],
                    profile["shell_gas_density_norm"][gas_density_valid],
                    color=color,
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.8,
                )

        ax_density.plot(
            profile_radii,
            total_nfw_profile,
            color="black",
            linestyle="-",
            linewidth=2.8,
            label=rf"NFW total ({CONCENTRATION_LABEL})",
            zorder=5,
        )
        ax_density.fill_between(
            profile_radii,
            gas_band_lower,
            gas_band_upper,
            color="0.45",
            alpha=0.38,
            linewidth=0.0,
            zorder=2,
        )
        ax_density.fill_between(
            profile_radii,
            np.maximum(gas_band_lower * low_fgas_scale, PAPER_DENSITY_YMIN),
            np.maximum(gas_band_upper * low_fgas_scale, PAPER_DENSITY_YMIN),
            color="0.75",
            alpha=0.24,
            linewidth=0.0,
            zorder=1,
        )
        for ref in reference_lines:
            ax_density.axhline(
                ref["y"],
                color=ref["color"],
                linestyle=ref["linestyle"],
                linewidth=2.5,
            )

        ax_density.set_xscale("log")
        ax_density.set_yscale("log")
        ax_density.axvline(1.0, color="black", linestyle=":", linewidth=1.2)
        ax_density.set_xlabel(r"$r/R_{\mathrm{vir}}$", fontsize=13)
        ax_density.set_ylabel(r"$\rho / \rho_{\mathrm{vir}}$", fontsize=13)
        ax_density.set_title("Density profile", fontsize=12)
        ax_density.tick_params(direction="in", which="both", labelsize=11)

        density_style_handles = [
            Line2D([0], [0], color="0.25", linewidth=1.5, linestyle="-", label="host total"),
            Line2D([0], [0], color="0.25", linewidth=1.3, linestyle="--", label="host gas"),
            Line2D([0], [0], color="black", linewidth=2.8, linestyle="-", label=rf"NFW total ({CONCENTRATION_LABEL})"),
            Patch(
                facecolor="0.45",
                alpha=0.38,
                label=rf"gas gNFW, $f_{{\rm g}}=\Omega_{{\rm b}}/\Omega_{{\rm m}}$",
            ),
            Patch(
                facecolor="0.75",
                alpha=0.24,
                label=rf"gas gNFW, $f_{{\rm g}}={PAPER_GAS_FRACTION_LOW:.2f}$",
            ),
        ]
        density_legend = ax_density.legend(
            handles=density_style_handles,
            fontsize=8.8,
            loc="upper right",
            frameon=True,
        )
        ax_density.add_artist(density_legend)
        density_reference_handles = [
            Line2D(
                [0],
                [0],
                color=ref["color"],
                linewidth=1.7,
                linestyle=ref["linestyle"],
                label=ref["label"],
            )
            for ref in reference_lines
        ]
        ax_density.legend(
            handles=density_reference_handles,
            fontsize=8.6,
            loc="lower left",
            frameon=True,
        )

        for profile, color in zip(host_profiles, host_colors):
            host_mvir_msun = profile["host_mass"] / h_Hubble
            host_tvir = Temperature_Virial_analytic(host_mvir_msun, redshift)
            temperature_ratio = profile["shell_gas_temperature_mw"] / host_tvir
            valid_temperature = (
                np.isfinite(temperature_ratio)
                & (temperature_ratio > 0.0)
                & np.isfinite(profile["shell_gas_mass_msun"])
                & (profile["shell_gas_mass_msun"] >= 1.0)
            )
            if not np.any(valid_temperature):
                continue
            ax_temperature.plot(
                profile["x_centers"][valid_temperature],
                temperature_ratio[valid_temperature],
                color=color,
                linewidth=1.2,
                alpha=0.75,
            )

        ax_temperature.set_xscale("log")
        ax_temperature.set_yscale("log")
        ax_temperature.set_ylim(bottom=1.0e-2)
        ax_temperature.axvline(1.0, color="black", linestyle=":", linewidth=1.2)
        ax_temperature.axhline(
            1.0,
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            label=r"$T_{\rm gas}=T_{\rm vir}$",
        )
        ax_temperature.set_xlabel(r"$r/R_{\mathrm{vir}}$", fontsize=13)
        ax_temperature.set_ylabel(r"$T_{\rm gas} / T_{\rm vir}$", fontsize=13)
        ax_temperature.set_title("Temperature profile", fontsize=12)
        ax_temperature.tick_params(direction="in", which="both", labelsize=11)
        ax_temperature.legend(fontsize=9.0, loc="lower right", frameon=True)

        fig.suptitle(
            rf"Pop2Prime host profiles, z={redshift:.2f}, "
            rf"$M_{{\mathrm{{host}}}} \geq 10^{{{np.log10(label_host_mass_min):.1f}}}\,M_\odot/h$",
            fontsize=13,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), w_pad=2.2)
        fig.savefig(density_temperature_output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved combined density-temperature figure to {density_temperature_output_path}")

    plot_specs = [
        (
            "shell_gas_to_total_density_ratio",
            r"$\rho_{\mathrm{gas}} / \rho_{\mathrm{tot}}$",
            output_dir / f"all_hosts_gas_to_total_ratio_{host_mass_tag}_DD{snapshot:04d}.png",
            None,
        ),
    ]
    if has_temperature_column:
        plot_specs.append(
            (
                "shell_gas_temperature_mw",
                r"$T_{\rm gas}\,[\mathrm{K}]$",
                output_dir / f"all_hosts_gas_temperature_mw_{host_mass_tag}_DD{snapshot:04d}.png",
                None,
            )
        )

    for profile_key, ylabel, output_path, ref_lines in plot_specs:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
        artificial_small = 1.0e-10
        for profile, color in zip(host_profiles, host_colors):
            y = profile[profile_key]
            y_plot = np.where(y > 0, y, artificial_small)
            ax.plot(
                profile["x_centers"],
                y_plot,
                color=color,
                linewidth=1.2,
                alpha=0.75,
                label=f"host {profile['host_id']}",
            )

        if ref_lines is not None:
            for ref in ref_lines:
                ax.axhline(
                    ref["y"],
                    color=ref["color"],
                    linestyle=ref["linestyle"],
                    linewidth=1.8,
                    label=ref["label"],
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(1.0, color="black", linestyle=":", linewidth=1.2)
        ax.set_xlabel(r"$x=r/R_{\mathrm{vir}}$", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(
            rf"Pop2Prime host profiles, z={redshift:.2f}, "
            rf"$M_{{\mathrm{{host}}}} \geq 10^{{{np.log10(label_host_mass_min):.1f}}}\,M_\odot/h$",
            fontsize=12,
        )
        ax.tick_params(direction="in", which="both", labelsize=12)
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved overplot figure to {output_path}")

    return loaded


def export_selected_host_gas_profiles(
    snapshot=None,
    target_redshift=TARGET_REDSHIFT,
    host_mass_min=HOST_MASS_MIN,
    output_dir=POP2PRIME_RESULTS_DIR,
    x_min=1.0e-2,
    x_max=2.0,
    num_x_bins=20,
    log_x_bins=True,
    density_unit="g/cm**3",
):
    """
    Save per-host Pop2Prime gas/total density profiles for hosts that contain
    at least one subhalo.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_pop2prime_halo_catalog(snapshot=snapshot, target_redshift=target_redshift)
    hydro_ds = load_pop2prime_snapshot(catalog["snapshot"])
    metadata = find_snapshot_metadata(catalog["snapshot"])
    geometric_parent_ids = get_geometric_parent_ids(catalog)
    host_selection = select_hosts_with_subhalos(
        catalog,
        geometric_parent_ids,
        host_mass_min=host_mass_min,
    )
    selected_host_indices = host_selection["host_indices"]
    if selected_host_indices.size == 0:
        raise RuntimeError("No host halos with subhalos were found for gas-profile export.")


    snapshot = catalog["snapshot"]
    redshift = metadata["z"] if metadata is not None else catalog["redshift"]
    host_mass_tag = format_host_mass_min_tag(host_mass_min)
    host_dir = output_dir / f"pop2prime_host_gas_profiles_DD{snapshot:04d}"
    host_dir.mkdir(parents=True, exist_ok=True)

    host_profiles = []
    for host_index in selected_host_indices:
        profile = compute_single_host_gas_and_total_profiles(
            catalog,
            hydro_ds,
            int(host_index),
            x_min=x_min,
            x_max=x_max,
            num_x_bins=num_x_bins,
            log_x_bins=log_x_bins,
            density_unit=density_unit,
            metadata=metadata,
        )
        host_profiles.append(profile)

    combined_table_path = host_dir / f"all_host_profiles_{host_mass_tag}_DD{snapshot:04d}.txt"
    save_host_profile_collection_table(
        host_profiles,
        combined_table_path,
        host_mass_min=host_mass_min,
    )

    summary_path = host_dir / f"host_summary_{host_mass_tag}_DD{snapshot:04d}.txt"
    with summary_path.open("w") as f:
        f.write(f"# snapshot {snapshot}\n")
        f.write(f"# redshift {redshift:.8e}\n")
        f.write(f"# host_mass_min_Msunh {host_mass_min:.8e}\n")
        f.write("# columns: host_index host_id host_mass_Msunh host_rvir_kpc rho_vir rho_200_crit rho_200_m rho_200_b\n")
        for profile in host_profiles:
            f.write(
                f"{profile['host_index']} {profile['host_id']} "
                f"{profile['host_mass']:.8e} {profile['host_rvir_kpc']:.8e} "
                f"{profile['rho_vir']:.8e} {profile['rho_200_crit']:.8e} "
                f"{profile['rho_200_m']:.8e} {profile['rho_200_b']:.8e}\n"
            )

    print(f"Saved selected-host gas profiles to {host_dir}")
    return {
        "catalog": catalog,
        "host_profiles": host_profiles,
        "selected_host_indices": selected_host_indices,
        "output_dir": host_dir,
    }


def test_gas_density_profile_for_a_single_halo():
    """Compute and save gas/total density profiles for one selected host halo."""
    snapshot = find_nearest_snapshots([TARGET_REDSHIFT])[0]
    catalog = load_pop2prime_halo_catalog(snapshot=snapshot)
    hydro_ds = load_pop2prime_snapshot(snapshot)
    metadata = find_snapshot_metadata(snapshot)
    geometric_parent_ids = get_geometric_parent_ids(catalog)
    host_selection = select_hosts_with_subhalos(
        catalog,
        geometric_parent_ids,
        host_mass_min=HOST_MASS_MIN,
    )
    selected_host_indices = host_selection["host_indices"]
    if selected_host_indices.size == 0:
        raise RuntimeError("No host halos with subhalos were found.")

    host_index = int(selected_host_indices[0])
    profile = compute_single_host_gas_and_total_profiles(
        catalog,
        hydro_ds,
        host_index,
        x_min=1.0e-2,
        x_max=2.0,
        num_x_bins=20,
        log_x_bins=True,
        density_unit="g/cm**3",
        metadata=metadata,
    )

    output_dir = Path(POP2PRIME_RESULTS_DIR) / f"single_host_debug_DD{snapshot:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / f"host_profile_DD{snapshot:04d}_host{profile['host_id']}.txt"
    save_single_host_profile_table(profile, table_path)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    artificial_small = 1.0e-10
    ax.plot(
        profile["x_centers"],
        np.where(profile["shell_total_density_norm"] > 0, profile["shell_total_density_norm"], artificial_small),
        color="tab:blue",
        linewidth=2.0,
        label="total density",
    )
    ax.plot(
        profile["x_centers"],
        np.where(profile["shell_gas_density_norm"] > 0, profile["shell_gas_density_norm"], artificial_small),
        color="tab:green",
        linewidth=2.0,
        label="gas density",
    )
    ax.axhline(
        profile["rho_200_crit_norm"],
        color="tab:red",
        linestyle="--",
        linewidth=1.8,
        label=rf"$200\,\rho_{{\rm crit}}(z={profile['redshift']:.2f}) / \rho_{{\rm vir}}$",
    )
    ax.axhline(
        profile["rho_200_m_norm"],
        color="tab:purple",
        linestyle="--",
        linewidth=1.8,
        label=rf"$200\,\rho_{{\rm m}}(z={profile['redshift']:.2f}) / \rho_{{\rm vir}}$",
    )
    ax.axhline(
        profile["rho_200_b_norm"],
        color="tab:orange",
        linestyle="--",
        linewidth=1.8,
        label=rf"$200\,\rho_{{\rm b}}(z={profile['redshift']:.2f}) / \rho_{{\rm vir}}$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(1.0, color="black", linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$x=r/R_{\mathrm{vir}}$", fontsize=14)
    ax.set_ylabel(r"$\rho / \rho_{\mathrm{vir}}$", fontsize=14)
    ax.set_title(
        rf"Single Pop2Prime host profile, DD{snapshot:04d}, "
        rf"host {profile['host_id']}, $M_{{\rm host}}={profile['host_mass']:.2e}\,M_\odot/h$",
        fontsize=12,
    )
    ax.tick_params(direction="in", which="both", labelsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plot_path = output_dir / f"host_profile_DD{snapshot:04d}_host{profile['host_id']}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved single-host profile plot to {plot_path}")
    return profile



if __name__ == "__main__":

    # test_gas_density_profile_for_a_single_halo()


    # plot_pop2prime_radial_subhalo_profiles_allpsi(
    #     target_redshift=12.0,
    #     show_average=True,
    #     show_individual=True,
    # )
    # plot_pop2prime_radial_subhalo_weighted_profile(
    #     target_redshift=12.0,
    #     psi_min=1.0e-3,
    #     show_average=True,
    #     show_individual=False,
    #     show_percentile=True,
    #     statistic="mass2_dx", #"mass2_dx", "mass_dx", "count_dx"
    # )

    input_path = (
        POP2PRIME_RESULTS_DIR
        / "pop2prime_host_gas_profiles_DD0525"
        / "all_host_profiles_DD0525.txt"
    )
    plot_all_host_profiles_overplot(input_path, host_mass_min=10**5.5)
