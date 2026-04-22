from matplotlib import pyplot as plt
import numpy as np
import scipy
import yt
from pathlib import Path
from HaloMassFunction import SHMF_BestFit_dN_dlgx


def summarize_halo_fields(ds, ad, sample_size=5):
    halo_fields = sorted([field for field in ds.derived_field_list if field[0] == "halos"])
    print(f"Number of halo fields: {len(halo_fields)}")
    print("Halo fields:")
    print(halo_fields)
    print()

    sample_fields = [
        ("halos", "particle_identifier"),
        ("halos", "particle_mass"),
        ("halos", "virial_radius"),
        ("halos", "particle_position_x"),
        ("halos", "particle_velocity_x"),
    ]
    print(f"Sample values for first {sample_size} halos:")
    for field in sample_fields:
        values = ad[field][:sample_size]
        print(f"{field}: {values}")
    print()


def get_snapshot_redshifts():
    """
    Read Rockstar out_*.list headers and return snapshot, scale factor, and redshift.

    Parameters
    ----------
    Returns
    -------
    list[dict]
        Each item has keys: snapshot, a, z, filename.
    """
    list_dir = Path("/home/zwu/21cm_project/pop2prime_data/rockstar_halos")
    results = []

    def snapshot_number(path):
        try:
            return int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            return float("inf")

    file_paths = sorted(list_dir.glob("out_*.list"), key=snapshot_number)
    total_files = len(file_paths)

    for i, path in enumerate(file_paths, start=1):
        print(f"Reading snapshot header {i}/{total_files}: {path.name}")
        snapshot_str = path.stem.split("_")[1]
        try:
            snapshot = int(snapshot_str)
        except ValueError:
            continue

        a_value = None
        with path.open("r") as f:
            for _ in range(10):
                line = f.readline()
                if not line:
                    break
                if line.startswith("#a ="):
                    a_value = float(line.split("=")[1].strip())
                    break

        if a_value is None:
            continue

        results.append(
            {
                "snapshot": snapshot,
                "a": a_value,
                "z": 1.0 / a_value - 1.0,
                "filename": str(path),
            }
        )

    return results


def write_snapshot_redshifts():
    """
    Write snapshot, scale factor, and redshift from Rockstar out_*.list headers.

    Parameters
    ----------
    Reads out_*.list files from the Rockstar halo directory and writes a text
    file containing snapshot, scale factor, and redshift.
    """
    redshift_data = get_snapshot_redshifts()
    output_path = Path("/home/zwu/21cm_project/unified_model/Pop2prime_results/snapshot_redshifts.txt")

    with output_path.open("w") as f:
        f.write("# snapshot a z\n")
        for entry in redshift_data:
            f.write(f"{entry['snapshot']} {entry['a']:.6f} {entry['z']:.6f}\n")

    print(f"Saved snapshot redshift table to {output_path}")


def load_snapshot_redshifts():
    """
    Load snapshot, scale factor, and redshift from the saved text table.
    """
    input_path = Path("/home/zwu/21cm_project/unified_model/Pop2prime_results/snapshot_redshifts.txt")
    results = []

    with input_path.open("r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            snapshot_str, a_str, z_str = line.split()[:3]
            results.append(
                {
                    "snapshot": int(snapshot_str),
                    "a": float(a_str),
                    "z": float(z_str),
                    "filename": "",
                }
            )

    return results


def find_nearest_snapshots(target_redshifts):
    """
    For each target redshift, return the nearest Rockstar snapshot.

    Parameters
    ----------
    target_redshifts : iterable of float
        Target redshifts to match.

    Returns
    -------
    list[int]
        Snapshot numbers nearest to the requested redshifts.
    """
    redshift_data = load_snapshot_redshifts()
    snapshots = []

    for target_z in target_redshifts:
        best_match = min(redshift_data, key=lambda entry: abs(entry["z"] - target_z))
        snapshots.append(best_match["snapshot"])

    return snapshots


def create_parent_dict(data_source, ptype="halos"):
    """
    Create a dictionary of halo parents to allow for filtering of subhalos.

    For a pair of halos whose distance is smaller than the radius of at least
    one of the halos, the parent is defined as the halo with the larger radius.
    Parent halos (halos with no parents of their own) have parent index values of -1.
    """
    pos = np.rollaxis(
        np.array(
            [
                data_source[ptype, "particle_position_x"].in_units("Mpc"),
                data_source[ptype, "particle_position_y"].in_units("Mpc"),
                data_source[ptype, "particle_position_z"].in_units("Mpc"),
            ]
        ),
        1,
    )
    rad = data_source[ptype, "virial_radius"].in_units("Mpc").to_ndarray()
    ids = data_source[ptype, "particle_identifier"].to_ndarray().astype("int")
    parents = -1 * np.ones_like(ids, dtype="int")
    boxsize = data_source.ds.domain_width.in_units("Mpc")
    my_tree = scipy.spatial.cKDTree(pos, boxsize=boxsize)

    for i in range(ids.size):
        neighbors = np.array(my_tree.query_ball_point(pos[i], rad[i], p=2))
        if neighbors.size > 1:
            parents[neighbors] = ids[neighbors[np.argmax(rad[neighbors])]]

    parents[ids == parents] = -1
    parent_dict = dict(zip(ids, parents))
    return parent_dict


def build_host_subhalo_catalog(halo_ids, parent_ids, masses, host_mass_min=None):
    """
    Build a host-centric catalog of subhalos for later SHMF analysis.

    Returns
    -------
    list[dict]
        One entry per host halo. If host_mass_min is set, only hosts above the
        threshold are included.
    """
    if host_mass_min is None:
        host_indices = np.where(parent_ids == -1)[0]
    else:
        host_indices = np.where((parent_ids == -1) & (masses >= host_mass_min))[0]
    host_catalog = []

    for host_index in host_indices:
        host_id = halo_ids[host_index]
        host_mass = masses[host_index]
        subhalo_indices = np.where(parent_ids == host_id)[0]
        subhalo_masses = masses[subhalo_indices]
        psi = subhalo_masses / host_mass if subhalo_indices.size > 0 else np.array([])

        host_catalog.append(
            {
                "host_index": host_index,
                "host_id": host_id,
                "host_mass": host_mass,
                "subhalo_indices": subhalo_indices,
                "subhalo_ids": halo_ids[subhalo_indices],
                "subhalo_masses": subhalo_masses,
                "psi": psi,
                "n_subhalos": subhalo_indices.size,
            }
        )

    return host_catalog


# NOTE: compute_orbital_angular_momentum() is defined but not yet enabled.
# It requires full halo catalog positions and velocities, which are available
# from Rockstar but not yet passed through build_host_subhalo_catalog().
# To be activated once the data pipeline is confirmed with the collaborator.
def compute_orbital_angular_momentum(
    halo_ids, parent_ids, masses, positions, velocities, boxsize
):
    """
    Compute the orbital angular momentum of each subhalo relative to its host.

    Parameters
    ----------
    halo_ids : np.ndarray, shape (N,)
        Halo identifiers.
    parent_ids : np.ndarray, shape (N,)
        Parent halo ID for each halo; -1 for host halos.
    masses : np.ndarray, shape (N,), units Msun/h
        Halo masses.
    positions : np.ndarray, shape (N, 3), units Mpc/h (comoving)
        Halo center positions.
    velocities : np.ndarray, shape (N, 3), units km/s (peculiar)
        Halo center velocities.
    boxsize : float, units Mpc/h
        Simulation box side length for periodic boundary correction.

    Returns
    -------
    L_orb : np.ndarray, shape (N, 3)
        Orbital angular momentum vector for each halo in units of
        (Msun/h) * (Mpc/h) * (km/s).  Host halos are assigned L_orb = 0.
    """
    id_to_index = {hid: i for i, hid in enumerate(halo_ids)}
    L_orb = np.zeros((len(halo_ids), 3))

    subhalo_mask = parent_ids != -1
    subhalo_indices = np.where(subhalo_mask)[0]

    for i in subhalo_indices:
        parent_id = parent_ids[i]
        if parent_id not in id_to_index:
            continue
        j = id_to_index[parent_id]

        # Relative position with periodic boundary correction
        dr = positions[i] - positions[j]
        dr -= boxsize * np.round(dr / boxsize)

        # Relative velocity
        dv = velocities[i] - velocities[j]

        # L_orb = m_sub * (dr x dv)
        L_orb[i] = masses[i] * np.cross(dr, dv)

    return L_orb


def print_host_catalog_summary(host_catalog, max_hosts=10):
    n_hosts = len(host_catalog)
    n_hosts_with_subhalos = sum(entry["n_subhalos"] > 0 for entry in host_catalog)
    print(f"Number of host halos: {n_hosts}")
    print(f"Number of host halos with at least one subhalo: {n_hosts_with_subhalos}")

    host_masses = np.array([entry["host_mass"] for entry in host_catalog])
    positive_host_masses = host_masses[host_masses > 0]
    if positive_host_masses.size > 0:
        log_mass_bins = np.linspace(
            np.log10(positive_host_masses.min()),
            np.log10(positive_host_masses.max()),
            6,
        )
        counts, _ = np.histogram(np.log10(positive_host_masses), bins=log_mass_bins)
        print("Host halo counts in 5 equal-width log10(M_host/[Msun/h]) bins:")
        for i in range(len(counts)):
            print(
                f"[{log_mass_bins[i]:.3f}, {log_mass_bins[i + 1]:.3f}): {counts[i]}"
            )

    sorted_catalog = sorted(host_catalog, key=lambda entry: entry["host_mass"], reverse=True)
    print(f"Top {min(max_hosts, len(sorted_catalog))} hosts by mass:")
    for entry in sorted_catalog[:max_hosts]:
        print(
            f"host_id={entry['host_id']} "
            f"M_host={entry['host_mass']:.3e} Msun/h "
            f"N_sub={entry['n_subhalos']}"
        )


def plot_shmf_for_host_mass_range(
    host_catalog,
    logM_min,
    logM_max,
    output_path,
    redshift,
    num_ratio_bins=50,
    log_psi_min=-4,
    log_psi_max=0,
    plot_individual_hosts=False,
):
    """
    Plot the average SHMF for hosts in a selected log10(M_host/[Msun/h]) range.
    """
    particle_mass_msun_h = 1.07423 # Msun/h, mass resolution from the Rockstar halo catalog metadata
    selected_hosts = [
        entry
        for entry in host_catalog
        if logM_min <= np.log10(entry["host_mass"]) < logM_max
    ]
    num_hosts = len(selected_hosts)
    print(
        f"Selected {num_hosts} host halos in "
        f"{logM_min:.2f} <= log10(M_host/[Msun/h]) < {logM_max:.2f}"
    )

    if num_hosts == 0:
        raise RuntimeError("No host halos found in the requested host-mass range.")

    bins = np.linspace(log_psi_min, log_psi_max, num_ratio_bins + 1)
    log_bin_width = bins[1] - bins[0]
    sub_host_mratio_matrix = np.zeros((num_hosts, num_ratio_bins))

    for i, entry in enumerate(selected_hosts):
        psi = entry["psi"]
        positive_psi = psi[psi > 0]
        if positive_psi.size == 0:
            continue
        sub_host_mratio_matrix[i, :], _ = np.histogram(np.log10(positive_psi), bins=bins)

    number_density = np.sum(sub_host_mratio_matrix, axis=0) / log_bin_width / num_hosts
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    plt.figure(facecolor="white")
    if plot_individual_hosts:
        for i in range(num_hosts):
            host_number_density = sub_host_mratio_matrix[i, :] / log_bin_width
            plt.step(
                bin_centers,
                host_number_density,
                where="mid",
                color="gray",
                alpha=0.35,
                linewidth=1.0,
            )

    plt.step(
        bin_centers,
        number_density,
        where="mid",
        color="C0",
        linewidth=2.5,
        label=f"Average over {num_hosts} host halos",
    )
    plt.yscale("log")
    resolution_psi = 50.0 * particle_mass_msun_h / (10**logM_min)
    plt.axvline(
        np.log10(resolution_psi),
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="50-particle resolution",
    )
    shmf_bestfit_z = SHMF_BestFit_dN_dlgx(bin_centers, redshift=redshift, SHMF_model="BestFit_z")
    shmf_bestfit_z0 = SHMF_BestFit_dN_dlgx(bin_centers, redshift=0.0, SHMF_model="BestFit_z")
    plt.plot(
        bin_centers,
        shmf_bestfit_z,
        color="tab:red",
        linestyle="-",
        linewidth=2.0,
        label=fr"TNG BestFit, z={redshift:.0f}",
    )
    plt.plot(
        bin_centers,
        shmf_bestfit_z0,
        color="tab:green",
        linestyle="--",
        linewidth=2.0,
        label="TNG BestFit, z=0",
    )
    plt.xlabel(r'$\lg$($\psi$) = $\lg$(m/M)',fontsize=14)
    plt.ylabel(r'dN/d$\lg(\psi)$',fontsize=14)
    plt.ylim(bottom=1e-3, top=1e4)
    plt.title(
        rf"Pop2Prime SHMF, ${logM_min:.1f} \leq \log_{{10}}(M/[M_\odot/h]) < {logM_max:.1f}$"
    )
    plt.legend()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved SHMF plot to {output_path}")


if __name__ == "__main__":

    # write_snapshot_redshifts()
    target_z = 12.0
    redshift_data = load_snapshot_redshifts()
    target_snapshot = find_nearest_snapshots([target_z])[0]
    matched_entry = next(entry for entry in redshift_data if entry["snapshot"] == target_snapshot)
    halo_path = f"/home/zwu/21cm_project/pop2prime_data/rockstar_halos/halos_DD{target_snapshot:04d}.0.bin"
    print(
        f"Using nearest snapshot for target z={target_z}: "
        f"DD{target_snapshot:04d} with a={matched_entry['a']:.6f}, z={matched_entry['z']:.6f}"
    )

    ds = yt.load(halo_path)
    print(f"yt-loaded dataset redshift: z={ds.current_redshift:.6f}")

    ad = ds.all_data()
    pdict = create_parent_dict(ad)
    halo_ids = ad["halos", "particle_identifier"].d.astype(int)
    parent_ids = np.array([pdict[halo_id] for halo_id in halo_ids])
    masses = ad["halos", "particle_mass"].to("Msun/h").to_ndarray()
    host_catalog = build_host_subhalo_catalog(halo_ids, parent_ids, masses)
    # print_host_catalog_summary(host_catalog)
    output_dir = Path("/home/zwu/21cm_project/unified_model/Pop2prime_results")
    host_logmasses = np.array([np.log10(entry["host_mass"]) for entry in host_catalog if entry["host_mass"] > 0])
    max_logM = np.max(host_logmasses)
    min_logM = 5.0
    shmf_filename = output_dir / f"shmf_hosts_logM_{min_logM:.1f}_to_max_z{ds.current_redshift:.2f}.png"
    plot_shmf_for_host_mass_range(
        host_catalog,
        logM_min=min_logM,
        logM_max=max_logM + 1e-6,
        output_path=shmf_filename,
        redshift=ds.current_redshift,
        plot_individual_hosts=True,
    )
