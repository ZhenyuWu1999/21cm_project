#This file contains the halo profile data extracted from Latif+ papers using PlotDigitizer.

#PlotDigitizer, 3.3.9, 2025, https://plotdigitizer.com
# @misc{
# url = {https://plotdigitizer.com},
# title = {PlotDigitizer: Version 3.3.9},
# year = {2025}
# }

import numpy as np
from collections import defaultdict
PC_TO_CM = 3.085677581e18
MSUN_G   = 1.98847e33
from physical_constants import Omega_b, Omega_m


# ---- basic container：halo -> quantity -> j21 -> {"r":..., "y":..., "units":...}
Latif2019_data = defaultdict(lambda: defaultdict(dict))


def add_profile(halo, j21, quantity, r, y, r_unit="pc", y_unit=None):
    Latif2019_data[halo][quantity][float(j21)] = {
        "r": np.asarray(r, float),
        "y": np.asarray(y, float),
        "units": {"r": r_unit, "y": y_unit},
    }

def set_scalar(halo, j21, name, value):
    Latif2019_data[halo][float(j21)][name] = float(value)

#Latif & Sadegh 2019 https://doi.org/10.1093/mnras/stz2812

#1. halo 6
#1.0 collapse properties
set_scalar("halo6", 0.1, "z_collapse", 25.4)
set_scalar("halo6", 0.1, "M_collapse_Msun", 7.6e5)
set_scalar("halo6", 1.0, "z_collapse", 24.5)
set_scalar("halo6", 1.0, "M_collapse_Msun", 1.2e6)
set_scalar("halo6", 10.0, "z_collapse", 11.5)
set_scalar("halo6", 10.0, "M_collapse_Msun", 5.7e7)
set_scalar("halo6", 100.0, "z_collapse", 11.0)
set_scalar("halo6", 100.0, "M_collapse_Msun", 6.3e7)
set_scalar("halo6", 1000.0, "z_collapse", 10.6)
set_scalar("halo6", 1000.0, "M_collapse_Msun", 6.7e7)

#1.1 M_enc profile
#(Menc, r) the same for J21=0.1 and 1.0
r01 = [0.00035, 0.00069, 0.00404, 0.0121, 0.10329, 1.0899, 7.85423, 45.46763, 379.38141, 1051.29983]
Menc01 = [0.01992, 0.30115, 16.70967, 69.69781, 320.09786, 2598.68239, 11766.59897, 34768.52715, 329098.2253, 2426517.68699]
add_profile("halo6", 0.1, "M_enc",  r01, Menc01, r_unit="pc", y_unit="Msun")


add_profile("halo6", 1.0, "M_enc",  r01, Menc01, r_unit="pc", y_unit="Msun")

r10 = [0.00056, 0.00164, 0.00452, 0.01914, 0.10766, 0.68431, 3.62174, 22.02055, 91.32361, 407.53673, 1484.40518]
Menc10 = [0.02091, 0.28719, 4.69984, 91.40065, 380.73462, 1755.5782, 9404.70814, 97885.65112, 725264.60559, 2583163.90617, 5338062.6527]
add_profile("halo6", 10.0, "M_enc",  r10, Menc10, r_unit="pc", y_unit="Msun")

r100 = [0.00079, 0.0023, 0.00635, 0.02173, 0.08802, 0.48188, 1.62388, 8.6568, 48.09039, 247.64147, 1305.93423]
Menc100 = [0.04954, 0.86036, 9.60501, 118.22416, 697.33893, 2384.36329, 9064.68081, 36120.78278, 686441.63339, 2760925.10654, 5135942.1426]
add_profile("halo6", 100.0, "M_enc",  r100, Menc100, r_unit="pc", y_unit="Msun")

r1e3 = [0.00045, 0.00102, 0.00299, 0.0093, 0.03864, 0.13931, 1.0459, 9.20296, 82.05128, 493.5039, 1848.25547]
Menc1e3 = [0.05776, 0.57917, 12.05993, 192.25975, 2386.47998, 14985.24882, 78619.98072, 361394.84373, 1494099.07065, 3944326.84803, 7615946.47632]
add_profile("halo6", 1000.0, "M_enc",  r1e3, Menc1e3, r_unit="pc", y_unit="Msun")


#1.2 density profile
r01 = [0.00029, 0.00078, 0.00258, 0.00619, 0.01679, 0.04151, 0.1117, 0.42741, 0.78844, 1.32157, 2.0675, 4.10838, 7.80126, 24.54502, 83.34281, 210.55258, 593.11402, 1172.9761]
rho01 = 1.0e-24 * np.array([6676036375.4725, 5567284274.42254, 2941752221.67314, 1004635218.58635, 84964879.48948, 8549438.59941, 953025.53732, 129167.03329, 34624.77271, 22766.28197, 3072.93953, 508.66154, 64.72145, 5.50033, 1.15906, 0.21366, 0.04147, 0.02405])
add_profile("halo6", 0.1, "density", r01, rho01, r_unit="pc", y_unit="g/cm^3")

r1 = [0.00029, 0.00078, 0.00258, 0.00718, 0.01919, 0.03995, 0.11464, 0.23425, 0.50577, 1.00734, 2.0675, 4.9583, 10.12032, 26.09247, 83.34281, 193.87067, 593.11402, 1172.9761]
rho1 = 1.0e-24*np.array([6676036375.4725, 5567284274.42254, 2941752221.67314, 1095622150.09972, 198226571.17579, 19775489.35551, 2148304.48494, 557040.58712, 35383.43833, 6297.38324, 3072.93953, 512.09462, 50.80204, 8.59027, 1.15906, 0.19954, 0.04147, 0.02405])
add_profile("halo6", 1.0, "density", r1, rho1, r_unit="pc", y_unit="g/cm^3")

r10 = [0.00051, 0.00143, 0.0031, 0.00847, 0.02277, 0.04529, 0.11058, 0.32619, 0.76996, 1.7908, 2.82628, 5.73591, 9.27555, 17.04728, 44.64756, 91.84813, 475.44081, 1579.44347, 25.41432, 1004.94017]
rho10 = 1.0e-24*np.array([1095212795.11404, 1091127647.88932, 1008396540.90551, 341556983.36877, 90031637.12634, 14180280.55537, 1613533.47786, 200077.82296, 38200.62571, 34289.98263, 8164.97848, 13296.88224, 8713.53436, 1155.65551, 147.2641, 5.27685, 0.20822, 0.16344, 138.61337, 0.0823])
add_profile("halo6", 10.0, "density", r10, rho10, r_unit="pc", y_unit="g/cm^3")

r100 = [0.00051, 0.00145, 0.00636, 0.03108, 0.8432, 0.11837, 0.27728, 1.5136, 2.82628, 5.29368, 9.55243, 112.70825, 475.44081, 1579.44347, 39.68564, 889.54949]
rho100 = 1.0e-24*np.array([1097717526.19304, 1096896621.25293, 656821109.84086, 142029271.05545, 63317.97653, 2467373.53578, 834868.35431, 53328.89635, 8091.95476, 2768.91762, 422.03676, 6.66972, 0.20429, 0.16031, 54.93343, 0.06053])
add_profile("halo6", 100.0, "density", r100, rho100, r_unit="pc", y_unit="g/cm^3")

r1e3 = [0.0006, 0.00201, 0.01221, 0.05365, 1.83971, 0.1469, 0.37665, 3.38012, 9.1015, 112.70825, 361.61643, 1360.91106, 39.68564, 820.71265, 1816.7673]
rho1e3 = 1.0e-24*np.array([6842183804.26088, 5095471065.99368, 1926704050.31779, 259565424.94541, 76140.32924, 129931609.87845, 3785100.43162, 45151.52889, 1950.26949, 6.66972, 0.25964, 0.13855, 54.93343, 0.06698, 0.06272])
add_profile("halo6", 1000.0, "density", r1e3, rho1e3, r_unit="pc", y_unit="g/cm^3")

#1.3 fH2 profile
r01 = [0.00029, 0.00076, 0.00154, 0.0036, 0.01057, 0.03824, 0.22964, 1.23441, 4.65921, 11.91119, 21.60988, 26.65703, 29.67299, 45.83186]
fH2_01 = [0.00536, 0.00565, 0.00459, 0.00312, 0.00196, 0.00133, 0.00104, 0.00087, 0.00055, 0.0002, 0.00007, 0.00003, 0.00002, 0.00001]
add_profile("halo6", 0.1, "fH2", r01, fH2_01, r_unit="pc", y_unit=None)

r1 = [0.00042, 0.00083, 0.00168, 0.0036, 0.01057, 0.03824, 0.22964, 0.99363, 3.36506, 6.06289, 8.21556, 10.29322, 13.72205, 15.50683]
fH2_1 = [0.00444, 0.00389, 0.00341, 0.00312, 0.00196, 0.00133, 0.00104, 0.00062, 0.00047, 0.00031, 0.00012, 0.00006, 0.00002, 0.00001]
add_profile("halo6", 1.0, "fH2", r1, fH2_1, r_unit="pc", y_unit=None)

r10 = [0.0005, 0.00193, 0.00542, 0.01464, 0.05303, 0.21793, 1.07813, 2.41321, 5.4761, 9.3343, 20.219, 42.07087, 65.86782, 79.97309]
fH2_10 = [0.00244, 0.0025, 0.00216, 0.00184, 0.00138, 0.00102, 0.0009, 0.00075, 0.00071, 0.0006, 0.0002, 0.00007, 0.00005, 0.00001]
add_profile("halo6", 10.0, "fH2", r10, fH2_10, r_unit="pc", y_unit=None)

r100 = [0.00091, 0.0025, 0.00578, 0.01524, 0.04354, 0.13427, 0.32564, 0.79233, 1.31568, 2.29682, 4.54791, 5.92353, 7.22422, 8.54987]
fH2_100 = [0.00196, 0.00197, 0.0018, 0.00134, 0.00085, 0.00052, 0.0005, 0.00039, 0.0005, 0.00031, 0.00015, 0.00008, 0.00004, 0.00001]
add_profile("halo6", 100.0, "fH2", r100, fH2_100, r_unit="pc", y_unit=None)

r1e3 = [2.14766, 2.2645, 2.88788, 3.95687, 4.9461, 5.10946]
fH2_1e3 = [0.00001, 0.00002, 0.00006, 0.00004, 0.00003, 0.00001]
add_profile("halo6", 1000.0, "fH2", r1e3, fH2_1e3, r_unit="pc", y_unit=None)



#2. halo 1
#2.1 M_enc profile

#2.2 density profile

#2.3 fH2 profile



# ------------------------------------------------------------
#                      Data analysis routines
# ------------------------------------------------------------


def _get_raw_xy(halo: str, j21: float, quantity: str):
    """
    Fetch raw (r, y) from Latif2019_data and make it well-behaved:
    - remove non-finite points
    - sort by radius ascending
    - drop duplicate radii (keep first)
    Returns (r_sorted_unique, y_aligned).
    """
    block = Latif2019_data[halo][quantity][float(j21)]
    r = np.asarray(block["r"], float)
    y = np.asarray(block["y"], float)

    # remove NaN/Inf
    m = np.isfinite(r) & np.isfinite(y)
    r, y = r[m], y[m]

    # sort ascending
    order = np.argsort(r)
    r, y = r[order], y[order]

    # drop duplicates in r
    r_unique, idx = np.unique(r, return_index=True)
    y_unique = y[idx]
    return r_unique, y_unique

def _interp_log_clamped(x, y, x_eval, *, floor=None, ceil=None):
    """
    Log–log interpolation with clamped extrapolation.

    Behavior:
    - If x>0 and y>0, interpolate in log10-space; otherwise fall back to linear.
    - Extrapolation is handled by clamping x_eval to [x_min, x_max] (i.e., use endpoint values).
    - Optionally enforce floor/ceil on the result.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    x_eval = np.asarray(x_eval, float)

    # clamp evaluation points to data range (endpoint extrapolation)
    x_clip = np.clip(x_eval, x[0], x[-1])

    # choose log or linear interpolation
    if np.all(x > 0) and np.all(y > 0) and np.all(x_clip > 0):
        yy = np.interp(np.log10(x_clip), np.log10(x), np.log10(y))
        y_eval = 10.0**yy
    else:
        y_eval = np.interp(x_clip, x, y)

    # apply optional bounds
    if floor is not None:
        y_eval = np.maximum(y_eval, floor)
    if ceil is not None:
        y_eval = np.minimum(y_eval, ceil)
    return y_eval

def _build_r_grid(r_min_pc, r_max_pc, n_samples):
    """Log-spaced radius grid in pc."""
    return np.logspace(np.log10(r_min_pc), np.log10(r_max_pc), int(n_samples))

# ------------------------------
# 1) Mass-weighted H2 fraction
# ------------------------------
def mass_weighted_fH2(halo: str,
                      j21: float,
                      r_min_pc: float = 1e-6,
                      r_max_pc: float = 1e3,
                      n_samples: int = 1000,
                      fH2_floor: float = 1e-8):
    """
    Compute mass-weighted <f_H2> and total gas mass over [r_min_pc, r_max_pc].

        <f_H2> = ∫ 4π r^2 ρ(r) f_H2(r) dr / ∫ 4π r^2 ρ(r) dr

    Assumptions:
    - density is in g/cm^3
    - radius is in pc
    - f_H2 is clamped to at least fH2_floor (useful for strong LW where f_H2 → very small)
    - extrapolation is endpoint-constant

    Returns dict with:
      r_pc, density, fH2, fH2_avg, Mgas_Msun
    """
    # fetch density and fH2 profiles
    r_rho, rho = _get_raw_xy(halo, j21, "density")
    r_fh2, fh2 = _get_raw_xy(halo, j21, "fH2")

    # evaluation grid
    r_pc = _build_r_grid(r_min_pc, r_max_pc, n_samples)

    # interpolate (rho positive; fH2 with a lower floor to avoid log(0))
    rho_eval = _interp_log_clamped(r_rho, rho, r_pc)
    fH2_eval = _interp_log_clamped(r_fh2, np.maximum(fh2, fH2_floor), r_pc, floor=fH2_floor)

    # integrate
    r_cm = r_pc * PC_TO_CM
    dMdr = 4.0 * np.pi * (r_cm**2) * rho_eval            # mass shell derivative
    M_g   = np.trapezoid(dMdr, r_cm)                     # total gas mass [g]
    fH2_avg = np.trapezoid(fH2_eval * dMdr, r_cm) / M_g

    return {
        "r_pc": r_pc,
        "density": rho_eval,
        "fH2": fH2_eval,
        "fH2_avg": fH2_avg,
        "Mgas_Msun": M_g / MSUN_G,
    }

# ------------------------------
# 2) Total gas mass from density
# ------------------------------
def total_gas_mass(halo: str,
                   j21: float,
                   r_min_pc: float = 1e-6,
                   r_max_pc: float = 1e3,
                   n_samples: int = 1000):
    """
    Compute total gas mass by integrating density only:

        M_gas = ∫ 4π r^2 ρ(r) dr   (returned in Msun)

    Extrapolation is endpoint-constant; interpolation prefers log–log when possible.
    """
    r_rho, rho = _get_raw_xy(halo, j21, "density")
    r_pc = _build_r_grid(r_min_pc, r_max_pc, n_samples)
    rho_eval = _interp_log_clamped(r_rho, rho, r_pc)

    r_cm = r_pc * PC_TO_CM
    dMdr = 4.0 * np.pi * (r_cm**2) * rho_eval
    M_g   = np.trapezoid(dMdr, r_cm)
    return M_g / MSUN_G

if __name__ == "__main__":
    halo_name = "halo6"
    j21_val  = 1000.0
    res = mass_weighted_fH2(halo_name, j21_val, r_min_pc=1e-3, r_max_pc=1e3, n_samples=1000, fH2_floor=1e-8)
    Mgas_Msun = res['Mgas_Msun']
    print(f"fH2_avg: {res['fH2_avg']}, Mgas_Msun: {Mgas_Msun:.3e}")
    #print z_collapse and M_collapse
    M_collapse_Msun = Latif2019_data[halo_name][j21_val]['M_collapse_Msun']
    print(f"z_collapse: {Latif2019_data[halo_name][j21_val]['z_collapse']}, M_collapse_Msun: {M_collapse_Msun:.3e}")
    print("Mgas/M_collapse:", Mgas_Msun / M_collapse_Msun)
    print("Omega_b/Omega_m:", Omega_b / Omega_m)