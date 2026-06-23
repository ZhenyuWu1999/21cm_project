
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from scipy.special import hyp2f1
from colossus.cosmology import cosmology
cosmology.setCosmology('planck15')
from colossus.halo import concentration

# Convert RuntimeWarnings to exceptions
warnings.filterwarnings('error', category=RuntimeWarning)


from physical_constants import *
from linear_evolution import D_z
from HaloProperties import Temperature_Virial_analytic

def generalized_NFW_profile(x, rho_s, alpha):
    '''
    parameters:
    x: r/r_s  (concentration: Rvir/r_s = c)
    rho_s: characteristic density
    alpha: slope
    '''
    return rho_s / x**alpha / (1 + x)**(3 - alpha)


def generalized_profile_normalization(concentration, alpha, num_points=4096):
    """
    Return rho_s/rho_vir for a generalized NFW profile with arbitrary alpha.

    The normalization is set so that the mean enclosed density inside Rvir is
    rho_vir, i.e. M(<Rvir) = (4/3) pi Rvir^3 rho_vir.
    """
    x_min = min(1.0e-8, concentration * 1.0e-6)
    x_grid = np.logspace(np.log10(x_min), np.log10(concentration), num_points)
    integrand = x_grid ** (2.0 - alpha) / (1.0 + x_grid) ** (3.0 - alpha)
    integral = np.trapezoid(integrand, x_grid)
    return concentration**3 / (3.0 * integral)

def f_NFW(x):
    return np.log(1+x) - x/(1+x)

def f_core(x):
    return np.log(1+x) -x*(3*x+2)/2/(x+1)**2

def get_profile_corr_for_cooling(profile_type, concentration):
    c = concentration
    if profile_type == 'NFW':
        factor1 = 1.0/3.0*(1 - 1.0/(c+1)**3)
        factor2 = c**3/(3 * f_NFW(c)**2)
    elif profile_type == 'core':
        factor1 = c**3*(c**2+5*c+10)/30/(c+1)**5
        factor2 = c**3/(3*f_core(c)**2)
    else:
        raise ValueError(f"Unknown profile_type: {profile_type}")

    return (factor1 * factor2)


def get_A_alpha_c(concentration, alpha):
    """
    Return A_alpha(c) = integral_0^c x^(2-alpha) / (1+x)^(3-alpha) dx.

    This is the mass-normalization factor for the generalized NFW profile
    rho/rho_vir = c^3 / (3 A_alpha(c)) / [x^alpha (1+x)^(3-alpha)].
    """
    if np.isclose(alpha, 1.0):
        return f_NFW(concentration)
    if np.isclose(alpha, 0.0):
        return f_core(concentration)
    if not (-0.5 < alpha < 1.0):
        raise ValueError("alpha must satisfy -0.5 < alpha <= 1.")

    z = concentration / (1.0 + concentration)
    b = 3.0 - alpha
    return (z**b / b) * hyp2f1(1.0, b, b + 1.0, z)


def get_A_alpha_x(x, alpha):
    """
    Return A_alpha(x) = integral_0^x t^(2-alpha) / (1+t)^(3-alpha) dt.
    """
    x_array = np.asarray(x, dtype=float)
    if np.any(x_array < 0):
        raise ValueError("x must be non-negative.")

    if np.isclose(alpha, 1.0):
        result = f_NFW(x_array)
    elif np.isclose(alpha, 0.0):
        result = f_core(x_array)
    else:
        if not (-0.5 < alpha < 1.0):
            raise ValueError("alpha must satisfy -0.5 < alpha <= 1.")
        z = x_array / (1.0 + x_array)
        b = 3.0 - alpha
        result = (z**b / b) * hyp2f1(1.0, b, b + 1.0, z)

    return result.item() if np.ndim(x) == 0 else result


def get_profile_corr_for_cooling_within_radius(radii_Rvir, concentration, alpha):
    """
    Return the cumulative cooling correction I_profile(<r) for a generalized
    NFW gas-density profile with inner slope alpha.

    Parameters
    ----------
    radii_Rvir : float or array-like
        Radii expressed in units of Rvir.
    concentration : float
        Halo concentration c = Rvir / r_s.
    alpha : float
        Inner slope of the generalized NFW profile.

    Returns
    -------
    float or np.ndarray
        Cumulative cooling correction relative to the uniform-density virial
        baseline. By construction, I_profile(<Rvir) equals the global
        profile correction.
    """
    scalar_input = np.isscalar(radii_Rvir)
    radii_Rvir = np.atleast_1d(np.asarray(radii_Rvir, dtype=float))
    if np.any(radii_Rvir < 0):
        raise ValueError("radii_Rvir must be non-negative.")

    A_alpha = get_A_alpha_c(concentration, alpha)
    radii_clipped = np.clip(radii_Rvir, 0.0, 1.0)
    x = radii_clipped * concentration
    X = x / (1.0 + x)

    prefactor = concentration**3 / (3.0 * A_alpha**2)
    profile_corr = prefactor * (
        X**(3.0 - 2.0 * alpha) / (3.0 - 2.0 * alpha)
        - 2.0 * X**(4.0 - 2.0 * alpha) / (4.0 - 2.0 * alpha)
        + X**(5.0 - 2.0 * alpha) / (5.0 - 2.0 * alpha)
    )

    if scalar_input:
        return float(profile_corr[0])
    return profile_corr


def get_cumulative_cooling_fraction(radii_Rvir, concentration, alpha):
    """
    Return C(<r) / C(<Rvir) for a generalized NFW gas-density profile.
    """
    cumulative_corr = get_profile_corr_for_cooling_within_radius(
        radii_Rvir, concentration, alpha
    )
    total_corr = get_profile_corr_for_cooling_within_radius(
        1.0, concentration, alpha
    )
    return cumulative_corr / total_corr


def get_differential_cooling_fraction(radii_Rvir, concentration, alpha):
    """
    Return d[C(<r) / C(<Rvir)] / d(r/Rvir).

    This is the shell contribution per unit x, where x = r/Rvir.  It integrates
    to unity from x=0 to x=1.
    """
    scalar_input = np.isscalar(radii_Rvir)
    radii_Rvir = np.atleast_1d(np.asarray(radii_Rvir, dtype=float))
    if np.any(radii_Rvir < 0):
        raise ValueError("radii_Rvir must be non-negative.")

    A_alpha = get_A_alpha_c(concentration, alpha)
    total_corr = get_profile_corr_for_cooling_within_radius(1.0, concentration, alpha)

    derivative = np.zeros_like(radii_Rvir, dtype=float)
    in_range = radii_Rvir <= 1.0
    x = radii_Rvir[in_range] * concentration
    derivative[in_range] = (
        concentration**4
        / (3.0 * A_alpha**2)
        * x**(2.0 - 2.0 * alpha)
        / (1.0 + x)**(6.0 - 2.0 * alpha)
        / total_corr
    )

    if scalar_input:
        return float(derivative[0])
    return derivative


def get_cumulative_heating_fraction_modelA(radii_Rvir, concentration, alpha):
    """
    Return H_A(<r) / H_A(<Rvir) for heating Model A, where h_A(r) ∝ rho_g(r).
    """
    scalar_input = np.isscalar(radii_Rvir)
    radii_Rvir = np.atleast_1d(np.asarray(radii_Rvir, dtype=float))
    if np.any(radii_Rvir < 0):
        raise ValueError("radii_Rvir must be non-negative.")

    radii_clipped = np.clip(radii_Rvir, 0.0, 1.0)
    x = radii_clipped * concentration
    heating_fraction = get_A_alpha_x(x, alpha) / get_A_alpha_c(concentration, alpha)

    if scalar_input:
        return float(heating_fraction[0])
    return heating_fraction


def get_differential_heating_fraction_modelA(radii_Rvir, concentration, alpha):
    """
    Return d[H_A(<r) / H_A(<Rvir)] / d(r/Rvir).

    Heating Model A has shell contribution proportional to rho_g(r) r^2 dr.
    The result integrates to unity from x=0 to x=1, with x = r/Rvir.
    """
    scalar_input = np.isscalar(radii_Rvir)
    radii_Rvir = np.atleast_1d(np.asarray(radii_Rvir, dtype=float))
    if np.any(radii_Rvir < 0):
        raise ValueError("radii_Rvir must be non-negative.")

    derivative = np.zeros_like(radii_Rvir, dtype=float)
    in_range = radii_Rvir <= 1.0
    x = radii_Rvir[in_range] * concentration
    derivative[in_range] = (
        concentration
        * x**(2.0 - alpha)
        / (1.0 + x)**(3.0 - alpha)
        / get_A_alpha_c(concentration, alpha)
    )

    if scalar_input:
        return float(derivative[0])
    return derivative


def get_toy_Ksub_top_hat(x, amplitude, x_min=0.1, x_max=1.0):
    """
    Return a toy top-hat K_sub(x) = d/dx sum psi^2.
    """
    x_array = np.asarray(x, dtype=float)
    kernel = np.zeros_like(x_array, dtype=float)
    mask = (x_array >= x_min) & (x_array <= x_max)
    kernel[mask] = amplitude
    return kernel.item() if np.ndim(x) == 0 else kernel


def get_cumulative_heating_ratio_modelB_toy(
    radii_Rvir,
    Mvir,
    redshift,
    concentration,
    alpha,
    H_global,
    mean_molecular_weight=mu,
    f_gas=Omega_b / Omega_m,
    ksub_model='top_hat',
    **kernel_kwargs,
):
    """
    Return H_B(<r) / H_global for a toy Model B subhalo kernel.

    The numerator follows the shell-integrated Model B expression,
        H_B(<r) \propto \int_0^x rho_g(x') K_sub(x') dx',
    while the denominator H_global is a fixed reference from the previous
    global-heating model. Therefore the toy-kernel amplitude is retained in the
    ratio and should not cancel out.
    """
    scalar_input = np.isscalar(radii_Rvir)
    radii_Rvir = np.atleast_1d(np.asarray(radii_Rvir, dtype=float))
    if np.any(radii_Rvir < 0):
        raise ValueError("radii_Rvir must be non-negative.")
    if H_global <= 0:
        raise ValueError("H_global must be positive.")

    radii_clipped = np.clip(radii_Rvir, 0.0, 1.0)
    order = np.argsort(radii_clipped)
    x_eval = radii_clipped[order]

    x_grid = np.logspace(-4, 0, 4000)
    if ksub_model == 'top_hat':
        ksub_grid = get_toy_Ksub_top_hat(x_grid, **kernel_kwargs)
    else:
        raise ValueError(f"Unknown ksub_model: {ksub_model}")

    rho_vir = 200.0 * rho_m0 * (1.0 + redshift) ** 3 * Msun / Mpc**3
    rho_shape = gasdensity_arbitrary_profile(x_grid * concentration, Mvir / h_Hubble, redshift, 'diemer19', alpha)
    rho_g_grid = rho_shape * (f_gas / (Omega_b / Omega_m)) * rho_vir

    Tvir = Temperature_Virial_analytic(Mvir / h_Hubble, redshift, mean_molecular_weight)
    cs = np.sqrt(5.0 / 3.0 * kB * Tvir / (mean_molecular_weight * mp))
    prefactor = 4.0 * np.pi * (G_grav * Mvir * Msun / h_Hubble) ** 2 / cs

    cumulative_integral = np.zeros_like(x_eval)
    integrand = rho_g_grid * ksub_grid
    for i, xmax in enumerate(x_eval):
        mask = x_grid <= xmax
        if np.any(mask):
            cumulative_integral[i] = np.trapezoid(integrand[mask], x_grid[mask])

    ratio_sorted = prefactor * cumulative_integral / H_global
    if scalar_input:
        return float(ratio_sorted[0])

    inverse_order = np.argsort(order)
    return ratio_sorted[inverse_order]



def get_modelC_subhalo_count_dx3_shape(
    radii_Rvir,
    concentration,
    use_jb17_correction=False,
    jb17_eta=2.0,
    jb17_mu=4.0,
    radial_bias_model=None,
    han16_gamma=1.33,
):
    """
    Return the Model C subhalo number-density shape P(x) = dN/dx^3.

    Here x = r/Rvir.  The baseline shape is NFW-like.  Optional radial-bias
    factors are JB17, 2^mu x^eta / (1+x)^mu, or Han16, x^gamma.
    """
    scalar_input = np.isscalar(radii_Rvir)
    x_values = np.atleast_1d(np.asarray(radii_Rvir, dtype=float))
    if np.any(x_values < 0):
        raise ValueError("radii_Rvir must be non-negative.")

    if radial_bias_model is None:
        radial_bias_model = 'jb17' if use_jb17_correction else 'nfw'
    radial_bias_model = radial_bias_model.lower()

    profile = np.zeros_like(x_values, dtype=float)
    positive = x_values > 0.0
    cx = concentration * x_values[positive]
    profile[positive] = 1.0 / (cx * (1.0 + cx) ** 2)

    if radial_bias_model in {'nfw', 'none'}:
        pass
    elif radial_bias_model == 'jb17':
        profile[positive] *= (
            (2.0 ** jb17_mu)
            * x_values[positive] ** jb17_eta
            / (1.0 + x_values[positive]) ** jb17_mu
        )
    elif radial_bias_model == 'han16':
        profile[positive] *= x_values[positive] ** han16_gamma
    else:
        raise ValueError(f"Unknown radial_bias_model: {radial_bias_model}")

    if scalar_input:
        return float(profile[0])
    return profile


def get_modelC_subhalo_radial_pdf(
    radii_Rvir,
    concentration,
    x_max=1.0,
    use_jb17_correction=False,
    num_points=4096,
    radial_bias_model=None,
    han16_gamma=1.33,
):
    """
    Return the normalized Model C radial PDF u(x) per unit x.

    P(x) is normalized over [0, x_max] using
        u(x) = 3 x^2 P(x) / integral_0^xmax 3 x'^2 P(x') dx'.
    """
    if x_max <= 0:
        raise ValueError("x_max must be positive.")

    scalar_input = np.isscalar(radii_Rvir)
    x_values = np.atleast_1d(np.asarray(radii_Rvir, dtype=float))
    if np.any(x_values < 0):
        raise ValueError("radii_Rvir must be non-negative.")

    x_floor = max(1.0e-8, x_max * 1.0e-6)
    x_grid = np.concatenate(([0.0], np.geomspace(x_floor, x_max, num_points)))
    p_grid = get_modelC_subhalo_count_dx3_shape(
        x_grid,
        concentration,
        use_jb17_correction=use_jb17_correction,
        radial_bias_model=radial_bias_model,
        han16_gamma=han16_gamma,
    )
    pdf_grid = 3.0 * x_grid**2 * p_grid
    normalization = np.trapezoid(pdf_grid, x_grid)
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("Model C radial PDF normalization failed.")

    pdf_values = np.zeros_like(x_values, dtype=float)
    in_range = x_values <= x_max
    p_values = get_modelC_subhalo_count_dx3_shape(
        x_values[in_range],
        concentration,
        use_jb17_correction=use_jb17_correction,
        radial_bias_model=radial_bias_model,
        han16_gamma=han16_gamma,
    )
    pdf_values[in_range] = 3.0 * x_values[in_range] ** 2 * p_values / normalization

    if scalar_input:
        return float(pdf_values[0])
    return pdf_values


def get_gas_density_ratio_to_virial(radii_Rvir, concentration, alpha):
    """
    Return rho_g(x) / rho_g,vir for the generalized NFW gas profile.

    The denominator is the mean gas density inside Rvir used by the global
    heating model.  The gas fraction cancels in this ratio.
    """
    scalar_input = np.isscalar(radii_Rvir)
    radii_Rvir = np.atleast_1d(np.asarray(radii_Rvir, dtype=float))
    if np.any(radii_Rvir < 0):
        raise ValueError("radii_Rvir must be non-negative.")

    x = radii_Rvir * concentration
    rho_s_over_rho_vir = concentration**3 / (3.0 * get_A_alpha_c(concentration, alpha))
    density_ratio = np.zeros_like(x, dtype=float)
    positive = x > 0.0
    density_ratio[positive] = generalized_NFW_profile(
        x[positive],
        rho_s_over_rho_vir,
        alpha,
    )

    if scalar_input:
        return float(density_ratio[0])
    return density_ratio


def get_cumulative_heating_ratio_modelC(
    radii_Rvir,
    concentration,
    alpha,
    x_max=1.0,
    use_jb17_correction=False,
    num_points=4096,
    radial_bias_model=None,
    han16_gamma=1.33,
):
    """
    Return H_C(<r) / H_global for Heating Model C.

    The curve is not renormalized to make H_C(<Rvir) equal H_global.  It keeps
    the extra correction from the gas-density profile and the subhalo radial
    PDF normalized over [0, x_max].
    """
    if x_max <= 0:
        raise ValueError("x_max must be positive.")

    scalar_input = np.isscalar(radii_Rvir)
    radii_Rvir = np.atleast_1d(np.asarray(radii_Rvir, dtype=float))
    if np.any(radii_Rvir < 0):
        raise ValueError("radii_Rvir must be non-negative.")

    radii_clipped = np.clip(radii_Rvir, 0.0, x_max)
    order = np.argsort(radii_clipped)
    x_eval = radii_clipped[order]

    x_floor = max(1.0e-8, x_max * 1.0e-6)
    x_grid = np.concatenate(([0.0], np.geomspace(x_floor, x_max, num_points)))
    u_grid = get_modelC_subhalo_radial_pdf(
        x_grid,
        concentration,
        x_max=x_max,
        use_jb17_correction=use_jb17_correction,
        num_points=num_points,
        radial_bias_model=radial_bias_model,
        han16_gamma=han16_gamma,
    )
    density_ratio_grid = get_gas_density_ratio_to_virial(
        x_grid,
        concentration,
        alpha,
    )
    integrand = density_ratio_grid * u_grid
    integrand[0] = 0.0

    cumulative_grid = np.zeros_like(x_grid)
    cumulative_grid[1:] = np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(x_grid)
    )
    ratio_sorted = np.interp(x_eval, x_grid, cumulative_grid)

    if scalar_input:
        return float(ratio_sorted[0])

    inverse_order = np.argsort(order)
    return ratio_sorted[inverse_order]

def get_concentration(M_in_Msun, z, model_name):
    '''
    parameters:
    M: halo mass in Msun
    z: redshift
    model: concentration model, see colossus tutorial, e.g. 'bullock01', 'ludlow16', 'child18','diemer19','ishiyama21'
    '''
    if model_name != 'bullock01_Dekel':
        M = M_in_Msun * h_Hubble # in Msun/h
        c = concentration.concentration(M, '200c', z, model = model_name, range_return = False)
    elif model_name == 'bullock01_Dekel':
        M13 = M_in_Msun/1e13
        c = 9.0*M13**(-0.15)/(1+z)

    return c

    
def density_NFW_profile(x, M, z, concentration_model):
    '''
    parameters:
    x: r/r_s  (concentration: Rvir/r_s = c)
    M: halo mass in Msun
    z: redshift
    concentration_model: concentration model name for colossus
    '''
    #assume 200 times the cosmological background matter density, unit kg/m^3
    rho_vir = 200 * rho_m0*(1+z)**3 *Msun/Mpc**3
    concentration = get_concentration(M, z, concentration_model)
    rho_s = concentration**3 / f_NFW(concentration) / 3.0 #in unit of rho_vir
    alpha = 1.0
    return generalized_NFW_profile(x, rho_s, alpha) #in unit of rho_vir

def gasdensity_NFW_profile(x, M, z, concentration_model):
    '''
    parameters:
    x: r/r_s  (concentration: Rvir/r_s = c)
    M: halo mass in Msun
    z: redshift
    concentration_model: concentration model name for colossus
    '''
    concentration = get_concentration(M, z, concentration_model)
    rho_s = concentration**3 / f_NFW(concentration) / 3.0
    alpha = 1.0
    f_gas = Omega_b/Omega_m
    return f_gas * generalized_NFW_profile(x, rho_s, alpha) #in unit of rho_vir

def gasdensity_core_profile(x, M, z, concentration_model):
    '''
    parameters:
    x: r/r_s  (concentration: Rvir/r_s = c)
    M: halo mass in Msun
    z: redshift
    concentration_model: concentration model name for colossus
    '''
    concentration = get_concentration(M, z, concentration_model)
    rho_s = concentration**3 / f_core(concentration) / 3.0  
    alpha = 0.0
    f_gas = Omega_b/Omega_m
    return f_gas * generalized_NFW_profile(x, rho_s, alpha) #in unit of rho_vir


def density_arbitrary_profile(x, M, z, concentration_model, alpha):
    '''
    parameters:
    x: r/r_s  (concentration: Rvir/r_s = c)
    M: halo mass in Msun
    z: redshift
    concentration_model: concentration model name for colossus
    alpha: inner slope of the generalized NFW profile
    '''
    concentration = get_concentration(M, z, concentration_model)
    rho_s = generalized_profile_normalization(concentration, alpha)
    return generalized_NFW_profile(x, rho_s, alpha) #in unit of rho_vir


def gasdensity_arbitrary_profile(x, M, z, concentration_model, alpha):
    '''
    parameters:
    x: r/r_s  (concentration: Rvir/r_s = c)
    M: halo mass in Msun
    z: redshift
    concentration_model: concentration model name for colossus
    alpha: inner slope of the generalized NFW profile
    '''
    f_gas = Omega_b/Omega_m
    return f_gas * density_arbitrary_profile(x, M, z, concentration_model, alpha)

def Velvir_NFW_profile(x, M, z, concentration_model):  #in virial velocity unit
    '''
    parameters:
    x: r/r_s  (concentration: Rvir/r_s = c)
    M: halo mass in Msun
    z: redshift
    concentration_model: concentration model name for colossus
    '''
    concentration = get_concentration(M, z, concentration_model)
    r_Rvir = x / concentration
    return np.sqrt(f_NFW(x)/f_NFW(concentration)/r_Rvir) #in virial velocity unit


def compare_concentration_model():
    output_dir = '/home/zwu/21cm_project/unified_model/Analytic_results/halo_profile'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for model_name in concentration.models:
        print(model_name)

    M = 10**np.arange(5.0, 15.4, 0.1)
    z_list = [0, 3, 6, 8, 10, 12, 15]
    models_to_plot = ['bullock01','ludlow16', 'child18','diemer19','ishiyama21']
    for z in z_list:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        for model_name in models_to_plot:
            c, mask = concentration.concentration(M, '200c', z, model = model_name, range_return = True)
            plt.plot(M[mask], c[mask], label = model_name.replace('_', '\\_'))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('M200c(Msun/h)',fontsize=14)
        plt.ylabel('Concentration',fontsize=14)
        plt.legend()
        plt.tight_layout()
        filename = os.path.join(output_dir, f'concentration_model_z{z}.png')
        plt.savefig(filename, dpi=300)

def plot_density_and_velocity_profile(concentration_model):
    
    output_dir = '/home/zwu/21cm_project/unified_model/Analytic_results/halo_profile'
    #plot total mass density NFW profile, gas NFW profile, gas core profile
    filename = os.path.join(output_dir, f'density_NFW_profile_{concentration_model}.png')
    z_list = [12, 6, 0]
    M_list = [1.0e7, 1.0e10, 1.0e13]
    
    for iz, z in enumerate(z_list):
        fig, ax = plt.subplots(1, len(M_list), figsize=(24, 6), facecolor='white')
        for iM, M in enumerate(M_list):
            concentration = get_concentration(M, z, concentration_model)
            r_Rvir_list = np.logspace(-2, 0, 100)
            x_list = r_Rvir_list * concentration

            rho_tot = density_NFW_profile(x_list, M, z, concentration_model)
            rho_gas_NFW = gasdensity_NFW_profile(x_list, M, z, concentration_model)
            rho_gas_core = gasdensity_core_profile(x_list, M, z, concentration_model)
            ax[iM].plot(r_Rvir_list, rho_tot, label='total', color='blue', linestyle='--')
            ax[iM].plot(r_Rvir_list, rho_gas_NFW, label='gas NFW', color='red', linestyle='-')
            ax[iM].plot(r_Rvir_list, rho_gas_core, label='gas core', color='green', linestyle='-')
            ax[iM].set_xscale('log')
            ax[iM].set_yscale('log')
            ax[iM].set_xlabel('r/Rvir', fontsize=14)
            ax[iM].set_ylabel('rho/rho_vir', fontsize=14)
            ax[iM].set_title(f'M={M:.1e}Msun, z={z}', fontsize=14)
            ax[iM].legend()
        plt.savefig(filename.replace('.png', f'_z{z}.png'))
    
    #plot velocity profile
    filename = os.path.join(output_dir, f'velocity_NFW_profile_{concentration_model}.png')
    z_list = [12, 6, 0]
    M_list = [1.0e7, 1.0e10, 1.0e13]

    for iz, z in enumerate(z_list):
        fig, ax = plt.subplots(1, len(M_list), figsize=(24, 6), facecolor='white')
        for iM, M in enumerate(M_list):
            concentration = get_concentration(M, z, concentration_model)
            r_Rvir_list = np.logspace(-2, 0, 100)
            x_list = r_Rvir_list * concentration

            Velvir = Velvir_NFW_profile(x_list, M, z, concentration_model)
            ax[iM].plot(r_Rvir_list, Velvir, label='velocity', color='blue', linestyle='--')
            ax[iM].set_xscale('log')
            ax[iM].set_yscale('log')
            ax[iM].set_xlabel('r/Rvir', fontsize=14)
            ax[iM].set_ylabel('velocity/virial velocity', fontsize=14)
            ax[iM].set_title(f'M={M:.1e}Msun, z={z}', fontsize=14)
            ax[iM].legend()
        plt.savefig(filename.replace('.png', f'_z{z}.png'))




if __name__ == "__main__":


    compare_concentration_model()
    # plot_density_and_velocity_profile('bullock01')
    # plot_density_and_velocity_profile('ludlow16')
    
