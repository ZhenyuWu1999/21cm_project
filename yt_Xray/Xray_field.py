import os
import numpy as np
from read_CloudyData import print_attrs, display_hdf5_contents, read_hdf5_data
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from yt.config import ytcfg
from yt.fields.derived_field import DerivedField
from yt.funcs import mylog, only_on_root, parse_h5_attr
from yt.units.yt_array import YTArray, YTQuantity
from yt.utilities.cosmology import Cosmology
from yt.utilities.exceptions import YTException, YTFieldNotFound
from yt.utilities.linear_interpolators import (
    BilinearFieldInterpolator,
    UnilinearFieldInterpolator,
)
from yt.utilities.on_demand_imports import _h5py as h5py

data_version = {"cloudy": 2, "apec": 3}

data_url = "http://yt-project.org/data"


def _get_data_file(table_type, data_dir=None):
    data_file = "%s_emissivity_v%d.h5" % (table_type, data_version[table_type])
    if data_dir is None:
        supp_data_dir = ytcfg.get("yt", "supp_data_dir")
        data_dir = supp_data_dir if os.path.exists(supp_data_dir) else "."
    data_path = os.path.join(data_dir, data_file)
    if not os.path.exists(data_path):
        msg = f"Failed to find emissivity data file {data_file}! Please download from {data_url}"
        mylog.error(msg)
        raise OSError(msg)
    return data_path


class EnergyBoundsException(YTException):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __str__(self):
        return f"Energy bounds are {self.lower:e} to {self.upper:e} keV."


class ObsoleteDataException(YTException):
    def __init__(self, table_type):
        data_file = "%s_emissivity_v%d.h5" % (table_type, data_version[table_type])
        self.msg = "X-ray emissivity data is out of date.\n"
        self.msg += f"Download the latest data from {data_url}/{data_file}."

    def __str__(self):
        return self.msg


class XrayEmissivityIntegrator:
    r"""Class for making X-ray emissivity fields. Uses hdf5 data tables
    generated from Cloudy and AtomDB/APEC.

    Initialize an XrayEmissivityIntegrator object.

    Parameters
    ----------
    table_type : string
        The type of data to use when computing the emissivity values. If "cloudy",
        a file called "cloudy_emissivity.h5" is used, for photoionized
        plasmas. If, "apec", a file called "apec_emissivity.h5" is used for
        collisionally ionized plasmas. These files contain emissivity tables
        for primordial elements and for metals at solar metallicity for the
        energy range 0.1 to 100 keV.
    redshift : float, optional
        The cosmological redshift of the source of the field. Default: 0.0.
    data_dir : string, optional
        The location to look for the data table in. If not supplied, the file
        will be looked for in the location of the YT_DEST environment variable
        or in the current working directory.
    use_metals : boolean, optional
        If set to True, the emissivity will include contributions from metals.
        Default: True
    """

    def __init__(self, table_type, redshift=0.0, data_dir=None, use_metals=True):
        filename = _get_data_file(table_type, data_dir=data_dir)
        in_file = h5py.File(filename, mode="r")
        print("Reading data from %s" % filename)
        
        self.log_T = in_file["log_T"][:]
        self.emissivity_primordial = in_file["emissivity_primordial"][:]
        if "log_nH" in in_file:
            self.log_nH = in_file["log_nH"][:]
        if use_metals:
            self.emissivity_metals = in_file["emissivity_metals"][:]
        self.ebin = YTArray(in_file["E"], "keV")
        in_file.close()
        self.dE = np.diff(self.ebin)
        self.emid = 0.5 * (self.ebin[1:] + self.ebin[:-1]).to("erg")
        self.redshift = redshift

    def get_interpolator(self, data_type, e_min, e_max, energy=True):
        data = getattr(self, f"emissivity_{data_type}")
        print(data.shape)
        if not energy:
            data = data[..., :] / self.emid.v
        e_min = YTQuantity(e_min, "keV") * (1.0 + self.redshift)
        e_max = YTQuantity(e_max, "keV") * (1.0 + self.redshift)
        if (e_min - self.ebin[0]) / e_min < -1e-3 or (
            e_max - self.ebin[-1]
        ) / e_max > 1e-3:
            raise EnergyBoundsException(self.ebin[0], self.ebin[-1])
        e_is, e_ie = np.digitize([e_min, e_max], self.ebin)
        e_is = np.clip(e_is - 1, 0, self.ebin.size - 1)
        e_ie = np.clip(e_ie, 0, self.ebin.size - 1)

        my_dE = self.dE[e_is:e_ie].copy()
        # clip edge bins if the requested range is smaller
        my_dE[0] -= e_min - self.ebin[e_is]
        my_dE[-1] -= self.ebin[e_ie] - e_max

        interp_data = (data[..., e_is:e_ie] * my_dE).sum(axis=-1)
        if data.ndim == 2:
            emiss = UnilinearFieldInterpolator(
                np.log10(interp_data),
                [self.log_T[0], self.log_T[-1]],
                "log_T",
                truncate=True,
            )
        else:
            emiss = BilinearFieldInterpolator(
                np.log10(interp_data),
                [self.log_nH[0], self.log_nH[-1], self.log_T[0], self.log_T[-1]],
                ["log_nH", "log_T"],
                truncate=True,
            )

        return emiss
    
    def get_sed_interpolator(self, data_type, e_min, e_max, energy=True):
        """Similar to get_interpolator but returns spectral data for each energy bin"""
        data = getattr(self, f"emissivity_{data_type}")
        if not energy:
            data = data[..., :] / self.emid.v
            
        e_min = YTQuantity(e_min, "keV") * (1.0 + self.redshift)
        e_max = YTQuantity(e_max, "keV") * (1.0 + self.redshift)
        
        if (e_min - self.ebin[0]) / e_min < -1e-3 or (
            e_max - self.ebin[-1]
        ) / e_max > 1e-3:
            raise EnergyBoundsException(self.ebin[0], self.ebin[-1])
            
        e_is, e_ie = np.digitize([e_min, e_max], self.ebin)
        e_is = np.clip(e_is - 1, 0, self.ebin.size - 1)
        e_ie = np.clip(e_ie, 0, self.ebin.size - 1)
        
        my_dE = self.dE[e_is:e_ie].copy()
        # clip edge bins if the requested range is smaller
        my_dE[0] -= e_min - self.ebin[e_is]
        my_dE[-1] -= self.ebin[e_ie] - e_max
        
        # Instead of summing over energy bins, we keep the spectral dimension
        if data.ndim == 2:
            return {
                'data': data[:, e_is:e_ie],
                'energy_bins': self.ebin[e_is:e_ie+1],
                'energy_mids': self.emid[e_is:e_ie],
                'dE': self.dE[e_is:e_ie],
                'log_T': self.log_T
            }
        else:
            return {
                'data': data[:, :, e_is:e_ie],
                'energy_bins': self.ebin[e_is:e_ie+1],
                'energy_mids': self.emid[e_is:e_ie],
                'dE': self.dE[e_is:e_ie],
                'log_nH': self.log_nH,
                'log_T': self.log_T
            }


def calculate_xray_emissivity(
    data,
    temperature_model,   #'Tvir' or 'T_DF' or 'T_allheating'
    e_min,
    e_max,
    use_metallicity=False,
    redshift=0.0,
    table_type="cloudy",
    data_dir=None,
    cosmology=None,
    dist=None,
):
    r"""Create X-ray emissivity fields for a given energy range.

    Parameters
    ----------
    e_min : float
        The minimum energy in keV for the energy band.
    e_min : float
        The maximum energy in keV for the energy band.
    redshift : float, optional
        The cosmological redshift of the source of the field. Default: 0.0.
    metallicity : str or tuple of str or float, optional
        Either the name of a metallicity field or a single floating-point
        number specifying a spatially constant metallicity. Must be in
        solar units. If set to None, no metals will be assumed. Default:
        ("gas", "metallicity")
    table_type : string, optional
        The type of emissivity table to be used when creating the fields.
        Options are "cloudy" or "apec". Default: "cloudy"
    data_dir : string, optional
        The location to look for the data table in. If not supplied, the file
        will be looked for in the location of the YT_DEST environment variable
        or in the current working directory.
    cosmology : :class:`~yt.utilities.cosmology.Cosmology`, optional
        If set and redshift > 0.0, this cosmology will be used when computing the
        cosmological dependence of the emission fields. If not set, yt's default
        LCDM cosmology will be used.
    dist : (value, unit) tuple or :class:`~yt.units.yt_array.YTQuantity`, optional
        The distance to the source, used for making intensity fields. You should
        only use this if your source is nearby (not cosmological). Default: None

    This will create at least three fields:

    "xray_emissivity_{e_min}_{e_max}_keV" (erg s^-1 cm^-3)
    "xray_luminosity_{e_min}_{e_max}_keV" (erg s^-1)
    "xray_photon_emissivity_{e_min}_{e_max}_keV" (photons s^-1 cm^-3)

    and if a redshift or distance is specified it will create two others:

    "xray_intensity_{e_min}_{e_max}_keV" (erg s^-1 cm^-3 arcsec^-2)
    "xray_photon_intensity_{e_min}_{e_max}_keV" (photons s^-1 cm^-3 arcsec^-2)

    These latter two are really only useful when making projections.


    """
    
    lognH = data['lognH']
    nH = 10**lognH
    if table_type == "cloudy":
        norm_field=YTArray(nH**2, "cm**-6")
    elif table_type == "apec":
        ne = nH #assume fully ionized ???
        norm_field=YTArray(nH*ne, "cm**-6")
    
    Temperature = data[temperature_model]
    logT = np.log10(Temperature)
    gas_metallicity_Zsun = data['gas_metallicity_host']
    

    my_si = XrayEmissivityIntegrator(table_type, data_dir=data_dir, redshift=redshift)

    #em: energy; emp: number of photons (divided by photon energy)
    em_0 = my_si.get_interpolator("primordial", e_min, e_max)
    emp_0 = my_si.get_interpolator("primordial", e_min, e_max, energy=False)
    # if metallicity is not None:
    #     em_Z = my_si.get_interpolator("metals", e_min, e_max)
    #     emp_Z = my_si.get_interpolator("metals", e_min, e_max, energy=False)
    if use_metallicity:
        em_Z = my_si.get_interpolator("metals", e_min, e_max)
        emp_Z = my_si.get_interpolator("metals", e_min, e_max, energy=False)
    
    
    def _emissivity_field():
        with np.errstate(all="ignore"):
            dd = {
                "log_nH": lognH,
                "log_T": logT,
            }

        my_emissivity = np.power(10, em_0(dd))
        # if metallicity is not None:
        #     if isinstance(metallicity, DerivedField):
        #         my_Z = data[metallicity.name].to_value("Zsun")
        #     else:
        #         my_Z = metallicity
        #     my_emissivity += my_Z * np.power(10, em_Z(dd))
        if use_metallicity:
            my_Z = gas_metallicity_Zsun
            my_emissivity += my_Z * np.power(10, em_Z(dd))
        
        my_emissivity[np.isnan(my_emissivity)] = 0

        return  YTArray(my_emissivity, "erg*cm**3/s")
        
    
    emiss_name = f"xray_emissivity_{e_min}_{e_max}_keV"
    # ds.add_field(
    #     emiss_name,
    #     function=_emissivity_field,
    #     display_name=rf"\epsilon_{{X}} ({e_min}-{e_max} keV)",
    #     sampling_type="local",
    #     units="erg/cm**3/s",
    # )
    emissivity = _emissivity_field() * norm_field #erg/cm**3/s
    
    return emissivity
    '''
    def _luminosity_field(field, data):
        return data[emiss_name] * data[ftype, "mass"] / data[ftype, "density"]

    lum_name = (ftype, f"xray_luminosity_{e_min}_{e_max}_keV")
    ds.add_field(
        lum_name,
        function=_luminosity_field,
        display_name=rf"\rm{{L}}_{{X}} ({e_min}-{e_max} keV)",
        sampling_type="local",
        units="erg/s",
    )

    def _photon_emissivity_field(field, data):
        dd = {
            "log_nH": np.log10(data[ftype, "H_nuclei_density"]),
            "log_T": np.log10(data[ftype, "temperature"]),
        }

        my_emissivity = np.power(10, emp_0(dd))
        if metallicity is not None:
            if isinstance(metallicity, DerivedField):
                my_Z = data[metallicity.name].to_value("Zsun")
            else:
                my_Z = metallicity
            my_emissivity += my_Z * np.power(10, emp_Z(dd))

        return data[ftype, "norm_field"] * YTArray(my_emissivity, "photons*cm**3/s")

    phot_name = (ftype, f"xray_photon_emissivity_{e_min}_{e_max}_keV")
    ds.add_field(
        phot_name,
        function=_photon_emissivity_field,
        display_name=rf"\epsilon_{{X}} ({e_min}-{e_max} keV)",
        sampling_type="local",
        units="photons/cm**3/s",
    )

    fields = [emiss_name, lum_name, phot_name]

    if redshift > 0.0 or dist is not None:
        if dist is None:
            if cosmology is None:
                if hasattr(ds, "cosmology"):
                    cosmology = ds.cosmology
                else:
                    cosmology = Cosmology()
            D_L = cosmology.luminosity_distance(0.0, redshift)
            angular_scale = 1.0 / cosmology.angular_scale(0.0, redshift)
            dist_fac = ds.quan(
                1.0 / (4.0 * np.pi * D_L * D_L * angular_scale * angular_scale).v,
                "rad**-2",
            )
        else:
            redshift = 0.0  # Only for local sources!
            try:
                # normal behaviour, if dist is a YTQuantity
                dist = ds.quan(dist.value, dist.units)
            except AttributeError as e:
                try:
                    dist = ds.quan(*dist)
                except (RuntimeError, TypeError):
                    raise TypeError(
                        "dist should be a YTQuantity or a (value, unit) tuple!"
                    ) from e

            angular_scale = dist / ds.quan(1.0, "radian")
            dist_fac = ds.quan(
                1.0 / (4.0 * np.pi * dist * dist * angular_scale * angular_scale).v,
                "rad**-2",
            )

        ei_name = (ftype, f"xray_intensity_{e_min}_{e_max}_keV")

        def _intensity_field(field, data):
            I = dist_fac * data[emiss_name]
            return I.in_units("erg/cm**3/s/arcsec**2")

        ds.add_field(
            ei_name,
            function=_intensity_field,
            display_name=rf"I_{{X}} ({e_min}-{e_max} keV)",
            sampling_type="local",
            units="erg/cm**3/s/arcsec**2",
        )

        i_name = (ftype, f"xray_photon_intensity_{e_min}_{e_max}_keV")

        def _photon_intensity_field(field, data):
            I = (1.0 + redshift) * dist_fac * data[phot_name]
            return I.in_units("photons/cm**3/s/arcsec**2")

        ds.add_field(
            i_name,
            function=_photon_intensity_field,
            display_name=rf"I_{{X}} ({e_min}-{e_max} keV)",
            sampling_type="local",
            units="photons/cm**3/s/arcsec**2",
        )

        fields += [ei_name, i_name]

    for field in fields:
        mylog.info("Adding ('%s','%s') field.", field[0], field[1])

    return fields
    
    '''

def calculate_xray_sed(
    lognH,
    Temperature,
    gas_metallicity_Zsun,
    e_min,
    e_max,
    use_metallicity=False,
    redshift=0.0,
    table_type="cloudy",
    data_dir=None,
):
    """Calculate X-ray SED for given temperature and density values.
    
    Parameters
    ----------
    lognH : float or array
        Log10 of the hydrogen number density in cm^-3
    Temperature : float or array
        Gas temperature in Kelvin
    gas_metallicity_Zsun : float or array
        Gas metallicity in solar units
    e_min : float
        Minimum energy in keV
    e_max : float
        Maximum energy in keV
    use_metallicity : bool, optional
        Whether to include metal contribution
    redshift : float, optional
        Cosmological redshift
    table_type : str, optional
        Type of emissivity table ("cloudy" or "apec")
    data_dir : str, optional
        Directory containing the emissivity tables
        
    Returns
    -------
    dict
        Dictionary containing energy bins and spectral emissivity values
    """
    #lognH = data['lognH']
    #temperature_model = 'T_DF_NonEq'
    #Temperature = data['T_DF_NonEq']
    #gas_metallicity_Zsun = data['gas_metallicity_host']
    lognH = np.array(lognH)
    nH = 10**lognH
    if table_type == "cloudy":
        norm_field = YTArray(nH**2, "cm**-6")
    elif table_type == "apec":
        ne = nH  # assume fully ionized
        norm_field = YTArray(nH*ne, "cm**-6")
    
    logT = np.array(np.log10(Temperature))
    
    my_si = XrayEmissivityIntegrator(table_type, data_dir=data_dir, redshift=redshift)
    
    # Get spectral data for primordial composition
    sed_0 = my_si.get_sed_interpolator("primordial", e_min, e_max)
    
    # Interpolate primordial emissivity for each energy bin
    spectral_emissivity = np.zeros(sed_0['data'].shape[-1])
    
    for i in range(sed_0['data'].shape[-1]):
        if sed_0['data'].ndim == 2:
            interpolator = UnilinearFieldInterpolator(
                np.log10(sed_0['data'][..., i]),
                [sed_0['log_T'][0], sed_0['log_T'][-1]],
                "log_T",
                truncate=True
            )
            dd = {"log_T": np.atleast_1d(logT)}
        else:
            interpolator = BilinearFieldInterpolator(
                np.log10(sed_0['data'][..., i]),
                [sed_0['log_nH'][0], sed_0['log_nH'][-1],
                 sed_0['log_T'][0], sed_0['log_T'][-1]],
                ["log_nH", "log_T"],
                truncate=True
            )
            dd = {
                "log_nH": np.atleast_1d(lognH),
                "log_T": np.atleast_1d(logT)
            }
            
        result = interpolator(dd)
        if np.isscalar(logT):
            spectral_emissivity[i] = 10**float(result)
        else:
            spectral_emissivity[i] = 10**result
    
    # Add metallicity contribution if requested
    if use_metallicity:
        gas_metallicity_Zsun = np.array(gas_metallicity_Zsun)
        sed_Z = my_si.get_sed_interpolator("metals", e_min, e_max)
        
        metal_spectral_emissivity = np.zeros(sed_Z['data'].shape[-1])
        for i in range(sed_Z['data'].shape[-1]):
            if sed_Z['data'].ndim == 2:
                interpolator = UnilinearFieldInterpolator(
                    np.log10(sed_Z['data'][..., i]),
                    [sed_Z['log_T'][0], sed_Z['log_T'][-1]],
                    "log_T",
                    truncate=True
                )
                dd = {"log_T": np.atleast_1d(logT)}
            else:
                interpolator = BilinearFieldInterpolator(
                    np.log10(sed_Z['data'][..., i]),
                    [sed_Z['log_nH'][0], sed_Z['log_nH'][-1],
                     sed_Z['log_T'][0], sed_Z['log_T'][-1]],
                    ["log_nH", "log_T"],
                    truncate=True
                )
                dd = {
                    "log_nH": np.atleast_1d(lognH),
                    "log_T": np.atleast_1d(logT)
                }
                
            result = interpolator(dd)
            if np.isscalar(logT):
                metal_spectral_emissivity[i] = gas_metallicity_Zsun * 10**float(result)
            else:
                metal_spectral_emissivity[i] = gas_metallicity_Zsun * 10**result
        
        spectral_emissivity += metal_spectral_emissivity
    
    # Apply normalization
    spectral_emissivity = spectral_emissivity * norm_field
    
    return {
        'energy_bins': sed_0['energy_bins'],
        'energy_mids': sed_0['energy_mids'],
        'dE': sed_0['dE'],
        'spectral_emissivity': YTArray(spectral_emissivity, "erg/cm**3/s") #erg/cm**3/s/keV
    }
    
def calculate_xray_sed_batch(
    AllData,
    e_min,
    e_max,
    use_metallicity=False,
    redshift=0.0,
    table_type="cloudy",
    data_dir=None,
):
    lognH = np.array(AllData['lognH'])
    Temperature = np.array(AllData['T_DF_NonEq'])
    logT = np.log10(Temperature)
    
    nH = 10**lognH
    if table_type == "cloudy":
        norm_field = YTArray(nH**2, "cm**-6")
    elif table_type == "apec":
        ne = nH  # assume fully ionized
        norm_field = YTArray(nH*ne, "cm**-6")
    
    my_si = XrayEmissivityIntegrator(table_type, data_dir=data_dir, redshift=redshift)
    
    # Get spectral data for primordial composition
    sed_0 = my_si.get_sed_interpolator("primordial", e_min, e_max)
    n_ebins = sed_0['data'].shape[-1]
    
    # Initialize array to store total spectral emissivity with proper units
    total_spectral_emissivity = np.zeros(n_ebins)
    
    # Create interpolators once for all energy bins
    if sed_0['data'].ndim == 2:
        # Temperature-only case
        for i in range(n_ebins):
            interpolator = UnilinearFieldInterpolator(
                np.log10(sed_0['data'][..., i]),
                [sed_0['log_T'][0], sed_0['log_T'][-1]],
                "log_T",
                truncate=True
            )
            dd = {"log_T": logT}
            result = interpolator(dd)
            # Convert to YTArray before summing
            emiss = YTArray(10**result * norm_field, "erg/cm**3/s")
            total_spectral_emissivity[i] = np.sum(emiss)
    else:
        # Temperature and density case
        for i in range(n_ebins):
            interpolator = BilinearFieldInterpolator(
                np.log10(sed_0['data'][..., i]),
                [sed_0['log_nH'][0], sed_0['log_nH'][-1],
                 sed_0['log_T'][0], sed_0['log_T'][-1]],
                ["log_nH", "log_T"],
                truncate=True
            )
            dd = {
                "log_nH": lognH,
                "log_T": logT
            }
            result = interpolator(dd)
            emiss = YTArray(10**result * norm_field, "erg/cm**3/s")
            total_spectral_emissivity[i] = np.sum(emiss)
    
    # Convert total_spectral_emissivity to YTArray before metallicity contribution
    total_spectral_emissivity = YTArray(total_spectral_emissivity, "erg/cm**3/s")
    
    # Add metallicity contribution if requested
    if use_metallicity:
        gas_metallicity_Zsun = np.array(AllData['gas_metallicity_host'])
        sed_Z = my_si.get_sed_interpolator("metals", e_min, e_max)
        
        if sed_Z['data'].ndim == 2:
            # Temperature-only case
            for i in range(n_ebins):
                interpolator = UnilinearFieldInterpolator(
                    np.log10(sed_Z['data'][..., i]),
                    [sed_Z['log_T'][0], sed_Z['log_T'][-1]],
                    "log_T",
                    truncate=True
                )
                dd = {"log_T": logT}
                result = interpolator(dd)
                # Convert to YTArray with proper units before adding
                metal_emiss = YTArray(gas_metallicity_Zsun * 10**result * norm_field, "erg/cm**3/s")
                total_spectral_emissivity[i] += np.sum(metal_emiss)
        else:
            # Temperature and density case
            for i in range(n_ebins):
                interpolator = BilinearFieldInterpolator(
                    np.log10(sed_Z['data'][..., i]),
                    [sed_Z['log_nH'][0], sed_Z['log_nH'][-1],
                     sed_Z['log_T'][0], sed_Z['log_T'][-1]],
                    ["log_nH", "log_T"],
                    truncate=True
                )
                dd = {
                    "log_nH": lognH,
                    "log_T": logT
                }
                result = interpolator(dd)
                metal_emiss = YTArray(gas_metallicity_Zsun * 10**result * norm_field, "erg/cm**3/s")
                total_spectral_emissivity[i] += np.sum(metal_emiss)
    
    return {
        'energy_bins': sed_0['energy_bins'],
        'energy_mids': sed_0['energy_mids'],
        'dE': sed_0['dE'],
        'spectral_emissivity': total_spectral_emissivity
    }

def plot_statistics(AllData, output_dir, current_redshift):
     
    
    #histogram of Tvir, T_DF, T_allheating (temperature log scale)
    fig = plt.figure(figsize=(8, 6),facecolor='white')
    bins = np.logspace(np.log10(1e3), np.log10(1e9), 100)
    plt.hist(AllData['Tvir'], bins=bins, color='blue', alpha=0.4, label='Tvir')
    plt.hist(AllData['T_DF'], bins=bins, color='green', alpha=0.6, label='T_DF')
    plt.hist(AllData['T_allheating'], bins=bins, color='red', alpha=0.4, label='T_allheating')
    plt.xscale('log')
    plt.xlabel('Temperature [K]')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Histogram of Temperature, z = '+str(current_redshift))
    plt.savefig(output_dir+"histogram_Temperature_SubhaloWake.png",dpi=300)    
    
    normalized_heating = AllData['normalized_heating']
    cooling_rate_Tvir = AllData['cooling_rate_Tvir']
    
    #ratio between normalized_heating and cooling_rate_Tvir
    ratio_heating_cooling_Tvir = normalized_heating / np.abs(cooling_rate_Tvir)
    print(ratio_heating_cooling_Tvir)
    print("max ratio_heating_cooling_Tvir: ", np.max(ratio_heating_cooling_Tvir))
    print("min ratio_heating_cooling_Tvir: ", np.min(ratio_heating_cooling_Tvir))
    
    log_ratio_heating_cooling_Tvir = np.log10(ratio_heating_cooling_Tvir)
    fig = plt.figure(figsize=(8, 6),facecolor='white')
    plt.hist(log_ratio_heating_cooling_Tvir, color='blue', alpha=0.4, label='log(heating / cooling at Tvir)')
    plt.xlabel('log(heating / cooling at Tvir)')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Histogram of log(heating / cooling at Tvir), z = '+str(current_redshift))
    plt.savefig(output_dir+"histogram_heating_cooling_Tvir_ratio.png",dpi=300)
    
    
    #log_ratio_heating_cooling_TDF  VS Tvir
    fig = plt.figure(figsize=(8, 6),facecolor='white')
    logTvir = np.log10(AllData['Tvir'])
    plt.scatter(logTvir, log_ratio_heating_cooling_Tvir, color='blue', alpha=0.4, label='log(heating / cooling at Tvir)')
    plt.xlabel('log(Tvir) [K]')
    plt.ylabel('log(heating / cooling at Tvir)')
    plt.legend()
    plt.title('log(heating / cooling at Tvir) vs Tvir, z = '+str(current_redshift))
    plt.savefig(output_dir+"scatter_heating_cooling_Tvir_ratio.png",dpi=300)
    
    



if __name__ == "__main__":
    TNG50_redshift_list = [20.05,14.99,11.98,10.98,10.00,9.39,9.00,8.45,8.01]
    snapNum = 1
    current_redshift = TNG50_redshift_list[snapNum]
    input_dir = "/home/zwu/21cm_project/grackle_DF_cooling/snap_"+str(snapNum)+"/"
    output_dir = "/home/zwu/21cm_project/yt_Xray/snap_"+str(snapNum)+"/"
    
    model = 'SubhaloWakeNonEq'
    if model == 'SubhaloWake': 
        HaloData = read_hdf5_data(input_dir + "Grackle_Cooling_SubhaloWake_FullModel_snap"+str(snapNum)+".h5")
        display_hdf5_contents(input_dir + "Grackle_Cooling_SubhaloWake_FullModel_snap"+str(snapNum)+".h5")
        
        AllData = HaloData['SubhaloWake']
    elif model == 'SubhaloWakeNonEq':
        input_filename = input_dir + "Grackle_Cooling_SubhaloWake_NonEq_Volume0.1_snap"+str(snapNum)+".h5"
        HaloData = read_hdf5_data(input_filename)
        print("keys: ", HaloData.keys())
        AllData = HaloData['SubhaloWakeNonEq']
        
    
    heating_rate = AllData['heating']
    specific_heating = AllData['specific_heating']
    volumetric_heating = AllData['volumetric_heating']
    normalized_heating = AllData['normalized_heating']
    volume_wake_tdyn_cm3 = AllData['volume_wake_tdyn_cm3']
    
    cooling_rate_Tvir = AllData['cooling_rate_Tvir']
    
    
    if model == 'SubhaloWake':
        cooling_rate_TDF = AllData['cooling_rate_TDF']
        cooling_rate_Tallheating = AllData['cooling_rate_Tallheating']
        net_heating_flag = AllData['net_heating_flag']
    elif model == 'SubhaloWakeNonEq':
        cooling_rate_TDF_NonEq = AllData['cooling_rate_TDF_NonEq']
        tfinal = AllData['tfinal']
        T_DF_NonEq = AllData['T_DF_NonEq']
           
    num_display = 5
    
    print(f"\nTvir: {AllData['Tvir'][:num_display]} K")
    
    
    #find index where Tvir > 1e5 K and T_DF_NonEq > 2* Tvir
    # mask = (AllData['Tvir'] > 1e5) & (AllData['T_DF_NonEq'] > 2*AllData['Tvir'])
    # test_index = np.where(mask)[0]
    # test_index = test_index[0]
  
    
    
    #print(f"T_DF: {AllData['T_DF'][:num_display]} K")
    # print(f"T_allheating: {AllData['T_allheating'][:num_display]} K")
    #print("Specific heating rate: ", specific_heating[:num_display], "erg/g/s")
    #print("Volumetric heating rate: ", volumetric_heating[:num_display], "erg/cm^3/s")
    # print("Normalized heating rate: ", normalized_heating[:num_display], "erg cm^3 s^-1")
    
    # print("Cooling rate at Tvir: ", cooling_rate_Tvir[:num_display])
    # print("Cooling rate at T_DF: ", cooling_rate_TDF[:num_display])
    # print("net DF heating: ", normalized_heating[:num_display] + cooling_rate_TDF[:num_display])
   
    total_heating_rate = np.sum(heating_rate)
    # total_heating_rate_2 = np.sum(volume_wake_tdyn_cm3 * volumetric_heating)
    
    #X-ray emissivity at Tvir
    use_metallicity = True  #do not consider metallicity here
    
    normalized_heating = YTArray(normalized_heating, "erg*cm**3/s")  
    
    print(AllData['T_DF_NonEq'])
    '''
    #test SED
    T = 10**7.0
    lognH = -0.39
    nH = 10**lognH
    gas_metallicity_Zsun = 0.0
    e_min = 0.5 
    e_max = 2.0
    redshift = 0.0
    table_type = "cloudy"
    data_dir = "."
    SED = calculate_xray_sed(lognH, T, gas_metallicity_Zsun, e_min, e_max, use_metallicity, redshift, table_type, data_dir)
    
    
    # print("emid: ", SED['energy_mids'])
    

    # plot
    fig = plt.figure(figsize=(8, 6),facecolor='white')
    plt.plot(SED['energy_mids'].in_units("keV"), SED['spectral_emissivity'], color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy [keV]')
    plt.ylabel(r'Spectral emissivity [erg/cm$^3$/s/keV]')
    plt.grid()
    plt.title(f'X-ray SED at T = {T:.1e} K, lognH = {lognH}')
    filename = f"SED/Xray_SED_T_{T:.1e}_nH_{lognH}.png"
    plt.savefig(filename,dpi=300)
    
    fig = plt.figure(figsize=(8, 6),facecolor='white')
    plt.plot(SED['energy_mids'].in_units("keV"), SED['spectral_emissivity']/nH**2, color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy [keV]')
    plt.ylabel(r'Spectral emissivity/nH$^2$ [erg cm$^3$/s/keV]')
    plt.grid()
    plt.title(f'X-ray SED at T = {T:.1e} K, lognH = {lognH}')
    filename = f"SED/Xray_normalizedSED_T_{T:.1e}_nH_{lognH}.png"
    plt.savefig(filename,dpi=300)
    '''
    
    #SED
    '''
    #plot SED at T_DF_NonEq
    e_min = 0.5
    e_max = 2.0
    redshift = 0.0
    table_type = "cloudy"
    data_dir = "."
    
    All_SED = calculate_xray_sed(AllData, e_min, e_max, use_metallicity, redshift, table_type, data_dir)
    energy_mids = All_SED['energy_mids']
    total_spectral_emissivity = All_SED['spectral_emissivity']
    print("emid: ", energy_mids)
    print("total_spectral_emissivity: ", total_spectral_emissivity)
    #plot
    fig = plt.figure(figsize=(8, 6),facecolor='white')
    plt.plot(energy_mids.in_units("keV"), total_spectral_emissivity, color='blue', alpha=0.4, label='T_DF_NonEq')
    plt.yscale('log')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Spectral emissivity [erg/cm^3/s/keV]')
    plt.legend()
    plt.title('X-ray SED at T_DF_NonEq')
    plt.savefig(output_dir+"Xray_SED_T_DF_NonEq.png",dpi=300)
    
    '''  
    
    
    # AllResult = calculate_xray_sed_batch(AllData, 0.5, 2.0, use_metallicity, redshift=0.0, table_type="cloudy", data_dir=".")
    # #plot the total SED
    # fig = plt.figure(figsize=(8, 6),facecolor='white')
    # energy_mids = AllResult['energy_mids']
    # total_spectral_emissivity = AllResult['spectral_emissivity']
    # plt.plot(energy_mids.in_units("keV"), total_spectral_emissivity, color='blue', alpha=0.4, label='Total')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Energy [keV]')
    # plt.ylabel('Spectral emissivity [erg/cm^3/s/keV]')
    # plt.title('X-ray SED at T_DF_NonEq')
    # plt.savefig(output_dir+"Xray_SED_T_DF_NonEq.png",dpi=300)
    
    
    
    
    
    #X-ray Emissivity
    
    #do not consider redshift here
    emissivity_Tvir_cloudy = calculate_xray_emissivity(AllData, 'Tvir', 0.5, 2.0, use_metallicity, redshift=0.0, table_type="cloudy", data_dir=".", cosmology=None, dist=None)
    print("Emissivity at Tvir: ", emissivity_Tvir_cloudy)
    print(np.sum(YTArray(volume_wake_tdyn_cm3, "cm**3") * emissivity_Tvir_cloudy))
    
    emissivity_Tvir_apec = calculate_xray_emissivity(AllData, 'Tvir', 0.5, 2.0, use_metallicity, redshift=0.0, table_type="apec", data_dir=".", cosmology=None, dist=None)
    print(np.sum(YTArray(volume_wake_tdyn_cm3, "cm**3") * emissivity_Tvir_apec))
    
    if model == 'SubhaloWake':
        emissivity_Tallheating_cloudy = calculate_xray_emissivity(AllData, 'T_allheating', 0.5, 2.0, use_metallicity, redshift=0.0, table_type="cloudy", data_dir=".", cosmology=None, dist=None)
        print("Emissivity at T_allheating: ", emissivity_Tallheating_cloudy)
        print(np.sum(YTArray(volume_wake_tdyn_cm3, "cm**3") * emissivity_Tallheating_cloudy))
        
        emissivity_Tallheating_apec = calculate_xray_emissivity(AllData, 'T_allheating', 0.5, 2.0, use_metallicity, redshift=0.0, table_type="apec", data_dir=".", cosmology=None, dist=None)
        print(np.sum(YTArray(volume_wake_tdyn_cm3, "cm**3") * emissivity_Tallheating_apec))
    
    elif model == 'SubhaloWakeNonEq':
        
        emissivity_TDF_NonEq_cloudy = calculate_xray_emissivity(AllData, 'T_DF_NonEq', 0.5, 2.0, use_metallicity, redshift=0.0, table_type="cloudy", data_dir=".", cosmology=None, dist=None)
        print("Emissivity at T_DF_NonEq: ", emissivity_TDF_NonEq_cloudy)
        print(np.sum(YTArray(volume_wake_tdyn_cm3, "cm**3") * emissivity_TDF_NonEq_cloudy))
        
        emissivity_TDF_NonEq_apec = calculate_xray_emissivity(AllData, 'T_DF_NonEq', 0.5, 2.0, use_metallicity, redshift=0.0, table_type="apec", data_dir=".", cosmology=None, dist=None)
        print(np.sum(YTArray(volume_wake_tdyn_cm3, "cm**3") * emissivity_TDF_NonEq_apec))
        
    #write to file
    if model == 'SubhaloWake':
        output_filename = output_dir + "Xray_emissivity_snap"+str(snapNum)+".h5"
        with h5py.File(output_filename, 'w') as f:
            f.create_dataset('heating_rate', data=heating_rate)
            f.create_dataset('emissivity_Tvir_cloudy', data=emissivity_Tvir_cloudy)
            f.create_dataset('emissivity_Tvir_apec', data=emissivity_Tvir_apec)
            f.create_dataset('emissivity_Tallheating_cloudy', data=emissivity_Tallheating_cloudy)
            f.create_dataset('emissivity_Tallheating_apec', data=emissivity_Tallheating_apec)
            f.create_dataset('volume_wake_tdyn_cm3', data=volume_wake_tdyn_cm3)
            f.create_dataset('Mach_rel', data=AllData['Mach_rel'])
            f.create_dataset('vel_rel', data=AllData['vel_rel'])
            f.create_dataset('Tvir', data=AllData['Tvir'])
            f.create_dataset('T_DF', data=AllData['T_DF'])
            f.create_dataset('T_allheating', data=AllData['T_allheating'])
    elif model == 'SubhaloWakeNonEq':
        output_filename = output_dir + "Xray_emissivity_NonEq_Volume0.1_snap"+str(snapNum)+".h5" 
        with h5py.File(output_filename, 'w') as f:
            f.create_dataset('index', data=AllData['index'])
            f.create_dataset('heating_rate', data=heating_rate)
            f.create_dataset("normalized_heating", data=normalized_heating)
            f.create_dataset("cooling_rate_Tvir", data=cooling_rate_Tvir)
            f.create_dataset('emissivity_Tvir_cloudy', data=emissivity_Tvir_cloudy)
            f.create_dataset('emissivity_Tvir_apec', data=emissivity_Tvir_apec)
            f.create_dataset('emissivity_TDF_NonEq_cloudy', data=emissivity_TDF_NonEq_cloudy)
            f.create_dataset('emissivity_TDF_NonEq_apec', data=emissivity_TDF_NonEq_apec)
            f.create_dataset('volume_wake_tdyn_cm3', data=volume_wake_tdyn_cm3)
            f.create_dataset('Mach_rel', data=AllData['Mach_rel'])
            f.create_dataset('vel_rel', data=AllData['vel_rel'])
            f.create_dataset('Tvir', data=AllData['Tvir'])
            f.create_dataset('T_DF_NonEq', data=AllData['T_DF_NonEq'])
            f.create_dataset('tfinal', data=tfinal)
              
            
            









    
    '''
    # Plot the histogram of X-ray fraction at Tvir
    fig = plt.figure(figsize=(8, 6), facecolor='white')
    log_Xrayfraction_Tvir = np.log10(Xrayfraction_Tvir)
    min_logXray = log_Xrayfraction_Tvir.min()
    max_logXray = log_Xrayfraction_Tvir.max()
    print("min log X-ray fraction at Tvir: ", min_logXray)
    print("max log X-ray fraction at Tvir: ", max_logXray)

    # Separate data into heating and cooling
    heating_mask = (net_heating_flag == 1)  # Assuming 1 indicates heating
    cooling_mask = (net_heating_flag == -1)  # Assuming 0 indicates cooling

    log_Xrayfraction_Tvir_heating = log_Xrayfraction_Tvir[heating_mask]
    log_Xrayfraction_Tvir_cooling = log_Xrayfraction_Tvir[cooling_mask]

    # Define bins
    bins = np.linspace(min_logXray, max_logXray, 101)  # 100 bins

    # Plot stacked histogram
    plt.hist([log_Xrayfraction_Tvir_heating, log_Xrayfraction_Tvir_cooling],
            bins=bins, color=['red', 'blue'], alpha=1.0, stacked=True, label=['Absolute DF Heating', 'Cooling with only DF Heating'])
    plt.xlabel('Log(X-ray fraction at Tvir)')
    plt.ylabel('Count')
    plt.title('Histogram of X-ray Fraction at Tvir')
    plt.legend()
    plt.savefig(output_dir + "histogram_Xrayfraction_Tvir_SubhaloWake.png", dpi=300)
    '''
        
    
    
    
    '''
    
    
    #plot the 2D distribution of Tvir and T_DF
    fig = plt.figure(figsize=(8, 6),facecolor='white')
    log_Tvir = np.log10(AllData['Tvir'])
    print("T_DF:", AllData['T_DF'])
    #check if T_DF has negative values
    print("min T_DF: ", np.min(AllData['T_DF']))
    exit()
    log_T_DF = np.log10(AllData['T_DF'])

    # Create the 2D histogram
    counts, xedges, yedges, Image = plt.hist2d(log_Tvir, log_T_DF, bins=[20,20], norm=colors.LogNorm())

    # Set up the plot with labels and a colorbar
    plt.colorbar(label='Count in bin')
    plt.xlabel('Log(Tvir) [K]')
    plt.ylabel('Log(T_DF) [K]')
    plt.title('2D Histogram of Virial and DF Gas Temperatures')
    plt.savefig(output_dir+"2D_histogram_Tvir_TDF_SubhaloWake.png",dpi=300)
        

    #plot T_DF vs X-ray fraction 2D histogram
    fig = plt.figure(figsize=(8, 6),facecolor='white')
    log_Xrayfraction_TDF = np.log10(Xrayfraction_TDF)
    min_logXray = log_Xrayfraction_TDF.min()
    max_logXray = log_Xrayfraction_TDF.max()
    print("min log X-ray fraction at T_DF: ", min_logXray)
    print("max log X-ray fraction at T_DF: ", max_logXray)
    # Create the 2D histogram
    counts, xedges, yedges, Image = plt.hist2d(log_T_DF, log_Xrayfraction_TDF, bins=[20,20], norm=colors.LogNorm())
    
    # Set up the plot with labels and a colorbar
    plt.colorbar(label='Count in bin')
    plt.xlabel('Log(T_DF) [K]')
    plt.ylabel('Log(X-ray fraction at T_DF)')
    plt.title('2D Histogram of DF Gas Temperature and X-ray Fraction')
    plt.savefig(output_dir+"2D_histogram_TDF_Xrayfraction_SubhaloWake.png",dpi=300)
    
    
    #plot Tvir vs lognH 2D histogram
    fig = plt.figure(figsize=(8, 6),facecolor='white')
    lognH = AllData['lognH']
    print(lognH)
    # Create the 2D histogram
    counts, xedges, yedges, Image = plt.hist2d(log_Tvir, lognH, bins=[20,20], norm=colors.LogNorm())
    
    # Set up the plot with labels and a colorbar
    plt.colorbar(label='Count in bin')
    plt.xlabel('Log(Tvir) [K]')
    plt.ylabel('Log(nH) [cm^-3]')
    plt.title('2D Histogram of Virial Temperature and lognH')
    plt.savefig(output_dir+"2D_histogram_Tvir_lognH_SubhaloWake.png",dpi=300)
    
    
    '''
    
    
