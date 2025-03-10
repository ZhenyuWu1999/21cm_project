import h5py
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Any
'''
def get_simulation_resolution_old(simulation_set):
    if simulation_set == 'TNG50-1':
        gas_resolution = 8.5e4 * h_Hubble #convert from Msun to Msun/h
        dark_matter_resolution = 4.5e5 * h_Hubble
    elif simulation_set == 'TNG100-1':
        gas_resolution = 1.4e6 * h_Hubble  
        dark_matter_resolution = 7.5e6 * h_Hubble  
    elif simulation_set == 'TNG300-1':
        gas_resolution = 1.1e7 * h_Hubble
        dark_matter_resolution = 5.9e7 * h_Hubble
'''

def get_simulation_resolution(simulation_set):
    #resolution in units of Msun/h, https://www.tng-project.org/data/docs/background/
    if simulation_set == 'TNG50-1':
        gas_resolution = 5.7e4
        dark_matter_resolution = 3.1e5 
    elif simulation_set == 'TNG100-1':
        gas_resolution = 9.4e5  
        dark_matter_resolution = 5.1e6  
    elif simulation_set == 'TNG300-1':
        gas_resolution = 7.6e6
        dark_matter_resolution = 4.0e7 
    return gas_resolution, dark_matter_resolution


@dataclass  # Decorator to automatically generate __init__ and __repr__ and __eq__ methods
class PhysicalQuantity:
    """Container for physical quantities with units"""
    value: np.ndarray
    units: str
    description: Optional[str] = None

class ProcessedTNGData:
    """Custom container for processed TNG data"""
    def __init__(self):
        self.header = {}
        self.halo_data = {}
        self.subhalo_data = {}
        self.metadata = {}
    
    def add_halo_quantity(self, name: str, value: np.ndarray, units: str, description: Optional[str] = None, dtype: Optional[np.dtype] = None):
        """Add a physical quantity to halo data"""
        if dtype is not None:
            value = value.astype(dtype)
        self.halo_data[name] = PhysicalQuantity(value, units, description)
    
    def add_subhalo_quantity(self, name: str, value: np.ndarray, units: str, description: Optional[str] = None, dtype: Optional[np.dtype] = None):
        """Add a physical quantity to subhalo data"""
        if dtype is not None:
            value = value.astype(dtype)
        self.subhalo_data[name] = PhysicalQuantity(value, units, description)

def save_processed_data(filepath: str, data: ProcessedTNGData) -> None:
    """
    Save processed TNG data to HDF5 file.
    
    Parameters:
    -----------
    filepath : str
        Path to save the HDF5 file
    data : ProcessedTNGData
        Processed data container
    """
    with h5py.File(filepath, 'w') as f:
        # Save header
        header_group = f.create_group('header')
        for key, value in data.header.items():
            if isinstance(value, (str, int, float, bool)):
                # Save as attribute if scalar
                header_group.attrs[key] = value
            else:
                try:
                    header_group.create_dataset(key, data=value)
                except:
                    # save as attribute if it's a scalar numpy array
                    if isinstance(value, np.ndarray) and value.size == 1:
                        header_group.attrs[key] = value.item()
                    else:
                        print(f"Warning: Could not save header item {key}")
                
                
        # Save metadata
        meta_group = f.create_group('metadata')
        for key, value in data.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                meta_group.attrs[key] = value
            else:
                meta_group.create_dataset(key, data=value)
        
        
        # Save halo data
        halo_group = f.create_group('halos')
        for name, quantity in data.halo_data.items():
            dset = halo_group.create_dataset(name, data=quantity.value)
            dset.attrs['units'] = quantity.units
            if quantity.description:
                dset.attrs['description'] = quantity.description
        
        # Save subhalo data
        subhalo_group = f.create_group('subhalos')
        for name, quantity in data.subhalo_data.items():
            dset = subhalo_group.create_dataset(name, data=quantity.value)
            dset.attrs['units'] = quantity.units
            if quantity.description:
                dset.attrs['description'] = quantity.description

def load_processed_data(filepath: str) -> ProcessedTNGData:
    """
    Load processed TNG data from HDF5 file.
    
    Parameters:
    -----------
    filepath : str
        Path to the HDF5 file
    
    Returns:
    --------
    ProcessedTNGData
        Loaded data container
    """
    data = ProcessedTNGData()
    
    with h5py.File(filepath, 'r') as f:
        # Load header
        print("loading header...")
        header_group = f['header']
        data.header.update(dict(header_group.attrs))
        for key in header_group.keys():
            dset = header_group[key]
            if dset.shape == ():  # scalar data set
                data.header[key] = dset[()]
            else:  # array data set
                data.header[key] = dset[:]
        
        # Load metadata
        print("loading metadata...")
        meta_group = f['metadata']
        data.metadata.update(dict(meta_group.attrs))
        for key in meta_group.keys():
            data.metadata[key] = meta_group[key][:]
        
        # Load halo data
        print("loading halo data...")
        for name in f['halos'].keys():
            dset = f['halos'][name]
            data.add_halo_quantity(
                name=name,
                value=dset[:],
                units=dset.attrs['units'],
                description=dset.attrs.get('description')
            )
        
        # Load subhalo data
        print("loading subhalo data...")
        for name in f['subhalos'].keys():
            dset = f['subhalos'][name]
            data.add_subhalo_quantity(
                name=name,
                value=dset[:],
                units=dset.attrs['units'],
                description=dset.attrs.get('description')
            )
    
    return data

def add_quantities(filepath: str, 
                  new_quantities: Dict[str, PhysicalQuantity], 
                  group: str = 'halos') -> None:
    """
    Add new quantities to existing processed data.
    
    Parameters:
    -----------
    filepath : str
        Path to the HDF5 file
    new_quantities : dict
        Dictionary of new quantities to add
    group : str
        Group to add quantities to ('halos' or 'subhalos')
    """
    with h5py.File(filepath, 'r+') as f:
        target_group = f[group]
        for name, quantity in new_quantities.items():
            if name in target_group:
                del target_group[name]
            dset = target_group.create_dataset(name, data=quantity.value)
            dset.attrs['units'] = quantity.units
            if quantity.description:
                dset.attrs['description'] = quantity.description