import h5py
import numpy as np
from scipy.integrate import solve_ivp


def print_attrs(name, obj):
    """Helper function to print the name of an HDF5 object and its attributes."""
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))

def display_hdf5_contents(filepath):
        # Open the HDF5 file in read-only mode
    with h5py.File(filepath, 'r') as f:       
        # Display all groups and datasets in the file
        for name, item in f.items():
            if isinstance(item, h5py.Group):
                print(f"Group: {name}")
                for key, val in item.attrs.items():
                    print(f"    {key}: {val}")
                for subname, subitem in item.items():
                    print_attrs(f"{name}/{subname}", subitem)
            elif isinstance(item, h5py.Dataset):
                print(f"Dataset: {name}")
                for key, val in item.attrs.items():
                    print(f"    {key}: {val}")

def read_hdf5_data(filepath):
    data_dict = {}
    # Open the HDF5 file in read-only mode
    with h5py.File(filepath, 'r') as f:
        # Function to recursively read data
        def read_recursive(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Store dataset data in dictionary
                data_dict[name] = np.array(obj)
                print(f"Dataset: {name} loaded")
                # for key, val in obj.attrs.items():
                #     print(f"    {key}: {val}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")
                for key, val in obj.attrs.items():
                    print(f"    {key}: {val}")
                for subname, subitem in obj.items():
                    read_recursive(f"{name}/{subname}", subitem)
        # Read data starting from root
        f.visititems(read_recursive)
    return data_dict


def integrate_ode_z(ode_func, z_initial, y_initial, z_final, 
                  n_points=1000, method='RK45', rtol=1e-8, atol=1e-10,
                  ode_args=(), verbose=False, **solver_kwargs):
    """
    Generalized ODE integration function for redshift-dependent problems.
    """
    
    def ode_wrapper(z, y):
        """Wrapper to handle additional arguments"""
        return ode_func(z, y, *ode_args)
    
    # Ensure y_initial is array-like
    y0 = np.atleast_1d(y_initial)
    
    # Set up integration bounds with small buffer to avoid boundary issues
    epsilon = 1e-5 * abs(z_initial - z_final) / max(abs(z_initial), abs(z_final), 1)
    if z_initial > z_final:
        z_span = (z_initial - epsilon, z_final + epsilon)
    else:
        z_span = (z_initial + epsilon, z_final - epsilon)
    
    # Create evaluation points
    z_eval = np.logspace(np.log10(max(z_initial, z_final)), 
                        np.log10(min(z_initial, z_final)), 
                        n_points)
    
    # Ensure evaluation points are within bounds
    z_eval = np.clip(z_eval, min(z_span), max(z_span))
    
    if verbose:
        print(f"Integration from z = {z_initial} to z = {z_final}")
        print(f"z_span = {z_span}")
        print(f"Initial conditions: {y0}")
        print(f"ODE arguments: {ode_args}")
        print(f"Number of evaluation points: {len(z_eval)}")
    
    # Integrate
    sol = solve_ivp(ode_wrapper, z_span, y0, t_eval=z_eval, 
                    method=method, rtol=rtol, atol=atol, **solver_kwargs)
    
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    
    # Package results
    result = {
        'z': sol.t,
        'y': sol.y
    }
    
    if verbose:
        print(f"Integration completed successfully!")
        print(f"Final values at z = {sol.t[-1]:.1f}: {sol.y[:, -1]}")
    
    return result