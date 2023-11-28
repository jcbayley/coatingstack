def physical_to_optical_thickness(d, n, wavelength):
    """
    Convert physical thickness to optical thickness.
    
    Parameters:
    - d: Physical thickness (in meters)
    - n: Refractive index of the material
    - wavelength: Wavelength of light (in meters)

    Returns:
    - Optical thickness (in meters)
    """
    optical_thickness = (n * d) / wavelength
    return optical_thickness

def optical_to_physical_thickness(ot, n, wavelength):
    """
    Convert optical thickness to physical thickness.
    
    Parameters:
    - ot: Optical thickness (in meters)
    - n: Refractive index of the material
    - wavelength: Wavelength of light (in meters)

    Returns:
    - Physical thickness (in meters)
    """
    physical_thickness = (ot * wavelength) / n
    return physical_thickness

# Given values
refractive_index = 2.04  # Refractive index of the material
wavelength = 1064e-9  # Wavelength in meters (1064 nm)

# Convert physical thickness to optical thickness
physical_thickness = 0.01  # Physical thickness in meters (e.g., 10 micrometers)
optical_thickness = physical_to_optical_thickness(physical_thickness, refractive_index, wavelength)
print(f"Physical Thickness: {physical_thickness} meters")
print(f"Optical Thickness: {optical_thickness} meters")

# Convert optical thickness to physical thickness
optical_thickness = 1e-6  # Optical thickness in meters (e.g., 1 micron)
physical_thickness = optical_to_physical_thickness(optical_thickness, refractive_index, wavelength)
print(f"Optical Thickness: {optical_thickness} meters")
print(f"Physical Thickness: {physical_thickness} meters")
