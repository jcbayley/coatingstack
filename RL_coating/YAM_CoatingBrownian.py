#from thermal_noise_hong import getCoatBrownian
from deap import base, creator, tools
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random
import os 
from gwinc import * 
import numpy as np



def generate_coating_stack(lambda_):
    """
    Randomly generates a coating stack with paired layers and writes to a text file.

    :return: n_input, dOpt
    """
    # Generate a random integer between 2 and 4 to determine the number of materials
    num_materials = random.randint(2, 4)

    # Generate the specified number of random refractive indices between 1 and 4
    n_input = sorted([random.uniform(1, 4) for _ in range(num_materials)])

    # Generate a random integer between 1 and 5 to determine the number of pairs
    num_pairs = random.randint(2, 50)

    # Generate the specified number of pairs with maximum contrast
    dOpt = []
    for _ in range(num_pairs):
        # Add a pair with maximum contrast
        # Using 1 and num_materials as material numbers
        dOpt.extend([1, num_materials])

    # Normalize dOpt
    unique_dOpt = np.unique(dOpt)
    mapping = {val: i+1 for i, val in enumerate(unique_dOpt)}
    dOpt = [mapping[val] for val in dOpt]

    # Filter n_input to only include values corresponding to materials in dOpt
    n_input = [n_input[i-1] for i in unique_dOpt]
     # Calculate individual physical thickness for each material
    d_physical = lambda_ / (4 * np.array(n_input))

    # Arrays for each layer
    n_layers = np.array(n_input)[np.array(dOpt) - 1]
    material_kind = dOpt
    d_physical_layers = np.array(d_physical)[np.array(dOpt) - 1]*1E6

    
    # Check if file exists and create a new filename with an increasing number
    counter = 1
    filename = 'generated_coating_01.txt'
    while os.path.exists(filename):
        filename = f'generated_coating_{counter:02d}.txt'
        counter += 1

    # Write to text file
    with open(filename, 'w') as file:
        for i in dOpt:
            file.write(f"material_{i}\t {d_physical_layers[i-1]:.2f}\n")

    return n_input, dOpt


def thin_film_stack(n_input, dOpt, lambda_):
    """
    Generates and plots a thin film coating stack.

    :param n_input: Array of refractive indices for each material.
    :param dOpt: Array specifying the material for each layer.
    :param lambda_: Wavelength.
    :return: n_layers, material_kind, d_physical_layers
    """
    if len(n_input) < max(dOpt):
        raise ValueError('The number of refractive indices provided is less than the required materials in dOpt.')

    # Calculate individual physical thickness for each material
    d_physical = lambda_ / (4 * np.array(n_input))

    # Arrays for each layer
    n_layers = np.array(n_input)[np.array(dOpt) - 1]
    material_kind = dOpt
    d_physical_layers = np.array(d_physical)[np.array(dOpt) - 1]


    # Plotting the thin film stack
    unique_materials = list(set(dOpt))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_materials)))  # generate distinct colors for materials

    # Check if the file exists to write headers
    file_exists = os.path.exists('coating_metadata.txt')

    # Write metadata to text file
    with open('coating_metadata.txt', 'a') as file:
        if not file_exists:
            file.write("Lambda (m)\tTotal Layers\t")
            for i in range(1, 7):
                file.write(f"Material_{i}\tNo. Layers_{i}\tPhysical Thickness_{i} (m)\tRefractive Index_{i}\t")
            file.write("\n")
        
        file.write(f"{lambda_:.2e}\t{len(dOpt)}\t")
        for i in range(1, 7):
            if i in dOpt:
                file.write(f"Material {i}\t")
                file.write(f"{len([x for x in dOpt if x == i])}\t")
                file.write(f"{sum(d_physical_layers[dOpt == i]):.2e}\t")
                file.write(f"{n_input[i-1]:.2f}\t")  # Directly access the refractive index from n_input
            else:
                file.write("NaN\tNaN\tNaN\tNaN\t")
        file.write("\n")
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.grid(True)
    depth_so_far = 0  # To keep track of where to plot the next bar
    for i in range(len(dOpt)):
        material_idx = dOpt[i]
        color_idx = unique_materials.index(material_idx)
        plt.bar(depth_so_far + d_physical_layers[i] / 2, d_physical_layers[i], color=colors[color_idx],
                width=d_physical_layers[i])
        depth_so_far += d_physical_layers[i]

    plt.xlim([0, sum(d_physical_layers) * 1.01])
    plt.ylabel('Physical Thickness [nm]')
    plt.xlabel('Layer Position')
    plt.title('Generated Stack')
    legend_str = ['n = ' + str(n) for n in n_input]
    plt.grid(False)
    plt.legend(legend_str)

    # Additional code for plotting the normalized electric field intensity squared can be added here

    plt.show()

   # Printing coating properties
    print("\nCoating Properties:\n")
    print(f"\nLaser Wavelength:\t\t{lambda_*1E9:.2f} nm")
    print(f"Number of Materials:\t\t{len(unique_materials):d}")
    print(f"Total Physical Thickness:\t{sum(d_physical_layers):.2e} m")

    
    for i in unique_materials:
        print(f"\n--------- Material {i} -------------\n")
        dOpt_array = np.array(dOpt)
        matching_indices = dOpt_array == i
        n_layers_matching = n_layers[matching_indices]
        d_physical_layers_matching = d_physical_layers[matching_indices]

        if len(n_layers_matching) > 0:
            print(f"No. Layers:\t\t\t{len(n_layers_matching)}")
            print(f"Total Physical Thickness:\t{sum(d_physical_layers_matching):.2e} m")
            print(f"Refractive Index:\t\t{np.unique(n_layers_matching)[0]:.2f}")
        else:
            print(f"No layers of material {i} found.")

    return n_layers, material_kind, d_physical_layers


#functions used to Calculate Coating Thermal Noise 
# not to be used to calculate optical properties 

def getCoatRefl2(nIn, nOut, nLayer, dOpt):
    # Vector of all refractive indices
    nAll = np.concatenate(([nIn], nLayer, [nOut]))
    
    # Reflectivity of each interface
    r = (nAll[:-1] - nAll[1:]) / (nAll[:-1] + nAll[1:])
    
    # Combine reflectivities
    rbar = np.zeros_like(r, dtype=complex)
    ephi = np.zeros_like(r, dtype=complex)
    
    ephi[-1] = np.exp(-4j * np.pi * dOpt[-1])
    rbar[-1] = ephi[-1] * r[-1]
    
    for n in range(len(dOpt)-1, -1, -1):
        # Round-trip phase in this layer
        ephi[n] = np.exp(-4j * np.pi * dOpt[n - 1]) if n > 0 else 1
        
        # Accumulate reflectivity
        rbar[n] = ephi[n] * (r[n] + rbar[n + 1]) / (1 + r[n] * rbar[n + 1])
    
    # Reflectivity derivatives
    dr_dphi = np.zeros_like(dOpt, dtype=complex)
    
    for n in range(len(dOpt)-1, -1, -1):
        dr_dphi[n] = -1j * rbar[n + 1]
        for m in range(n, -1, -1):
            dr_dphi[n] = dr_dphi[n] * ephi[m] * (1 - r[m]**2) / (1 + r[m] * rbar[m + 1])**2
    
    # Shift rbar index
    rCoat = rbar[0]
    rbar = rbar[1:]
    
    # Phase derivatives
    dcdp = np.imag(dr_dphi / rCoat)
    
    return rCoat, dcdp, rbar, r


def getCoatAbsorption(lambda_, dOpt, aLayer, nLayer, rbar, r):
    """
    Returns coating absorption as a function of depth.

    Parameters:
    - lambda_ : wavelength
    - dOpt : optical thickness/lambda of each layer
             = geometrical thickness * refractive index/lambda
    - aLayer : absorption per unit length in each layer
    - nLayer : refractive index of each layer, ordered input to output (N x 1)
    - rbar : amplitude reflectivity of coating from this layer down
    - r : amplitude reflectivity of this interface (r[0] is nIn to nLayer[0])

    Returns:
    - rho : power ratio in each layer
    - absLayer : absorption contribution from each layer
    - absCoat : coating total absorption = sum(absLayer)
    """
    
    # Power in each layer
    powerLayer = np.cumprod(np.abs((1 - r[:-1]**2) / (1 + r[:-1] * rbar)**2))
    
    # One-way phases in each layer
    phi = 2 * np.pi * dOpt
    
    # Average E-field squared in each layer
    rho = (1 + np.abs(rbar)**2) + 2 * (np.sin(phi) / phi) * np.real(rbar * np.exp(1j * phi))
    
    # Geometrical thickness of each layer
    dGeo = lambda_ * dOpt / nLayer
    
    # Compute power weighting for each layer
    absLayer = aLayer * rho * powerLayer * dGeo
    
    # Total coating absorption
    absCoat = np.sum(absLayer)
    
    return absCoat, absLayer, powerLayer, rho


#functions used to Calculate Coating Thermal Noise 
# not to be used to calculate optical properties 

def getCoatRefl2(nIn, nOut, nLayer, dOpt):
    # Vector of all refractive indices
    nAll = np.concatenate(([nIn], nLayer, [nOut]))
    
    # Reflectivity of each interface
    r = (nAll[:-1] - nAll[1:]) / (nAll[:-1] + nAll[1:])
    
    # Combine reflectivities
    rbar = np.zeros_like(r, dtype=complex)
    ephi = np.zeros_like(r, dtype=complex)
    
    ephi[-1] = np.exp(-4j * np.pi * dOpt[-1])
    rbar[-1] = ephi[-1] * r[-1]
    
    for n in range(len(dOpt)-1, -1, -1):
        # Round-trip phase in this layer
        ephi[n] = np.exp(-4j * np.pi * dOpt[n - 1]) if n > 0 else 1
        
        # Accumulate reflectivity
        rbar[n] = ephi[n] * (r[n] + rbar[n + 1]) / (1 + r[n] * rbar[n + 1])
    
    # Reflectivity derivatives
    dr_dphi = np.zeros_like(dOpt, dtype=complex)
    
    for n in range(len(dOpt)-1, -1, -1):
        dr_dphi[n] = -1j * rbar[n + 1]
        for m in range(n, -1, -1):
            dr_dphi[n] = dr_dphi[n] * ephi[m] * (1 - r[m]**2) / (1 + r[m] * rbar[m + 1])**2
    
    # Shift rbar index
    rCoat = rbar[0]
    rbar = rbar[1:]
    
    # Phase derivatives
    dcdp = np.imag(dr_dphi / rCoat)
    
    return rCoat, dcdp, rbar, r


def getCoatAbsorption(light_wavelength, layer_thicknesses, layer_absorbtion, layer_refractive_indices, rbar, r):
    """
    Returns coating absorption as a function of depth.

    Parameters:
    - lambda_ : wavelength
    - layer_thicknesses :  thickness= of each layer
             = optical thickness * lambda/ refractive index
    - aLayer : absorption per unit length in each layer
    - nLayer : refractive index of each layer, ordered input to output (N x 1)
    - rbar : amplitude reflectivity of coating from this layer down
    - r : amplitude reflectivity of this interface (r[0] is nIn to nLayer[0])

    Returns:
    - rho : power ratio in each layer
    - absLayer : absorption contribution from each layer
    - absCoat : coating total absorption = sum(absLayer)
    """

    optical_thickness = layer_thicknesses*layer_refractive_indices/light_wavelength
    
    # Power in each layer
    powerLayer = np.cumprod(np.abs((1 - r[:-1]**2) / (1 + r[:-1] * rbar)**2))
    
    # One-way phases in each layer
    phi = 2 * np.pi * optical_thickness
    
    # Average E-field squared in each layer
    rho = (1 + np.abs(rbar)**2) + 2 * (np.sin(phi) / phi) * np.real(rbar * np.exp(1j * phi))
    
    # Compute power weighting for each layer
    absLayer = layer_absorbtion * rho * powerLayer * layer_thicknesses
    
    # Total coating absorption
    absCoat = np.sum(absLayer)
    
    return absCoat, absLayer, powerLayer, rho


def getCoatNoise2(f, light_wavelength, wBeam, Temp, material_parameters, substrate_index, layer_material_indices, layer_thicknesses, dcdp):
    """
    Returns coating noise as a function of depth.

    Parameters:
    - f : frequency
    - lambda_ : wavelength
    - wBeam : beam width
    - Temp : temperatur
    - materialParams : dictionary containing material properties
    - materialSub : substrate material
    - materialLayer : list of layer materials
    - dOpt : optical thickness / lambda of each layer
    - dcdp : phase derivatives

    Returns:
    - SbrZ, StoZ, SteZ, StrZ, brLayer
    """
    
    # Boltzmann constant and temperature
    kBT = 1.3807e-23 * Temp
    
    # Angular frequency
    w = 2 * np.pi * f
    
    # Substrate properties
    alphaSub = material_parameters[substrate_index]['alpha']
    cSub = material_parameters[substrate_index]['C']
    kappaSub = material_parameters[substrate_index]['kappa']
    ySub = material_parameters[substrate_index]['Y']
    pratSub = material_parameters[substrate_index]['prat']
    
    # Initialize vectors of material properties
    nN = np.zeros_like(layer_thicknesses)
    aN = np.zeros_like(layer_thicknesses)
    alphaN = np.zeros_like(layer_thicknesses)
    betaN = np.zeros_like(layer_thicknesses)
    kappaN = np.zeros_like(layer_thicknesses)
    cN = np.zeros_like(layer_thicknesses)
    yN = np.zeros_like(layer_thicknesses)
    pratN = np.zeros_like(layer_thicknesses)
    phiN = np.zeros_like(layer_thicknesses)
    
    for n, mat in enumerate(layer_material_indices):
        nN[n] = material_parameters[mat]['n']
        aN[n] = material_parameters[mat]['a']
        alphaN[n] = material_parameters[mat]['alpha']
        betaN[n] = material_parameters[mat]['beta']
        kappaN[n] = material_parameters[mat]['kappa']
        cN[n] = material_parameters[mat]['C']
        yN[n] = material_parameters[mat]['Y']
        pratN[n] = material_parameters[mat]['prat']
        phiN[n] = material_parameters[mat]['phiM']
    
    # Geometrical thickness of each layer and total
    dGeo = light_wavelength * layer_thicknesses / nN
    dCoat = np.sum(dGeo)
    
    # Brownian
    brLayer = ((1 + nN * dcdp / 2)**2 * (ySub / yN) + 
               (1 - pratSub - 2 * pratSub**2)**2 * yN / 
               ((1 + pratN)**2 * (1 - 2 * pratN) * ySub)) / (1 - pratN) * ((1 - pratN - 2 * pratN**2)) / ((1 - pratSub - 2 * pratSub**2))
    
    SbrZ = (4 * kBT / (np.pi * wBeam**2 * w)) * np.sum(dGeo * brLayer * phiN * (1 - pratSub - 2 * pratSub**2) / ySub)
    
    # Thermo-optic
    alphaBarSub = 2 * (1 + pratSub) * alphaSub
    
    alphaBar = (dGeo / dCoat) * ((1 + pratSub) / (1 - pratN)) * ((1 + pratN) / (1 + pratSub) + (1 - 2 * pratSub) * yN / ySub) * alphaN
    
    betaBar = (-dcdp) * layer_thicknesses * (betaN / nN + alphaN * (1 + pratN) / (1 - pratN))
    
    # Thermo-elastic
    SteZ = (4 * kBT * Temp / (np.pi * wBeam**2 * np.sqrt(2 * kappaSub * cSub * w))) * (np.sum(alphaBar * dCoat) - alphaBarSub * np.sum(dGeo * cN) / cSub)**2
    
    # Thermo-refractive
    StrZ = (4 * kBT * Temp / (np.pi * wBeam**2 * np.sqrt(2 * kappaSub * cSub * w))) * np.sum(betaBar * light_wavelength)**2
    
    # Total thermo-optic
    StoZ = (4 * kBT * Temp / (np.pi * wBeam**2 * np.sqrt(2 * kappaSub * cSub * w))) * (np.sum(alphaBar * dCoat) - np.sum(betaBar * light_wavelength) - alphaBarSub * np.sum(dGeo * cN) / cSub)**2
    
    return SbrZ, StoZ, SteZ, StrZ, brLayer

def getCoatingThermalNoise(
        layer_thicknesses, 
        layer_material_index, 
        material_parameters, 
        substrate_index=1, 
        air_index=0,
        light_wavelength=1, 
        f=1, 
        wBeam=1, 
        Temp=1,
        plots=True):
    """_summary_

    Args:
        dOpt (_type_): In quarter wavelengths
        materialLayer (_type_): _description_
        materialParams (_type_): _description_
        materialSub (int, optional): _description_. Defaults to 1.
        lambda_ (int, optional): _description_. Defaults to 1.
        f (int, optional): _description_. Defaults to 1.
        wBeam (int, optional): _description_. Defaults to 1.
        Temp (int, optional): _description_. Defaults to 1.
        plots (bool, optional): _description_. Defaults to True.
    """
    # Set seaborn style and viridis color palette

    # Extract substrate properties
    n_substrate = material_parameters[substrate_index]['n']
    y_substrate = material_parameters[substrate_index]['Y']
    prat_substrate = material_parameters[substrate_index]['prat']

    n_air = material_parameters[air_index]['n']
    y_air = material_parameters[air_index]['Y']
    prat_air = material_parameters[air_index]['prat']

    # Initialize vectors of material properties
   # Initialize vectors of material properties
    layer_refractive_indices = np.zeros(len(layer_material_index))
    layer_absorbtions = np.zeros(len(layer_material_index))
    
    for n, mat in enumerate(layer_material_index):
        layer_refractive_indices[n] = material_parameters[mat]['n']
        layer_absorbtions[n] = material_parameters[mat]['a']
    

    # Compute reflectivities
    rCoat, dcdp, rbar, r = getCoatRefl2(n_air, n_substrate, layer_refractive_indices, layer_thicknesses)
    #print(rCoat)

    # Compute absorption
    absCoat, absLayer, powerLayer, rho = getCoatAbsorption(light_wavelength, layer_thicknesses, layer_absorbtions, layer_refractive_indices, rbar, r)

    # Compute brownian and thermo-optic noises
    SbrZ, StoZ, SteZ, StrZ, brLayer = getCoatNoise2(f, light_wavelength, wBeam, Temp, material_parameters, substrate_index, layer_material_index, layer_thicknesses, dcdp)
    
    if plots ==True: 
        sns.set_style("whitegrid")
        sns.set_palette("tab10")
        # Plotting
        # Absorption values
        plt.figure()
        plt.semilogy(rho,'o')
        plt.semilogy(powerLayer,'o')
        plt.semilogy(rho * powerLayer)
        plt.legend(['rho_j', 'P_j / P_0', 'rho_bar_j'])
        plt.xlabel('Layer number')

        # Noise weights for each layer
        plt.figure()
        materials = np.unique(materialLayer)
        # Get a list of colors from the Seaborn viridis palette
        colors = sns.color_palette("viridis", n_colors=len(materials) + 2)  # +2 for the two additional plots


        for idx, i in enumerate(materials):
            matidx = np.where(materialLayer == i)[0]  # Extract the array from the tuple
            plt.bar(matidx, nLayer[matidx], color=colors[idx], label=materialParams[i]['name'])
        # Use the next color in the palette for the following plots
        plt.plot(-dcdp, 'o', color=colors[-2], markersize=10, label='-dphi_c / dphi_j')
        plt.plot(brLayer, 'o', color=colors[-1], markersize=10, label='b_j')
        plt.xlabel('Layer number')
        plt.legend()

        
        # Noise plots
        plt.figure()
        plt.loglog(f, np.sqrt(SbrZ), '--')
        plt.loglog(f, np.sqrt(StoZ))
        plt.loglog(f, np.sqrt(SteZ), '-.')
        plt.loglog(f, np.sqrt(StrZ), '-.')
        plt.legend(['Brownian Noise', 'Thermo-optic Noise', 'TE Component', 'TR Component'])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Thermal noise [m/sqrt(Hz)]')

        plt.show()

    # Return Noise Summary
    noise_summary = {
        'Frequency': f,
        'BrownianNoise': np.sqrt(SbrZ),
        'ThermoOptic': np.sqrt(StoZ),
        'ThermoElastic': np.sqrt(SteZ),
        'ThermoRefractive': np.sqrt(StrZ)
    }

    return noise_summary, rCoat, dcdp, rbar, r

