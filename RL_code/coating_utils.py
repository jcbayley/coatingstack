import numpy as np

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


def getCoatNoise2(f, lambda_, wBeam, Temp, materialParams, materialSub, materialLayer, dOpt, dcdp):
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
    alphaSub = materialParams[materialSub]['alpha']
    cSub = materialParams[materialSub]['C']
    kappaSub = materialParams[materialSub]['kappa']
    ySub = materialParams[materialSub]['Y']
    pratSub = materialParams[materialSub]['prat']
    
    # Initialize vectors of material properties
    nN = np.zeros_like(dOpt)
    aN = np.zeros_like(dOpt)
    alphaN = np.zeros_like(dOpt)
    betaN = np.zeros_like(dOpt)
    kappaN = np.zeros_like(dOpt)
    cN = np.zeros_like(dOpt)
    yN = np.zeros_like(dOpt)
    pratN = np.zeros_like(dOpt)
    phiN = np.zeros_like(dOpt)
    
    for n, mat in enumerate(materialLayer):
        nN[n] = materialParams[mat]['n']
        aN[n] = materialParams[mat]['a']
        alphaN[n] = materialParams[mat]['alpha']
        betaN[n] = materialParams[mat]['beta']
        kappaN[n] = materialParams[mat]['kappa']
        cN[n] = materialParams[mat]['C']
        yN[n] = materialParams[mat]['Y']
        pratN[n] = materialParams[mat]['prat']
        phiN[n] = materialParams[mat]['phiM']
    
    # Geometrical thickness of each layer and total
    dGeo = lambda_ * dOpt / nN
    dCoat = np.sum(dGeo)
    
    # Brownian
    brLayer = ((1 + nN * dcdp / 2)**2 * (ySub / yN) + 
               (1 - pratSub - 2 * pratSub**2)**2 * yN / 
               ((1 + pratN)**2 * (1 - 2 * pratN) * ySub)) / (1 - pratN) * ((1 - pratN - 2 * pratN**2)) / ((1 - pratSub - 2 * pratSub**2))
    
    SbrZ = (4 * kBT / (np.pi * wBeam**2 * w)) * np.sum(dGeo * brLayer * phiN * (1 - pratSub - 2 * pratSub**2) / ySub)
    
    # Thermo-optic
    alphaBarSub = 2 * (1 + pratSub) * alphaSub
    
    alphaBar = (dGeo / dCoat) * ((1 + pratSub) / (1 - pratN)) * ((1 + pratN) / (1 + pratSub) + (1 - 2 * pratSub) * yN / ySub) * alphaN
    
    betaBar = (-dcdp) * dOpt * (betaN / nN + alphaN * (1 + pratN) / (1 - pratN))
    
    # Thermo-elastic
    SteZ = (4 * kBT * Temp / (np.pi * wBeam**2 * np.sqrt(2 * kappaSub * cSub * w))) * (np.sum(alphaBar * dCoat) - alphaBarSub * np.sum(dGeo * cN) / cSub)**2
    
    # Thermo-refractive
    StrZ = (4 * kBT * Temp / (np.pi * wBeam**2 * np.sqrt(2 * kappaSub * cSub * w))) * np.sum(betaBar * lambda_)**2
    
    # Total thermo-optic
    StoZ = (4 * kBT * Temp / (np.pi * wBeam**2 * np.sqrt(2 * kappaSub * cSub * w))) * (np.sum(alphaBar * dCoat) - np.sum(betaBar * lambda_) - alphaBarSub * np.sum(dGeo * cN) / cSub)**2
    
    return SbrZ, StoZ, SteZ, StrZ, brLayer
