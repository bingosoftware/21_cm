import numpy as np
from astropy.io import fits as pyfits
import healpy as hp
import subprocess

from scipy import interpolate as sint

# Import integration routines
from scipy import integrate as si
# Import special functions
from scipy import special as ss
# Import curve fitting
from scipy.optimize import curve_fit
from scipy import interpolate as sint

import pdb

################################################################################
################################################################################

##### COSMOLOGY

### Speed of light - km / s

c = 3.0 * 10**5

### Universe components - current time - flat Universe 

omega_b = 0.04915 
omega_c = 0.2647
omega_nu = 0.001421

omega_d = 1. - omega_b - omega_c - omega_nu
omega_m = omega_b + omega_c

omega_k = 0.0

### Dark energy parameters

w_0 = -1.0
w_a =  0.0

### Hubble parameters - km/s/Mpc

h = 0.6727
H_0 = 100.0 * h

### HI parameter - constant

omega_hi = 6.2 * 10**(-4)
omega_hi_h = omega_hi * h
bias = 1.0

n_0 = 0.03 * h**3

### CAMB parameters

# ns = 0.9641
# ln 1.e10 * As = 3.096
# tau = 0.081
# yhe = 0.2453

################################################################################
################################################################################

##### Redshift Functions

### Hubble parameter: E(z) (dimensionless)

def hubble(z): 
    
    return ((2./3.) * omega_nu * (1.0 + z)**4 + (1./3.) * omega_nu * (1.0 + z)**3 + omega_m * (1.0 + z)**3 + omega_k * (1.0 + z)**2 + omega_d * (1.0 + z)**(3.0 * (1.0 + w_0 + w_a)) * np.exp(-3.0 * w_a * z / (1 + z)))**0.5

### 21 cm background temperature - mK

def temp_hi(z):

    omega_hi_h_redshift = omega_hi_h

    return 44.0 * 10**(-3) * (omega_hi_h_redshift / (2.45 * 10**(-4))) * ((1.0 + z)**2 / hubble(z))

### Comoving distance = dimensionless

def comoving_dist(z):

    if isinstance(z, np.ndarray) == True:
        samples = z.size
        comoving_dist = np.zeros(samples)

        for i in range(0, samples):

            fz = lambda x: 1.0 / ((2. / 3.) * omega_nu * (1.0 + x)**4 + (1. / 3.) * omega_nu * (1.0 + x)**3 + omega_m * (1.0 + x)**3 + omega_k * (1.0 + x)**2 + omega_d * (1.0 + x)**(3 * (1.0 + w_0 + w_a)) * np.exp(-3 * w_a * x / (1 + x)))**0.5

            comoving_dist[i] = (c / H_0) * si.romberg(fz, 0, z[i])
    else:

        fz = lambda x: 1.0 / ((2. / 3.) * omega_nu * (1.0 + x)**4 + (1. / 3.) * omega_nu * (1.0 + x)**3 + omega_m * (1.0 + x)**3 + omega_k * (1.0 + x)**2 + omega_d * (1.0 + x)**(3 * (1.0 + w_0 + w_a)) * np.exp(-3 * w_a * x / (1 + x)))**0.5

        comoving_dist = (c / H_0) * si.romberg(fz, 0, z)

    return comoving_dist

### Growth factor for a spatially flat cosmology with dust and a cosmological constant (w const or w = w_0 + (z / 1 + z) * w_a)

def deriv(D, a):

    derivative = np.array([D[1], -1.0 * (3. * a**(-1) + 0.5 * (1. / ((2. / 3.) * omega_nu * a**(-4) + (1. / 3.) * omega_nu * a**(-3) + omega_m * a**(-3) + omega_d * a**(-3 * (1 + w_0 + w_a)) * np.exp(-3. * w_a * (1 - a)))) * (-4. * (2. / 3.) * omega_nu * a**(-5) - 3. * (1. / 3.) * omega_nu * a**(-4) - 3. * omega_m * a**(-4) - 3. * (1 + w_0 + w_a) * omega_d * a**(-3 * (1 + w_0 + w_a) - 1) * np.exp(-3. * w_a * (1 - a)) + 3. * w_a * omega_d * a**(-3 * (1 + w_0 + w_a)) * np.exp(-3. * w_a * (1 - a)))) * D[1] + 1.5 * omega_m * a**(-5) * ((2. / 3.) * omega_nu * a**(-4) + (1. / 3.) * omega_nu * a**(-3) + omega_m * a**(-3) + omega_d * a**(-3 * (1 + w_0 + w_a)) * np.exp(-3. * w_a * (1 - a)))**(-1) * D[0]])

    return derivative

def growth_factor(z):

    a = 1. /(1 + z)

    if isinstance(a, np.ndarray) == True:
        samples = a.size
        growth_factor_norm = np.zeros(samples)

        for i in range(0, samples):           
            N = 100
            if a[i] != 1.0:
                scale_factor = np.linspace(1.e-1, a[i], N)
                scale_factor_0 = np.linspace(1.e-1, 1, N)
                growth_initial = np.array([1.e-1, 1])
                growth_factor = si.odeint(deriv, growth_initial, scale_factor)
                growth_factor_0 = si.odeint(deriv, growth_initial, scale_factor_0)
                growth_factor_0 = growth_factor_0[N - 1, 0]
                growth_factor_z = growth_factor[N - 1, 0]
                growth_factor_norm[i] = growth_factor_z / growth_factor_0
            else:
                growth_factor_norm[i] = 1.0
    else:
        N = 100
        if a != 1.0:
            scale_factor = np.linspace(1.e-1, a, N)
            scale_factor_0 = np.linspace(1.e-1, 1, N)
            growth_initial = np.array([1.e-1, 1])   
            growth_factor = si.odeint(deriv, growth_initial, scale_factor)
            growth_factor_0 = si.odeint(deriv, growth_initial, scale_factor_0)
            growth_factor_0 = growth_factor_0[N - 1, 0]
            growth_factor_z = growth_factor[N - 1, 0]
            growth_factor_norm = growth_factor_z / growth_factor_0
        else:
            growth_factor_norm = 1.0

    return growth_factor_norm


def growth_factor_not_normalized(z):

    a = 1. /(1 + z)

    if isinstance(a, np.ndarray) == True:
        samples = a.size
        growth_factor_norm = np.zeros(samples)

        for i in range(0, samples):           
            N = 100
            scale_factor = np.linspace(1.e-1, a[i], N)
            growth_initial = np.array([1.e-1, 1])
            growth_factor = si.odeint(deriv, growth_initial, scale_factor)
            growth_factor_z = growth_factor[N - 1, 0]
            growth_factor_norm[i] = growth_factor_z #/ growth_factor_0
            
    else:
        N = 100
        scale_factor = np.linspace(1.e-1, a, N)
        growth_initial = np.array([1.e-1, 1])   
        growth_factor = si.odeint(deriv, growth_initial, scale_factor)
        growth_factor_z = growth_factor[N - 1, 0]
        growth_factor_norm = growth_factor_z #/ growth_factor_0

    return growth_factor_norm

################################################################################
################################################################################

##### Power Spectrum

### Power spectrum (calculated with CAMB) - Output in h/Mpc

def func_1(x, a, b, c):

    # interpolation function: quadratic

    return a + b*x + c*x**2

def func_2(x, a, b):

    return a + b * x

def power_spectrum(k_input, k_calc, pk_calc, log_pk_int):

    index = np.where(k_calc <= k_input)
    if np.array(index).size >= 1:
        index = (index[-1])[-1]
    else:
        index = 0

    nk = k_calc.size

    k_high = k_calc[nk - 5:nk]
    pk_high = pk_calc[nk - 5:nk]
    par_high, pcov_high = curve_fit(func_2, np.log10(k_high), np.log10(pk_high))

    k_low = k_calc[0:3]
    pk_low = pk_calc[0:3]
    par_low, pcov_low = curve_fit(func_2, np.log10(k_low), np.log10(pk_low))

    if k_input > 10.:
        p_out = 10**par_high[0] * k_input**par_high[1]
    elif index == 0 or index == 1 or index ==2 :
        p_out = 10**par_low[0] * k_input**par_low[1]
    else:
        p_out = 10**(log_pk_int(np.log10(k_input)))

    return p_out

################################################################################
################################################################################

##### Angular Power Spectrum

### Cl using Limber approximation        
    
def cl_limber(l, z_min, z_max, h_input, cd_input, t_input, gf_input, k_calc, pk_calc, log_pk_int):

    if l == 0:
        l = 1.0e-3
    
    windows = 1.0 / (z_max - z_min)

    N = h_input.size # samples - odd
    z = np.linspace(z_min, z_max, N, endpoint=True)

    integrand_array = np.zeros(N)

    for i in range (0, N):
        integrand_array[i] = h_input[i] * t_input[i]**2 * gf_input[i]**2 * (1.0 / (cd_input[i])**2) * power_spectrum((l + 0.5) / cd_input[i], k_calc, pk_calc, log_pk_int)

    cl_limber = ((H_0 * bias**2) / c) * windows**2 * si.simps(integrand_array, z)

    return cl_limber

def cl_shoot(z_min, z_max, h_input, cd_input, t_input):

    windows = 1.0 / (z_max - z_min)

    N = h_input.size # samples - odd
    z = np.linspace(z_min, z_max, N, endpoint=True)

    return (windows * si.simps(t_input, z))**2 / (((n_0 * c) / H_0) * si.simps(cd_input**2 / h_input, z))





        
