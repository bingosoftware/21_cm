import numpy as np
from astropy.io import fits as pyfits
import healpy as hp
import subprocess

from scipy import interpolate as sint

import hi_ps

################################################################################
################################################################################

##### Main function: calculates the HI map for each frequency channel (no correlation for now) - mK

# Inputs: freq_min -- minimum frequency in MHz
#         freq_width -- width of the frequency channels in MHz
#         nchannels -- number of channels
#         nside (lmax = 3 * nside - 1) -- nside of the Healpix maps
#         delta_ell -- the ell bin used to calculate the HI C_ell
#         suffix -- string for the file suffix -- string with the suffix of the output fits file
#
# Output: fits file named 'map_hi_' + suffix + '.fits' with an array a[nchannels, npixels], where npixel is the number of pixels of the maps. The maps are given in mK.
###

# Defining the experiment

freq_min = 960. # MHz
freq_width = 500. # MHz
nchannels = 2
nside = 128
delta_ell = 4. # it has to be an even number
suffix = 'test'

# Code

h = 0.6727

freq_half = freq_width / 2.0
channels = np.zeros(nchannels)
ind = np.arange(nchannels)
channels = freq_min + ind * freq_width

lmax = 3 * nside - 1
nsamples_ell = int((lmax + 1) / delta_ell + 1)
ind = np.arange(nsamples_ell)
ell_samples = delta_ell * ind
cl_int = np.zeros(nsamples_ell)
ell = np.arange(lmax + 1)
c_ell = np.zeros(lmax + 1)

# Calculating the power spectrum - should really call CAMB to calculate the power spectrum for a particular Cosmology

file_pk = "p_k_paper.txt"
file_k = "k_h_paper.txt"
k_calc = np.loadtxt(file_k)
k_calc = k_calc * h
pk_calc = np.loadtxt(file_pk)
pk_calc = pk_calc * h**(-3)
log_pk_int = sint.interp1d(np.log10(k_calc), np.log10(pk_calc), kind='cubic')

# Hubble parameter, comoving distance, HI temp, growth factor

samples = 11 # redshift samples - odd
maps = np.zeros((nchannels, hp.pixelfunc.nside2npix(nside)))
cls = np.zeros((nchannels, nchannels, lmax + 1))

for i in range (0, nchannels):
    z_min = (1420.4 / (channels[i] + freq_half)) - 1.0
    z_max = (1420.4 / (channels[i] - freq_half)) - 1.0
    z = np.linspace(z_min, z_max, samples, endpoint=True)

    h_input = hi_ps.hubble(z)
    cd_input = hi_ps.comoving_dist(z)
    t_input = hi_ps.temp_hi(z)
    gf_input = hi_ps.growth_factor(z)

    # Calculating the C_ell

    for m in range (0, nsamples_ell):
        l = delta_ell * m
        cl_int[m] = hi_ps.cl_limber(l, z_min, z_max, h_input, cd_input, t_input, gf_input, k_calc, pk_calc, log_pk_int)

        cl_func = sint.interp1d(ell_samples, cl_int, kind='cubic')
        c_ell = cl_func(ell)

    maps[i, :] = hp.sphtfunc.synfast(c_ell, nside)

    cls[i, i, :] = c_ell[:]

file_map = 'map_hi_2channels' + suffix + '.fits'
pyfits.writeto(file_map, maps, clobber=True)

file_map = "cls_hi21_theorical_2channels" + suffix + ".fits"
pyfits.writeto(file_map, cls, clobber=True)
