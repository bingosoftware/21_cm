#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script plots in Mollweide projection the Healpix maps saved as fits files
and passed to it as arguments.

USAGE:   ViewMap.py <HEALPIX_MAP_1> <HEALPIX_MAP_2> ...
EXAMPLE: ViewMap.py euclid_footprint.fits

"""

import healpy as hp
import matplotlib.pyplot as plt
import sys
import pyfits

# Docstring output:
if len(sys.argv) < 1 + 1:
    print(__doc__)
    sys.exit(1)


for infile in sys.argv[1:]:
    #m = hp.read_map(infile)
    map = pyfits.getdata(infile)
    hp.mollview(map[1, :])
    plt.show()
