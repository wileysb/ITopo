#!/usr/bin/python

'''
* Read project slope to get highest possible solar elevation (lowest possible solar zenith) where the
  topography might shade the sun.  If there are no slopes steeper than 20 degrees, for example, there will be no
  topographic shade anywhere on the map when the sun is more than 20 degrees above the horizon.

* for azimuth(0:360)
    for zenith(90:(90-maxSlope)
      Rscript mk_shade_grid.R prj_dem.tif zenith azimuth outfile.asc (tif possible??)
'''

import os
import sys
import yaml
import math
import numpy as np

from topocorr_funcs import gdal_load

prj_name = sys.argv[1] # $ python start_project.py prjName
prj_param_fn = '{}_parameters.yaml'.format(prj_name)

with file(prj_param_fn,'r') as f:
    prj = yaml.safe_load(f)

slope = gdal_load(prj['dem'])