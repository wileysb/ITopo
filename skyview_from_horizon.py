#!/usr/bin/python

"""
* Run after 'find_horizon.py'
* Produces a set of 360 raster grids, each giving the altitude of the last obstruction in the given azimuth direction for every cell
  (ie, the degrees above horizontal where the sky or sun is first visible instead of topography)

The sum of visible angles is weighted
by cosine(90-zenith) to account for smaller sectors per degree
as zenith angle approaches noon. This weighted sum, divided by the maximum 
possible sum, gives the percent sky viewable from each point in the grid"""

import yaml
import gdal
import numpy as np
import os
import sys
from topocorr_funcs import gdal_save_grid, gdal_load

project_name = sys.argv[1]  # $ python accumulate_skyview.py projectName

project_param_fn = '{0}/{0}_parameters.yaml'.format(project_name)
with file(project_param_fn) as f:
    project_parameters = yaml.safe_load(f)

horizon_fmt = os.path.join(project_parameters['dem_dir'],'horizon','horizon_{0}azi.tif')#.format(azi)
outfn = os.path.join(project_parameters['sky'])

def sum_azi_slice(horizon_grid):
    skyview_slice = np.zeros_like(horizon_grid)
    for altitude in range(1,90):
        skyview_slice += np.where(horizon_grid<=altitude, np.cos(np.deg2rad(altitude)), 0)
    return skyview_slice


slice_max = np.cos(np.deg2rad(np.arange(1,90))).sum()

horizon = gdal_load(horizon_fmt.format(0.0))
skyview = sum_azi_slice(horizon)/slice_max
for azimuth in range(1,360):
    horizon = gdal_load(horizon_fmt.format(float(azimuth)))
    skyview += sum_azi_slice(horizon)/slice_max

skyview = skyview / 360.
skyview_args = {'x_size': project_parameters['x_size'], 'y_size': project_parameters['y_size'],
                'ulx': project_parameters['xmin'], 'uly': project_parameters['ymax'],
                'epsg': project_parameters['epsg'], 'dx': project_parameters['dx'], 'dy': project_parameters['dy'],
                'outfn': outfn, 'from_dset': skyview}
gdal_save_grid(**skyview_args)

