#!/usr/bin/python

"""Define skyview using the shade grids for az=0:369 and zen=0:89

Not all angles are reasonable orientations for a sun, especially in Norway,
but when regarded as 'viewangles' they suddenly become very meaningful.
Each grid is then a 'view' test of whether that sector of the sky is
visible from each point in the grid.  The sum of visible angles is weighted
by cosine(90-zenith) to account for smaller sectors per degree
as zenith angle approaches noon. This weighted sum, divided by the maximum 
possible sum, gives the percent sky viewable from each point in the grid"""

import yaml
import gdal
import numpy as np
import os
import sys
from topocorr_funcs import gdal_save_grid


def sum_zen_ring(zen, gridmap):
    ringview = np.zeros_like(gridmap)
    zen_weight = np.cos(np.deg2rad(90-zen))
    max_sum = 360*zen_weight
    for az in range(360):
        bog = np.loadtxt(project_parameters['BOG'].format(az, zen), skiprows=6, delimiter=' ')
        ringview += bog*zen_weight
    return ringview, max_sum


project_name = sys.argv[1]  # $ python accumulate_skyview.py projectName

project_param_fn = '{0}/{0}_parameters.yaml'.format(project_name)
with file(project_param_fn) as f:
    project_parameters = yaml.safe_load(f)

gridmap = np.loadtxt(project_parameters['BOG'].format(0, 89), skiprows=6, delimiter=' ')

skyview = np.zeros_like(gridmap)
max_sum = np.zeros_like(gridmap)
outfn = project_parameters['sky']
zen_over_horizon = 90-project_parameters['steepest_slope']
print 'Accumulating skyview, highest possible horizon:', project_parameters['steepest_slope']
for zen in range(1, 90):
    if zen < zen_over_horizon:
        zen_weight = np.cos(np.deg2rad(90-zen))
        skyview += np.ones_like(gridmap)*360*zen_weight
        max_sum += np.ones_like(gridmap)*360*zen_weight
    else:
        skyview_ring, max_sum_ring = sum_zen_ring(zen, gridmap)
        skyview += skyview_ring
        max_sum += max_sum_ring
    gdal.TermProgress_nocb(zen/float(90))

# Normalize skyview (0:1)
skyview_out = skyview / max_sum


# Save out skyview
# from_dset, outfn, epsg, x_size, y_size, ulx, uly, dx, dy
skyview_args = {'x_size': project_parameters['x_size'], 'y_size': project_parameters['y_size'],
                'ulx': project_parameters['xmin'], 'uly': project_parameters['ymax'],
                'epsg': project_parameters['epsg'], 'dx': project_parameters['dx'], 'dy': project_parameters['dy'],
                'outfn': outfn, 'from_dset': skyview_out}
gdal_save_grid(**skyview_args)
gdal.TermProgress_nocb(100)