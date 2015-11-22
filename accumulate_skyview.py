#!/usr/bin/python

'''Define skyview using the shade grids for az=0:369 and zen=0:89

Not all angles are reasonable orientations for a sun, especially in Norway,
but when regarded as 'viewangles' they suddenly become very meaningful.
Each grid is then a 'view' test of whether that sector of the sky is
visible from each point in the grid.  The sum of visible angles is weighted
by cosine(90-zenith) to account for smaller sectors per degree
as zenith angle approaches noon. This weighted sum, divided by 
the maximum possible sum, gives the % sky viewable from each point in the grid'''

import yaml
import numpy as np
import os
import sys
from topocorr_funcs import gdal_save_grid


def sum_zen_ring(zen, gridmap):
    ringview = np.zeros_like(gridmap)
    zen_weight = np.cos(np.deg2rad(90-zen))
    max_sum = 360*zen_weight
    for az in range(360):
        bog = np.loadtxt(prj['BOG'].format(az, zen), skiprows=6, delimiter=' ')
        ringview+=bog*zen_weight
    return ringview, max_sum


prj_name = sys.argv[1] # $ python accumulate_skyview.py prjName

prj_param_fn = '{}_parameters.yaml'.format(prj_name)
with file(prj_param_fn) as f:
    prj = yaml.safe_load(f)

gridmap = np.loadtxt(prj['BOG'].format(0, 89), skiprows=6, delimiter=' ')

skyview = np.zeros_like(gridmap)
max_sum = np.zeros_like(gridmap)
outfn = os.path.join(prj['dem_dir'],prj_name + '_skyview.asc')
zen_over_horizon = 90-prj['steepest_slope']
for zen in range(1,90):
    if zen < zen_over_horizon:
        print zen,'above highest possible horizon'
        zen_weight = np.cos(np.deg2rad(90-zen))
        skyview += np.ones_like(gridmap)*360*zen_weight
        max_sum += np.ones_like(gridmap)*360*zen_weight
    else:
        print zen, 'loading BOGs'
        skyview_ring,max_sum_ring = sum_zen_ring(zen,gridmap)
        skyview += skyview_ring
        max_sum += max_sum_ring
        #for az in range(0,360):
        #    skyview+=np.loadtxt(prj['BOG'].format(az,zen),skiprows=6,delimiter=' ') * zen_weight

# Normalize skyview (0:1)
skyview_out = skyview / max_sum


# Save out skyview
skyview_args = {'ncols': prj['x_size'], 'nrows': prj['y_size'],
                       'ulx': prj['xmin'], 'uly': prj['ymax'], 'epsg': prj['epsg'],
                       'cellsize': prj['dx'], 'out_ds': outfn, 'from_dset': skyview_out}
gdal_save_grid(**skyview_args)
