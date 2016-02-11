#!/usr/bin/python
"""
Give the altitude of the last obstruction in the given azimuth direction for every cell 
"""

import os
import sys
import gdal
import yaml
import numpy as np
from rpy2.robjects.packages import importr
from topocorr_funcs import gdal_save_grid

project_name = sys.argv[1]
# demfn = sys.argv[3]
azi   = float(sys.argv[2])
# horizon_out = sys.argv[4]

project_param_fn = '{0}/{0}_parameters.yaml'.format(project_name)
with file(project_param_fn) as f:
    project_parameters = yaml.safe_load(f)

demfn = project_parameters['dem']
horizon_out = os.path.join(project_parameters['dem_dir'],'horizon','horizon_{0}azi.tif'.format(azi))

r_raster = importr('raster')
r_insol = importr("insol")

dem_ds = gdal.Open(demfn,gdal.GA_ReadOnly)
dem_b = dem_ds.GetRasterBand(1)
pydem = dem_b.ReadAsArray()

rdem = r_raster.raster(demfn)

# Create horizon_grid
horizon = np.zeros_like(pydem)
altitude = 1
all_clear = horizon.all()
while not all_clear:
    sol_vector = r_insol.normalvector(90-altitude,azi)
    shd = r_insol.doshade(rdem,sol_vector)
    rpy2shd = np.array(r_raster.as_matrix(shd))
    horizon[(horizon==0)  & (rpy2shd==1)] = altitude
    print str(altitude)+':', int(rpy2shd.size - rpy2shd.sum())
    altitude+=1
    all_clear = horizon.all()
   
# write out horizon.
horizon_args = {'x_size': project_parameters['x_size'], 'y_size': project_parameters['y_size'],
                'ulx': project_parameters['xmin'], 'uly': project_parameters['ymax'],
                'epsg': project_parameters['epsg'], 'dx': project_parameters['dx'], 'dy': project_parameters['dy'],
                'outfn': horizon_out, 'from_dset': horizon}
gdal_save_grid(**horizon_args)
