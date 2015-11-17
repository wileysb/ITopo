#!/usr/bin/python

__author__ = 'Wiley Bogren'

import subprocess
import shlex
import os
from topocorr_funcs import prj_mkdir
from example_parameters import *

# Create dirs
dem_dir: os.path.join(prj_dir, 'dem_derivs/')
BOG_dir: os.path.join(dem_dir, 'BOGs')

prj_mkdir(prj_dir)
prj_mkdir(dem_dir)
prj_mkdir(BOG_dir)

# Clip DEM
prj_dem = os.path.join(dem_dir,prj_name+'_dem.tif')
if not os.path.isfile(prj_dem):
    gdalwarp_cmd = 'gdalwarp -t_srs "EPSG:{0}" -te {1} {2} {3} {4} -r cubic -of GTiff {5} {6}'.format(prj_epsg,
                    prj_extents['xmin'],prj_extents['ymin'],prj_extents['xmax'],prj_extents['ymax'],
                    src_dem,prj_dem)
    args = shlex.split(gdalwarp_cmd)
    p = subprocess.Popen(args)  # todo test this!

# Slope and Aspect


###  DEFINE GEOTRANSFORMS
# Replace these dicts with methods for deriving them from SRB and DEM extents:
dem_gt   = { 'outfn':None,  # utm33n_md
                'from_dset':None,  # [y,x] array or gdal dataset
                'epsg':32633,
                'x_size':1210,
                'y_size':1532,
                'dx':1000,
                'dy':-1000,
                'ulx':-84500.00,
                'uly':7961500.0}

srb_gt =     { 'outfn': 'MEM',  # wgs84lo
               'from_dset': None,
               'epsg': 4326,
               'x_size': 28,  # srb_3hr_vars['diffuse'].shape[2],
               'y_size': 15,  # srb_3hr_vars['diffuse'].shape[1],
               'dx': 1,
               'dy': -1,
               'ulx': 4.5,
               'uly': 72.5}


srb_hi_gt = srb_gt.copy()
srb_hi_gt['dx'] = 0  # 'dx':0.008333333333333333 todo solve this
srb_hi_gt['dy'] = 0  # 'dy':0.008333333333333333 todo solve this
srb_hi_gt['x_size'] = 0  # 'x_size':srb_gt['x_size']*120 todo solve this
srb_hi_gt['y_size'] = 0  # 'y_size':srb_gt['y_size']*120 todo solve this
