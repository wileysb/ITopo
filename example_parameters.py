#!/usr/bin/python

import os

### DEFINE PATHS TO IMPORTANT DIRECTORIES ###
dem_fn = '/path/to/dem.tif' 
srb_dir = '/path/to/srb/'
dem_derivs_dir = os.path.split(dem_fn)[0] # Unless you want the DEM to be somewhere else
BOG_dir = os.path.join(dem_derivs_dir,'BOGs')

###  DEFINE GEOTRANSFORMS
# Replace these dicts with methods for deriving them from SRB and DEM extents:
dem_gt   = { 'outfn':None, # utm33n_md
                'from_dset':None, # [y,x] array or gdal dataset
                'epsg':32633,
                'x_size':1210,
                'y_size':1532,
                'dx':1000,
                'dy':-1000,
                'ulx':-84500.00,
                'uly':7961500.0}

srb_gt =     { 'outfn':'MEM', # wgs84lo
                'from_dset':None,
                'epsg':4326,
                'x_size':28, #srb_3hr_vars['diffuse'].shape[2],
                'y_size':15,#srb_3hr_vars['diffuse'].shape[1],
                'dx':1,
                'dy':-1,
                'ulx':4.5,
                'uly':72.5}

srb_hi_gt =     { 'outfn':'MEM',  # wgs84hi
                'from_dset':None,
                'epsg':4326,
                'x_size':srb_gt['x_size']*120, #srb_3hr_vars['diffuse'].shape[2]*120,  TODO this should be adapted for 10-30m DEM/output, rather than as currently for 1km DEM/output
                'y_size':srb_gt['y_size']*120, #srb_3hr_vars['diffuse'].shape[1]*120,
                'dx':0.008333333333333333,
                'dy':-0.008333333333333333,
                'ulx':4.5,
                'uly':72.5}