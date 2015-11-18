#!/usr/bin/python
'''
TODO:
* Init Paths
 - path to DEM
 - DEM derivatives directory (default DEM directory)
 - SRB directory
 - temporary timestep and processing output directory

* Init spatial parameters
 - define geotransforms for:
  + [x] Source (srb, WGS84 lo res)
  + [x] Intermediate, WGS84 high res
  + [x] Output, from DEM (must be rectilinear/slope- & aspect-able

* Init temporal parameters?
 - months and years to cover?  default 1984-2007, all 12 months? Possibly useful to run 'just April and June' or something?

* Prepare DEM derivatives:
 - Slope and Aspect (gdaldem slope, gdaldem aspect -zero_when_flat)
  + [x] gdaldem slope, gdaldem aspect -zero_for_flat
 - Lat and Lon
  + [ ]  Should be a simple function based on geotransforms
 - Sunview and Skyview
  + [ ]  Binary Obstruction Grids (no obstruction when solar zenith is higher than max slope angle)
  + [ ]  accumulate Skyview
  + [ ]  pre-stack a year's worth of 3hr Sunview grids, or just save them if they aren't saved yet?
'''
__author__ = 'Wiley Bogren'

import os
import sys
import yaml
import math
import numpy as np
from topocorr_funcs import Prj_mkdir, Get_gt_dict, transform_epsg2epsg, t_xy, mk_latlon_grids

prj_name = sys.argv[1] # $ python start_project.py prjName
prj_param_fn = '{}_parameters.yaml'.format(prj_name)

if os.path.isfile(prj_param_fn.format(prj_name)):
    print prj_param_fn.format(prj_name), 'exists.  To start over, delete or rename the existing file.'
else:
    ### LOAD PROJECT PARAMETERS ###
    with file('parameters.yaml','r') as f:
        parameters = yaml.safe_load(f)
    if prj_name in parameters.keys():
        prj = parameters[prj_name]
        prj['tmp_dir'] = os.path.join(prj['itopo_dir'], 'tmp/')
        prj['dem_dir'] = os.path.join(prj['itopo_dir'], 'dem_derivs/')
        prj['BOG_dir'] = os.path.join(prj['dem_dir'], 'BOGs')

        ### CREATE DIRS, IF NECESSARY
        Prj_mkdir(prj['itopo_dir'])
        Prj_mkdir(prj['tmp_dir'])
        Prj_mkdir(prj['dem_dir'])
        Prj_mkdir(prj['BOG_dir'])

        prj['init_cmds'] = []


        ### Clip DEM ###
        prj['dem'] = os.path.join(prj['dem_dir'],prj_name+'_dem.tif')
        if os.path.isfile(prj['dem']):
            prj['dem_gt']   = Get_gt_dict(prj['dem'])
            prj['epsg'] =  prj['dem_gt']['epsg']
            prj['dx'] = prj['dem_gt']['dx']
            prj['dy']= prj['dem_gt']['dy']
            prj['xmin']= prj['dem_gt']['ulx']
            prj['ymax']= prj['dem_gt']['uly']
            prj['ymin']= prj['ymax'] + prj['dy']*prj['dem_gt']['y_size']
            prj['xmax']= prj['xmin'] + prj['dx']*prj['dem_gt']['x_size']
        else:
            gdalwarp_cmd = 'gdalwarp -t_srs "EPSG:{0}" -tr {1} {2} -te {3} {4} {5} {6} -r cubic -of GTiff {7} {8}'.format(
                            prj['epsg'],prj['dx'],np.abs(prj['dy']),
                            prj['xmin'],prj['ymin'],prj['xmax'],prj['ymax'],
                            prj['src_dem'],prj['dem'])
            prj['init_cmds'].append(gdalwarp_cmd)

            # set xmin,ymin,xmax,ymax,dx,dy from same
        # Slope and Aspect
        prj['asp'] = os.path.join(prj['dem_dir'],prj_name+'_asp.tif')
        prj['slp'] = os.path.join(prj['dem_dir'],prj_name+'_slp.tif')
        slp_cmd = 'gdaldem slope {0} {1}'.format(prj['dem'],prj['slp'])
        asp_cmd = 'gdaldem aspect -zero_for_flat {0} {1}'.format(prj['dem'],prj['asp'])
        prj['init_cmds'].append(slp_cmd)
        prj['init_cmds'].append(asp_cmd)

        for cmd in prj['init_cmds']:
            print cmd
            ret = os.system(cmd)

        prj['lat'] = os.path.join(prj['dem_dir'],prj_name+'_lat.tif')
        prj['lon'] = os.path.join(prj['dem_dir'],prj_name+'_lon.tif')
        #mk_latlon_grids(ncols,nrows,xll,yll,cellsize, out_ds)
        latlon_args = {
                        'ncols': prj['x_size'],'nrows':prj['y_size'],
                        'ulx':prj['ulx'],'uly':prj['uly'],
                        'cellsize':prj['dx'],'out_ds':os.path.join(prj['dem_dir'],prj_name)}
        mk_latlon_grids(**latlon_args)

        ###  DEFINE GEOTRANSFORMS
        if not 'dem_gt' in prj.keys():
            prj['dem_gt']   = Get_gt_dict(prj['dem'])
        # todo NOTE:
        #  in the hardcoded utm33n_md, 'ulx' and 'uly' off by 500m (pixel corners vs pixel center)
        #   'ulx':-84500.00 hardcoded, vs  -85000.0  from function
        #   'uly':7961500.0 hardcoded, vs  7962000.0 from function

        # Get srb bounding ulx and uly + x_size, y_size
         # This is only OK as long as corners from prj_extents form geographic max + min values
        t = transform_epsg2epsg(int(prj['epsg']),4326)
        srb_xmin,srb_ymax = t_xy(t,prj['xmin'],prj['ymax'])
        srb_xmax,srb_ymin = t_xy(t,prj['xmax'],prj['ymin'])

        # expand to integers
        srb_ymin = math.floor(srb_ymin)
        srb_xmin = math.floor(srb_xmin)
        srb_ymax = math.ceil(srb_ymax)
        srb_xmax = math.ceil(srb_xmax)

        srb_ulx, srb_uly = srb_xmin,srb_ymax
        srb_x_size = int((srb_xmax - srb_xmin) / prj['dx'])
        srb_y_size = int((srb_ymax - srb_ymin)/ prj['dy'])

        # If the input radiation is anything but srb, lots more will have to be recoded
        prj['srb_gt'] =     { 'outfn': 'MEM',  # wgs84lo
                              'from_dset': None,
                              'epsg': 4326,
                              'x_size': srb_x_size,
                              'y_size': srb_y_size,
                              'dx': 1,
                              'dy': -1,
                              'ulx': srb_ulx,
                              'uly': srb_uly}

        # 1 degree = 3600 arc seconds.  1 arc second ~ 30m. 1000m output used 30 arc seconds = zoom 120
        prj['srb_zoom_factor'] = int(108000 / prj['dem_gt']['dx'])
        prj['srb_hi_gt'] = prj['srb_gt'].copy()

        prj['srb_hi_gt']['dx'] = float(prj['srb_gt']['dx'])/prj['srb_zoom_factor']  # 1 / 120 = 0.008333
        prj['srb_hi_gt']['dy'] = float(prj['srb_gt']['dy'])/prj['srb_zoom_factor']  # 1 / 120 = 0.008333
        prj['srb_hi_gt']['x_size'] = prj['srb_gt']['x_size'] * prj['srb_zoom_factor']
        prj['srb_hi_gt']['y_size'] = prj['srb_gt']['y_size'] * prj['srb_zoom_factor']

        with file(prj_param_fn,'w') as f:
                yaml.safe_dump(prj,f)
    else:
        print prj_name,'not found in parameters.yaml'



