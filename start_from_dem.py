#!/usr/bin/python
"""
* Script called with project name and path to DEM
* Loads project parameters
* Creates output directories for both final and temporary products
* Loads spatial parameters from DEM: projection, extent, and resolution
* Creates Slope, Aspect, Lat, and Lon raster derivatives from the elevation model
* Generates syntax for further processing commands based on project parameters, DEM spatial parameters, and directory structure
* Prints initial commands to standard out, and writes parallel-processing and sequential commands to file


* Init Paths
 - path to DEM
 - DEM derivatives directory
 - SRB directory
 - horizon directory
 - temporary timestep and processing directory
* Init spatial parameters
 - Define geotransforms for:
 -- Source(srb,WGS84 lo res)
 -- Intermediate, wGS84 high res
 -- Output, rectilinear projection from DEM
* Init Temporal Parameters
 - define start:end years and months to iterate over
* Prepare DEM derivatives
 - Slope and Aspect
 - Lat and Lon
 - Sunview and Skyview
 -- horizons
 -- accumulate skyview
"""

import os
import sys
import yaml
import math
import shutil
import numpy as np
from topocorr_funcs import Safe_mkdir, Get_gt_dict, transform_epsg2epsg, t_xy, mk_latlon_grids, gdal_load

__author__ = 'Wiley Bogren'

project_name = sys.argv[1]  # $ python start_project.py projectName path/to/dem.tif|.asc|.dem
dem_fn = sys.argv[2]
project_param_fn = '{0}/{0}_parameters.yaml'.format(project_name)

if os.path.isfile(project_param_fn.format(project_name)):
    print project_param_fn.format(project_name), 'exists.  To start over, delete or rename the existing file.'
else:
    ### LOAD PROJECT PARAMETERS ###
    with file('parameters.yaml', 'r') as f:
        parameters = yaml.safe_load(f)

    project_parameters = {}
    for key in ['srb_dir', 'first_year', 'last_year', 'months']:
        project_parameters[key] = parameters[key]
    project_parameters['itopo_dir'] = os.path.join(parameters['itopo_dir'], project_name)
    project_parameters['tmp_dir'] = os.path.join(project_parameters['itopo_dir'], 'tmp/')
    project_parameters['dem_dir'] = os.path.join(project_parameters['itopo_dir'], 'dem_derivs/')
    project_parameters['dem'] = os.path.join(project_parameters['dem_dir'], project_name+'_dem.tif')
    project_parameters['sky'] = os.path.join(project_parameters['dem_dir'], project_name+'_skyview.tif')
    project_parameters['hor_dir'] = os.path.join(project_parameters['dem_dir'], 'horizon')
    project_parameters['out_dir'] = os.path.join(project_parameters['itopo_dir'], 'monthly_means')
    project_parameters['hor'] = os.path.join(project_parameters['hor_dir'], 'horizon_{0}azi.tif')

    project_parameters['srb'] = os.path.join(project_parameters['srb_dir'], 'srb_rel3.0_shortwave_3hrly_{0}{1}.nc')
    # .format(yyyy, mm)

    # These are probably not used:
    sunview_dir = os.path.join(project_parameters['dem_dir'], 'sunview')
    project_parameters['sunview'] = os.path.join(sunview_dir, 'sunview_{0}_{1}.tif')

    if not os.path.isfile(dem_fn) | os.path.isfile(project_parameters['dem']):
        print "USAGE: python start_from_dem.py projectName /path/to/src_dem.tif"
        sys.exit(1)

    ### CREATE DIRS, IF NECESSARY
    Safe_mkdir(project_parameters['itopo_dir'])
    Safe_mkdir(project_parameters['tmp_dir'])
    Safe_mkdir(project_parameters['dem_dir'])
    Safe_mkdir(project_parameters['hor_dir'])
    Safe_mkdir(project_parameters['out_dir'])

    project_parameters['init_cmds'] = []

    ### Load Spatial Parameters from DEM ###
    if not os.path.isfile(project_parameters['dem']):
        shutil.copyfile(dem_fn, project_parameters['dem'])

    project_parameters['dem_gt'] = Get_gt_dict(project_parameters['dem'])
    project_parameters['x_size'] = project_parameters['dem_gt']['x_size']
    project_parameters['y_size'] = project_parameters['dem_gt']['y_size']
    project_parameters['epsg'] = project_parameters['dem_gt']['epsg']
    project_parameters['dx'] = project_parameters['dem_gt']['dx']
    project_parameters['dy'] = project_parameters['dem_gt']['dy']
    project_parameters['xmin'] = project_parameters['dem_gt']['ulx']
    project_parameters['ymax'] = project_parameters['dem_gt']['uly']
    project_parameters['ymin'] = project_parameters['ymax'] + project_parameters['dy'] * \
        project_parameters['dem_gt']['y_size']
    project_parameters['xmax'] = project_parameters['xmin'] + project_parameters['dx'] * \
        project_parameters['dem_gt']['x_size']

    # Slope and Aspect
    project_parameters['asp'] = os.path.join(project_parameters['dem_dir'], project_name+'_asp.tif')
    project_parameters['slp'] = os.path.join(project_parameters['dem_dir'], project_name+'_slp.tif')
    slp_cmd = 'gdaldem slope {0} {1}'.format(project_parameters['dem'], project_parameters['slp'])
    asp_cmd = 'gdaldem aspect -zero_for_flat {0} {1}'.format(project_parameters['dem'], project_parameters['asp'])
    project_parameters['init_cmds'].append(slp_cmd)
    project_parameters['init_cmds'].append(asp_cmd)

    for cmd in project_parameters['init_cmds']:
        print cmd
        ret = os.system(cmd)

    slope = gdal_load(project_parameters['slp'])
    project_parameters['steepest_slope'] = int(math.ceil(np.nanmax(slope)))

    ###  DEFINE GEOTRANSFORMS
    if 'dem_gt' not in project_parameters.keys():
        project_parameters['dem_gt'] = Get_gt_dict(project_parameters['dem'])
        project_parameters['x_size'] = project_parameters['dem_gt']['x_size']
        project_parameters['y_size'] = project_parameters['dem_gt']['y_size']

    # Get srb bounding ulx and uly + x_size, y_size
    # This is only OK as long as corners from project_extents form geographic max + min values
    t = transform_epsg2epsg(int(project_parameters['epsg']), 4326)
    srb_xmin, srb_ymax = t_xy(t, project_parameters['xmin'], project_parameters['ymax'])
    srb_xmax, srb_ymin = t_xy(t, project_parameters['xmax'], project_parameters['ymin'])

    # expand to integers
    srb_ymin = math.floor(srb_ymin)
    srb_xmin = math.floor(srb_xmin)
    srb_ymax = math.ceil(srb_ymax)
    srb_xmax = math.ceil(srb_xmax)

    srb_ulx, srb_uly = srb_xmin, srb_ymax
    srb_x_size = int((srb_xmax - srb_xmin)) + 1  # / project_parameters['dx'])
    srb_y_size = int((srb_ymax - srb_ymin)) + 1  # / project_parameters['dy'])

    # If the input radiation is anything but srb, lots more will have to be recoded
    project_parameters['srb_gt'] = {'outfn': 'MEM',  # wgs84lo
                                    'from_dset': None,
                                    'epsg': 4326,
                                    'x_size': srb_x_size,
                                    'y_size': srb_y_size,
                                    'dx': 1,
                                    'dy': -1,
                                    'ulx': srb_ulx,
                                    'uly': srb_uly}

    # 1 degree = 3600 arc seconds.  1 arc second ~ 30m. 1000m output used 30 arc seconds = zoom 120
    project_parameters['srb_zoom_factor'] = int(108000 / project_parameters['dem_gt']['dx'])
    project_parameters['srb_hi_gt'] = project_parameters['srb_gt'].copy()

    project_parameters['srb_hi_gt']['dx'] = float(project_parameters['srb_gt']['dx']) / \
        project_parameters['srb_zoom_factor']  # 1 / 120 = 0.008333
    project_parameters['srb_hi_gt']['dy'] = float(project_parameters['srb_gt']['dy']) / \
        project_parameters['srb_zoom_factor']  # 1 / 120 = 0.008333
    project_parameters['srb_hi_gt']['x_size'] = project_parameters['srb_gt']['x_size'] * \
        project_parameters['srb_zoom_factor']
    project_parameters['srb_hi_gt']['y_size'] = project_parameters['srb_gt']['y_size'] * \
        project_parameters['srb_zoom_factor']

    # Make latlon
    project_parameters['lat'] = os.path.join(project_parameters['dem_dir'], project_name+'_lat.tif')
    project_parameters['lon'] = os.path.join(project_parameters['dem_dir'], project_name+'_lon.tif')
    latlon_args = {'ncols': project_parameters['x_size'], 'nrows': project_parameters['y_size'],
                   'ulx': project_parameters['xmin'], 'uly': project_parameters['ymax'],
                   'epsg': project_parameters['epsg'], 'cellsize': project_parameters['dx'],
                   'out_ds': os.path.join(project_parameters['dem_dir'], project_name)}
    print 'Making Lat and Lon grids in output projection'
    mk_latlon_grids(**latlon_args)

    # Prepare itopo_1month.sh parallel arguments
    months_cmds_fn = os.path.join(project_parameters['itopo_dir'], 'itopo_months_{0}.cmds'.format(project_name))
    months_to_run = []
    for mm in project_parameters['months'].split():
        for year in range(project_parameters['first_year'], project_parameters['last_year']+1):
            months_to_run.append('./itopo_1month.sh {0} {1} {2} && rm -rf {0}/tmp/{1}{2} \n'.format(project_name,
                                                                                                    year,
                                                                                                    str(mm).zfill(2)))

    with open(months_cmds_fn, 'w') as f:
        f.writelines(months_to_run)

    with file(project_param_fn, 'w') as f:
        yaml.safe_dump(project_parameters, f)

    print ' '
    print 'Choose the number of processors to dedicate (numThreads) then execute:'
    print './run_project.sh {0} numThreads'.format(project_name)
