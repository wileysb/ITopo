#!/usr/bin/python
"""
* Init Paths
 - path to DEM
 - DEM derivatives directory
 - SRB directory
 - Binary Obstruction Grid (BOG) directory
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
 -- Binary Obstruction Grids
 -- accumulate skyview
 -- pre-stack a year's worth of 3hr sunview grids, or just save them if they aren't saved yet?
"""

import os
import sys
import yaml
import math
import shutil
import numpy as np
from topocorr_funcs import Safe_mkdir, Get_gt_dict, transform_epsg2epsg, t_xy, mk_latlon_grids, gdal_load

__author__ = 'Wiley Bogren'

project_name = sys.argv[1]  # $ python start_project.py projectName
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
    project_parameters['BOG_dir'] = os.path.join(project_parameters['dem_dir'], 'BOGs')
    project_parameters['out_dir'] = os.path.join(project_parameters['itopo_dir'], 'monthly_means')
    project_parameters['BOG'] = os.path.join(project_parameters['BOG_dir'], 'az{0}zen{1}.asc')#.format(solar_az,solar_zen)

    sunview_dir = os.path.join(project_parameters['dem_dir'], 'sunview')
    project_parameters['sunview'] = os.path.join(sunview_dir, 'sunview_{0}_{1}.tif')

    project_parameters['srb'] = os.path.join(project_parameters['srb_dir'],'srb_rel3.0_shortwave_3hrly_{0}{1}.nc')#.format(yyyy,mm)

    if not os.path.isfile(dem_fn) | os.path.isfile(project_parameters['dem']):
        print "USAGE: python start_from_dem.py projectName /path/to/src_dem.tif"
        sys.exit(1)

    ### CREATE DIRS, IF NECESSARY
    Safe_mkdir(project_parameters['itopo_dir'])
    Safe_mkdir(project_parameters['tmp_dir'])
    Safe_mkdir(project_parameters['dem_dir'])
    Safe_mkdir(project_parameters['BOG_dir'])
    Safe_mkdir(project_parameters['out_dir'])
    Safe_mkdir(sunview_dir)

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
    bog_cmd_list = []
    for az in range(0, 360):
        for zen in range(90-project_parameters['steepest_slope'], 90):
            outfn = project_parameters['BOG'].format(az, zen)
            bog_cmd_list.append("Rscript mk_shade_grid.R {0} {1} {2} {3}\n".format(
                project_parameters['dem'], zen, az, outfn))
    with open(os.path.join(project_parameters['itopo_dir'], 'BOG.cmds'), 'w') as bog_cmds:
        bog_cmds.writelines(bog_cmd_list)

    ###  DEFINE GEOTRANSFORMS
    if 'dem_gt' not in project_parameters.keys():
        project_parameters['dem_gt'] = Get_gt_dict(project_parameters['dem'])
        project_parameters['x_size'] = project_parameters['dem_gt']['x_size']
        project_parameters['y_size'] = project_parameters['dem_gt']['y_size']
    # todo NOTE:
    #  in the hardcoded utm33n_md, 'ulx' and 'uly' off by 500m (pixel corners vs pixel center)
    #   'ulx':-84500.00 hardcoded, vs  -85000.0  from function
    #   'uly':7961500.0 hardcoded, vs  7962000.0 from function

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

    # Prepare gen_sunview.py parallel arguments
    sunview_cmds_fn = os.path.join(project_parameters['itopo_dir'], 'sunview_{0}.cmds'.format(project_name))
    with open(sunview_cmds_fn, 'w') as f:
        for utc_hr in range(0, 22, 3):
            f.write('python gen_sunview.py {0} {1}\n'.format(project_name, utc_hr))

    # Prepare itopo_1month.sh parallel arguments
    months_cmds_fn = os.path.join(project_parameters['itopo_dir'], 'itopo_months_{0}.cmds'.format(project_name))
    months_to_run = []
    for mm in project_parameters['months']:
        for year in range(project_parameters['first_year'], project_parameters['last_year']+1):
            months_to_run.append('./itopo_1month.sh {0} {1} {2} $$ rm -rf {0}/tmp/{1}{2} \n'.format(project_name, year, str(mm).zfill(2)))

    with open(months_cmds_fn, 'w') as f:
        f.writelines(months_to_run)

    # Prepare Stage 2 commands
    bash_cmds = ['#!/usr/bin/env bash\n', '\n']

    bash_cmds.append('parallel -j $1 -- < {0}/BOG.cmds\n'.format(project_name))
    bash_cmds.append('python accumulate_skyview.py '+project_name+'\n')
    bash_cmds.append('parallel -j $1 -- < {0}\n'.format(sunview_cmds_fn))
    bash_cmds.append('parallel -j $1 -- < {0}\n'.format(months_cmds_fn))

    with open(project_name+'.sh', 'w') as f:
        f.writelines(bash_cmds)

    with file(project_param_fn, 'w') as f:
        yaml.safe_dump(project_parameters, f)

    print 'Choose the number of processors to dedicate (numThreads) then execute these two commands to continue:'
    print 'chmod 744 ./'+project_name+'.sh'
    print './'+project_name+'.sh', 'numThreads'