#!/usr/bin/python

__author__ = 'wiley'

import sys
import os
import yaml

from topocorr_funcs import unpack_srb_variables, Cast_shade, Apply_topo_corr,gdal_load, gdal_save_grid

project_name = sys.argv[1] # $ python accumulate_skyview.py projectName

project_param_fn = '{}_parameters.yaml'.format(project_name)
with file(project_param_fn) as f:
    project_parameters = yaml.safe_load(f)

if __name__ == "__main__":
    yyyy=str(sys.argv[2])
    mm  = str(sys.argv[3]).zfill(2)

    out_dir = os.path.join(project_parameters['tmp_dir'],yyyy+mm)
    file_fmt = os.path.join(out_dir, '{0}_{1}_{2}_{3}.tif')
    shade_fmt = project_parameters['sunview']

    srb_3hr_fn = project_parameters['srb'].format(yyyy,mm)

    srb_3hr_vars = unpack_srb_variables(srb_3hr_fn)

    ydays     = srb_3hr_vars['ydays']
    utc_hours = srb_3hr_vars['utc_hours']

    # load slope, aspect, lat, lon, skyview
    slope_fn = project_parameters['slp']  #  os.path.join(project_parameters['dem_dir'], 'gtopo30slp_utm33n.asc')
    aspect_fn = project_parameters['asp']  #  os.path.join(project_parameters['dem_dir'], 'gtopo30asp_utm33n.asc')
    lat_fn = project_parameters['lat']  #  os.path.join(project_parameters['dem_dir'], 'gtopo30lat_utm33n.asc')
    lon_fn = project_parameters['lon']  #  os.path.join(project_parameters['dem_dir'], 'gtopo30lon_utm33n.asc')
    sky_fn = os.path.join(project_parameters['dem_dir'],project_name + '_skyview.asc')  #  os.path.join(project_parameters['dem_dir'], 'gtopo30sky_utm33n.asc')

    slope  = gdal_load(slope_fn)  # np.loadtxt(slope_fn,skiprows=6,delimiter=' ')
    aspect = gdal_load(aspect_fn) # np.loadtxt(aspect_fn,skiprows=6,delimiter=' ')
    lat    = gdal_load(lat_fn)    # np.loadtxt(lat_fn,skiprows=6,delimiter=' ')
    lon    = gdal_load(lon_fn)    # np.loadtxt(lon_fn,skiprows=6,delimiter=' ')
    skyview= gdal_load(sky_fn)    #np.loadtxt(sky_fn,skiprows=6,delimiter=' ')

    for i in range(len(ydays)):
        yday = ydays[i]
        utc_hour = utc_hours[i]

        if os.path.isfile(shade_fmt.format(yday,utc_hour)):
            shade = gdal_load(shade_fmt.format(yday,utc_hour)) # turbo!
        else:
            shade = Cast_shade(lat, lon, yday, utc_hour)    # not-turbo :-(

        #  load_params = {'dset':'sw_sfc_dn','yyyy':yyyy,'yday':yday,'utc_hour':utc_hour, 'tif_dir':out_dir}
        sw_sfc_dn_utm = gdal_load(file_fmt.format('sw_sfc_dn',yyyy,yday,utc_hour))  #ld_srb_utm33(**load_params)
        #load_params['dset'] = 'diffuse'
        diffuse_utm = gdal_load(file_fmt.format('diffuse',yyyy,yday,utc_hour))  # ld_srb_utm33(**load_params)
        diffuse_utm[diffuse_utm>1]=1


        project_parameters['dem_gt']['outfn'] = file_fmt.format('itopo',yyyy, yday, utc_hour)
        project_parameters['dem_gt']['nanhandle'] = -999
        topo_params = {'sw_sfc_dn':sw_sfc_dn_utm, 'p_diffuse':diffuse_utm, 'shade':shade,
                       'slope':slope, 'aspect':aspect, 'skyview':skyview, 'lat':lat, 'lon':lon,
                       'yday':yday, 'utc_hr':utc_hour}
        project_parameters['dem_gt']['from_dset'] = Apply_topo_corr(**topo_params)
        subzero = project_parameters['dem_gt']['from_dset'] < 0
        project_parameters['dem_gt']['from_dset'][subzero] = 0
        gdal_save_grid(**project_parameters['dem_gt'])