#!/usr/bin/python

__author__ = 'wiley'

from topocorr_funcs import gdal_load, gdal_save_grid, Cast_shade
import numpy as np
import sys
import osr
import gdal
import yaml

# from yaml: dem_dir,utm33n_md

project_name = sys.argv[1]

project_param_fn = '{0}/{0}_parameters.yaml'.format(project_name)
with file(project_param_fn) as f:
    project_parameters = yaml.safe_load(f)


if __name__ == "__main__":
    utc_hour = int(sys.argv[2])

    sunview_fmt = project_parameters['sunview']

    ydays = np.arange(1,367)

    # load lat, lon
    lat = gdal_load(project_parameters['lat'])
    lon = gdal_load(project_parameters['lon'])

    sunview_args = {'x_size': project_parameters['x_size'], 'y_size': project_parameters['y_size'],
                'ulx': project_parameters['xmin'], 'uly': project_parameters['ymax'], 'epsg': project_parameters['epsg'],
                'dx': project_parameters['dx'], 'dy': project_parameters['dy']}

    for yday in ydays:

        sunview_args['outfn'] = sunview_fmt.format(yday, utc_hour)
        sunview_args['dtype'] = gdal.GDT_Byte
        sunview_args['from_dset'] = Cast_shade(project_parameters, lat, lon, yday, utc_hour).astype('int')
        gdal_save_grid(**sunview_args)
