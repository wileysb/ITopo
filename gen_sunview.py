#!/usr/bin/python

__author__ = 'wiley'

from topocorr_funcs import Prj_mkdir, gdal_load, Cast_shade
import numpy as np
import os
import sys
import osr
import gdal
import yaml

# from yaml: dem_dir,utm33n_md

project_name = sys.argv[1] # $ python accumulate_skyview.py prjName

prj_param_fn = '{}_parameters.yaml'.format(project_name)
with file(prj_param_fn) as f:
    project_parameters = yaml.safe_load(f)




def gdal_save_binary_grid(from_dset, outfn, epsg, x_size, y_size, ulx, uly, dx, dy,
                          nanhandle=False, rot0=0, rot1=0):

    # Bring projection info from EPSG to WKT string
    prj = osr.SpatialReference()
    prj.ImportFromEPSG(epsg)
    prj_string = prj.ExportToWkt()

     # Create memory dataset
    if outfn == 'MEM':
        mem_drv = gdal.GetDriverByName('MEM')
        dst = mem_drv.Create('', x_size, y_size, 1, gdal.GDT_Byte)

    else:
        gtiff_drv = gdal.GetDriverByName('GTiff')
        dst = gtiff_drv.Create(outfn,x_size, y_size, 1, gdal.GDT_Byte)

    # Assemble geotransform
    geo_t = (ulx, dx, rot0, uly, rot1, dy)

    # Write geotransform, projection, and array to memory dataset
    dst.SetGeoTransform(geo_t)
    dst.SetProjection(prj_string)
    bnd = dst.GetRasterBand(1)
    if nanhandle is not False:
        from_dset[np.isinf(from_dset)] = nanhandle
        from_dset[np.isnan(from_dset)] = nanhandle
        bnd.SetNoDataValue(nanhandle)
    bnd.WriteArray(from_dset)

    del dst # return dst


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
        sunview_args['from_dset'] = Cast_shade(project_parameters, lat, lon, yday, utc_hour).astype('int')
        gdal_save_binary_grid(**sunview_args)
