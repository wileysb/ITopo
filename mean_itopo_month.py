#!/usr/bin/python

import os
import sys
import glob
import yaml
import numpy as np

from topocorr_funcs import gdal_load, unpack_srb_variables, gdal_save_grid

__author__ = 'wiley'


project_name = sys.argv[1]  # $ python accumulate_skyview.py projectName

project_param_fn = '{}_parameters.yaml'.format(project_name)
with file(project_param_fn) as f:
    project_parameters = yaml.safe_load(f)


if __name__ == "__main__":
    yyyy = str(sys.argv[2])
    mm = str(sys.argv[3]).zfill(2)

    tmp_dir = os.path.join(project_parameters['tmp_dir'], yyyy+mm)
    in_fmt = os.path.join(tmp_dir, '{0}_{1}_{2}_{3}.tif')  # .format(dset,yyyy,mm,utc_hour)

    out_fmt = os.path.join(project_parameters['out_dir'], '{0}_{1}_{2}.tif')  # .format(dset,yyyy,mm)

    srb_3hr_fn = project_parameters['srb'].format(yyyy, mm)

    srb_3hr_vars = unpack_srb_variables(srb_3hr_fn, project_parameters)

    ydays = srb_3hr_vars['ydays']
    utc_hours = srb_3hr_vars['utc_hours']

    dsets = {}

    for dset in ['itopo', 'diffuse', 'sw_sfc_dn']:
        fn_pattern = os.path.join(tmp_dir, dset+'_*.tif')
        dset_files = glob.glob(fn_pattern)
        arr = np.ones((len(dset_files), project_parameters['dem_gt']['y_size'],
                       project_parameters['dem_gt']['x_size']), dtype='float')
        for i in range(len(dset_files)):
            arr[i] = gdal_load(dset_files[i])
            arr[i][arr[i] < 0] = 0

        dsets[dset] = np.nanmean(arr, axis=0)
        project_parameters['dem_gt']['from_dset'] = dsets[dset]
        project_parameters['dem_gt']['nanhandle'] = -999
        project_parameters['dem_gt']['outfn'] = out_fmt.format(dset, yyyy, mm)
        gdal_save_grid(**project_parameters['dem_gt'])

    diff = (dsets['itopo'] - dsets['sw_sfc_dn']) / dsets['sw_sfc_dn']
    diff[np.isinf(diff)] = 0
    project_parameters['dem_gt']['from_dset'] = diff
    project_parameters['dem_gt']['nanhandle'] = -999
    project_parameters['dem_gt']['outfn'] = out_fmt.format('irrad_rel_diff', yyyy, mm)
    gdal_save_grid(**project_parameters['dem_gt'])
