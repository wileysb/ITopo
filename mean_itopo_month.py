#!/usr/bin/python

__author__ = 'wiley'

import sys, os, glob
import numpy as np

from topocorr_funcs import srb_utm33_dir, srb_3hr_fmt, gdal_load, utm33n_md, \
    gdal_save_grid, unpack_srb_variables

if __name__ == "__main__":
    yyyy=str(sys.argv[1])
    mm  = str(sys.argv[2]).zfill(2)

    out_dir = 'srb_utm_daily/'+yyyy+mm

    srb_3hr_fn = srb_3hr_fmt.format(yyyy,mm)

    srb_3hr_vars = unpack_srb_variables(srb_3hr_fn)

    ydays     = srb_3hr_vars['ydays']
    utc_hours = srb_3hr_vars['utc_hours']

    dsets = {}

    for dset in ['itopo','diffuse','sw_sfc_dn']:
        fn_pattern = os.path.join(out_dir, dset+'_*.tif')
        dset_files = glob.glob(fn_pattern)
        arr = np.ones((len(dset_files), utm33n_md['y_size'], utm33n_md['x_size']),dtype='float')
        for i in range(len(dset_files)):
            arr[i] = gdal_load(dset_files[i])
            arr[i][arr[i]<0] = 0

        dsets[dset] = np.nanmean(arr,axis=0)
        utm33n_md['from_dset'] = dsets[dset]
        utm33n_md['nanhandle'] = -999
        utm33n_md['outfn'] = os.path.join(srb_utm33_dir,dset+'_'+yyyy+mm+'.tif')
        gdal_save_grid(**utm33n_md)


    diff = (dsets['itopo'] - dsets['sw_sfc_dn']) / dsets['sw_sfc_dn']
    diff[np.isinf(diff)] = 0
    utm33n_md['from_dset'] = diff
    utm33n_md['nanhandle'] = -999
    utm33n_md['outfn'] = os.path.join(srb_utm33_dir,'irrad_rel_diff_'+yyyy+mm+'.tif')
    gdal_save_grid(**utm33n_md)