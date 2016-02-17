#!/usr/bin/python
"""
Give the altitude of the last obstruction in the given azimuth direction for every cell
"""

import gdal
import numpy as np
from topocorr_funcs import gdal_save_grid, Get_gt_dict


def get_slope_area(slope_fn, area_ratio=False):
    """
    slope_area_grid = get_slope_area(slope_fn)
    slope_ratio_grid = get_slope_area(slope_fn, area_ratio=True)

    :param slope_fn: string, path to slope grid
    :param area_ratio: bool (default False),
                T      |       F
    :return: area_grid, or area_ratio_grid
    """

    # Load slope array and geotransform
    slope_ds = gdal.Open(slope_fn, gdal.GA_ReadOnly)
    slope_gt = slope_ds.GetGeoTransform()
    slope_bn = slope_ds.GetRasterBand(1)
    slope_arr = slope_bn.ReadAsArray()
    slope_nodata = slope_bn.GetNoDataValue()

    # Calculate footprint area of each cell
    dx, dy = np.abs(slope_gt[1]), np.abs(slope_gt[5])
    footprint = dx*dy

    # Calculate surface area
    if area_ratio:
        # return area_ratio
        return np.where(slope_arr!=slope_nodata,
                        1 / np.cos(np.deg2rad(slope_arr)), 1)
    else:
        # return sfc_area
        return np.where(slope_arr!=slope_nodata,
                        footprint / np.cos(np.deg2rad(slope_arr)), footprint)


def save_grid_like(gt_fn, out_fn, out_arr):
    """wrapper for gdal_save_grid :-P"""

    grid_dict = Get_gt_dict(gt_fn)
    grid_dict['from_dset'] = out_arr
    grid_dict['outfn'] = out_fn

    gdal_save_grid(**grid_dict)


if __name__ == '__main__':
    import sys

    return_area_ratio = len(sys.argv) > 3

    slope_fn = sys.argv[1]
    out_fn = sys.argv[2]

    if return_area_ratio:
        out_arr = get_slope_area(slope_fn, area_ratio=True)
    else:
        out_arr = get_slope_area(slope_fn)

    save_grid_like(slope_fn, out_fn, out_arr)