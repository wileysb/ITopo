#!/usr/bin/python
"""
* Call after horizon and skyview functions
* Call once for all timesteps within specified month
* Script called with 3 arguments: projectName, year, month
* For each timestep within the specified month, produces diffuse proportion and surface irradiance grids at the DEM projection, extent, and resolution:
    * Derives diffuse proportion from 1-degree lat-lon irradiance at surface and TOA, loaded from SRB files
    * Resamples in memory diffuse proportion and surface irradiance to finer resolution lat-lon grid over project area
    * Resamples diffuse proportion and surface irradiance to DEM spatial parameters (saved to disk)

About 6 minutes per month, at 100m
"""

import os
import sys
import yaml

from topocorr_funcs import unpack_srb_variables, srb_to_projectEpsg, Safe_mkdir

__author__ = 'wiley'

project_name = sys.argv[1]

project_param_fn = '{0}/{0}_parameters.yaml'.format(project_name)
with file(project_param_fn) as f:
    project_parameters = yaml.safe_load(f)


if __name__ == "__main__":
    yyyy = str(sys.argv[2])
    mm = str(sys.argv[3]).zfill(2)

    out_dir = os.path.join(project_parameters['tmp_dir'], yyyy+mm)
    Safe_mkdir(out_dir)

    srb_3hr_fn = project_parameters['srb'].format(yyyy, mm)

    srb_3hr_vars = unpack_srb_variables(srb_3hr_fn, project_parameters)
    srb_3hr_vars['year'] = yyyy

    srb_to_projectEpsg(srb_3hr_vars, out_dir, project_parameters)
