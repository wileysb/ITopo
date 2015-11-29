#!/usr/bin/python
"""
About 6 minutes per month, at 100m
"""
__author__ = 'wiley'

import os
import sys
import yaml

from topocorr_funcs import unpack_srb_variables, srb_to_projectEpsg, Safe_mkdir

project_name = sys.argv[1] # $ python accumulate_skyview.py projectName

project_param_fn = '{}_parameters.yaml'.format(project_name)
with file(project_param_fn) as f:
    project_parameters = yaml.safe_load(f)


if __name__ == "__main__":
    yyyy=str(sys.argv[2])
    mm  = str(sys.argv[3]).zfill(2)

    out_dir = os.path.join(project_parameters['tmp_dir'],yyyy+mm)
    Safe_mkdir(out_dir)

    srb_3hr_fn = project_parameters['srb'].format(yyyy,mm)

    srb_3hr_vars = unpack_srb_variables(srb_3hr_fn, project_parameters)

    srb_to_projectEpsg(srb_3hr_vars, out_dir, project_parameters)
