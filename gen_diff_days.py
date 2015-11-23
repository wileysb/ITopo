#!/usr/bin/python

__author__ = 'wiley'

import sys
import yaml
import numpy as np

from topocorr_funcs import unpack_srb_variables, srb_to_prjEpsg

prj_name = sys.argv[1] # $ python accumulate_skyview.py prjName

prj_param_fn = '{}_parameters.yaml'.format(prj_name)
with file(prj_param_fn) as f:
    prj = yaml.safe_load(f)


if __name__ == "__main__":
    yyyy=str(sys.argv[1])
    mm  = str(sys.argv[2]).zfill(2)

    out_dir = prj['tmp_dir']

    srb_3hr_fn = prj['srb'].format(yyyy,mm)

    srb_3hr_vars = unpack_srb_variables(srb_3hr_fn)

    srb_to_prjEpsg(srb_3hr_vars, out_dir)
