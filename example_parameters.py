#!/usr/bin/python

import os

parameters = {

### PROJECT NAME ###
'prj_name': os.path.basename(__file__).split['_'][0],  # for 'example_parameters.py', prj_name is 'example'

### DEFINE PATHS TO IMPORTANT DIRECTORIES ###
'src_dem': '/path/to/dem.tif',
'srb_dir': '/path/to/srb/',
'itopo_dir': '/path/to/itopo_output/',


### DEFINE SPATIAL EXTENTS ###
'epsg': 32633,
'xmin': 0, 'ymin': 0, 'xmax': 4, ' ymax': 4
}