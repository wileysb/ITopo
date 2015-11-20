#!/usr/bin/python

'''Define skyview using the shade grids for az=0:369 and zen=0:89

Not all angles are reasonable orientations for a sun, especially in Norway,
but when regarded as 'viewangles' they suddenly become very meaningful.
Each grid is then a 'view' test of whether that sector of the sky is
visible from each point in the grid.  The sum of visible angles is weighted
by cosine(90-zenith) to account for smaller sectors per degree
as zenith angle approaches noon. This weighted sum, divided by 
the maximum possible sum, gives the % sky viewable from each point in the grid'''

import numpy as np
import os
import sys

if len(sys.argv)==1:
    sz_start=0
    sz_end=90
    out_fn='/home/sl_wib/skyview.asc' # todo has to go into dem_derivs folder
elif len(sys.argv)==2:
    sz_start=int(sys.argv[1])
    sz_end=sz_start+1
    out_fmt = '/home/sl_wib/skyview_{}zen_{}max.asc' #.format(zen,max) todo has to go into tmp folder

# define locations
#out_fn = '/home/sl_wib/skyview_{}.asc' # We can manually add the header later to make it a proper ascii grid
shade_dir = '/home/sl_wib/shade/' # todo load path from yaml
shade_fmt = os.path.join(shade_dir,'solaraz{}solarzen{}.asc')#.format(sol_az,sol_zen) todo load path from yaml

# create max_sum counter
max_sum = 0

# create sum container
shade1 = np.loadtxt(shade_fmt.format(1,sz_start),skiprows=6,delimiter=' ')
skyview = np.zeros_like(shade1)
del shade1

# Loop through the grids in 'shade/' and accumulate the sums to skyview
for sol_zen in range(sz_start,sz_end):
    zen_weight = np.cos(np.deg2rad(90-sol_zen))
    print 'Zenith:',sol_zen,'Weight:',zen_weight
    for sol_az in range(360):
        shade = np.loadtxt(shade_fmt.format(sol_az,sol_zen),skiprows=6,delimiter=' ')
        skyview+=shade*zen_weight
        max_sum+=zen_weight
        
if len(sys.argv)==1:
    skyview/=max_sum
    np.savetxt(out_fn,skyview,delimiter=' ')
else:
    np.savetxt(out_fmt.format(sol_zen,max_sum), skyview ,delimiter=' ')
