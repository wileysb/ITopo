#!/usr/bin/python
"""
Give the altitude of the last obstruction in the given azimuth direction for every cell
"""

import time
import sys
import gdal
import numpy as np
from rpy2.robjects.packages import importr

# R imports
r_raster = importr('raster')
r_insol = importr("insol")

# Timer function
def inner_shader(altitude, azi, rdem):
    sol_vector = r_insol.normalvector(90-altitude, azi)
    shd = r_insol.doshade(rdem, sol_vector)
    rpy2shd = np.array(r_raster.as_matrix(shd))

# demfn from commandline
demfn = sys.argv[1]

#
rdem = r_raster.raster(demfn)
dem_ds = gdal.Open(demfn, gdal.GA_ReadOnly)
dem_b = dem_ds.GetRasterBand(1)

iter_times = []
for az in range(360):
    for alt in [1, 10, 20, 30]:
        start = time.time()
        inner_shader(alt, az, rdem)
        iter_times.append(time.time() - start)

print 'mean time:', np.mean(iter_times)
print '20th %ile:', np.percentile(iter_times, 20)
print '80th %ile:', np.percentile(iter_times, 80)


