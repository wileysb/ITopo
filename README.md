# Topographic correction of irradiance.


### Directions for use:
1. Prepare a dem in the desired output projection and alignment
2. Check the 5 lines of generic parameters in parameters.yaml
3. Run $ python start_from_dem.py projectName /path/to/dem.tif
4. Run $ chmod 744 itopo_1month.sh
5. Choose a number of processors to tie up, and run the commands prompted at the completion of [4].


### TODO:
* Check grid products for alignment!
 - [x] DEM, slope, aspect
 - [ ] BOG, skyview, sunview(t)
 - [ ] srb (lo,hi,project)
* Consider best srb resampling method (hi to project)
* polish project workflow, topographic correction from DEM to monthly climatology
* replace R shade with PyForShade
* [x] Implement quicker, more compact shade
 - wean itopo calculations from BOGs --> horizon
* Can I run shade to measure 'obstruction at 2m' for each cell, with all other cells from topo elevation?
 - Simulating irrad that would be measured by a sensor at 2m at that location . . .
* [ ] script to test r2py doshade timing 


### WORKFLOW SUMMARY
* Derive time-independent grids (DEM derivatives)
 - Slope, aspect, latitude, and longitude
 - Binary Obstruction Grids
 - sum over hemisphere, weighted by cos zenith, to derive skyview grid
* for each 3 hour UTC timestep in 1 arbitrary year:
 - compute solar position grids (azimuth and zenith) in local time
 - For each pixel, load binary 'obstruction' value from topographic obstruction grid matching local solar position and save grid as sunview(t)
* for every month from January 1984 - December 2007:
 - 3 hour UTC timestep within the given month:
  1. compute solar position grids (azimuth and zenith) in local time
  2. derive diffuse fraction of global radiation, after Skartveit et al \cite{skartveit98}
  3. resample 'diffuse fraction' and 'downwelling shortwave at the surface' grids from 1x1\degree to 1x1km UTM-33N
  4. topographic correction factor, direct irradiance
  5. diffuse correction factor: diffuse fraction*skyview
  6. topographic correction = diffuse correction + direct correction
 - average all timesteps for monthly averages
* average all 23 years of each month for 'climatological monthly means'
