# Topographic correction of irradiance.

Trying to make the code more streamlined and flexible, rather than locked 
in extent and resolution.


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
