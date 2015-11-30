# Topographic correction of irradiance.

Trying to make the code more streamlined and flexible, rather than locked 
in extent and resolution.

### Directions for use:
1'a. Create 'projectName' directory within ITopo/
1'b. Define 'projectName' parameters in parameters.yaml
 * For best quality control, preload projectName_dem.tif in dem_derivs/. 
 * Then you can also leave blank the project's spatial parameters.
1'c. run $ python start_project.py projectName

OR

1. run $ python start_from_dem.py projectName /path/to/src_dem.tif

2. run $ parallel -j numThreads -- < projectDir/BOG.cmds
3. run $ python accumulate_skyview.py projectName
4. run $ parallel -j numThreads ./gen_sunview.py projectName {1}::: 0 3 6 9 12 15 18 21

Implement workflow: 
5.  python gen_diff_days.py projectName yyyy mm
6.  python gen_itopo_days.py projectName yyyy mm
7. python mean_itopo_month.py projectName yyyy mm
8. Free space with rm -rf projectName/tmp/yyyymm

#### Timing
Cols x Rows | Step 5^  | Step 6 | Step 7^
----------- | -------- | ------ | -----
 504 x 504  | 1557min  | ~30min | ~5min
~~5041 x 5041~~ |Too large!|   ...  | 

^ time can be divided by numThreads

Porsgrunn10m (5041x5041) BOGs is already 30G at azimuth 7/359.  Each BOG.asc is 49M.  This is unmanageably large.
Maybe more manageable as compressed tifs? for comparison, Norway at 1km (1210x1532) was ~3.6M.

Possible solution: pick a subregion within Porsgrunn at half cols, half rows? or 33% cols and rows...



### TODO:
* Check grid products for alignment!
 - [x] DEM, slope, aspect
 - [ ] BOG, skyview, sunview(t)
 - [ ] srb (lo,hi,project)
* Consider best srb resampling method (hi to project)
* polish project workflow, topographic correction from DEM to monthly climatology