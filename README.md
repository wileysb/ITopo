Topographic correction of irradiance.

Trying to make the code more streamlined and flexible, rather than locked 
in extent and resolution.

Directions for use:
1. Clone ITopo project to a disk with plenty of space
2. Create 'prjName' directory within ITopo/
3. Define 'prjName' parameters in parameters.yaml
4. run $ python start_project.py prjName

TODO:
* Check extents of DEM derivatives:
 - [ ] Slope and Aspect 
 - [ ] Lat and Lon
* Produce Sunview and Skyview 
  + [ ]  Binary Obstruction Grids (no obstruction when solar zenith is higher than max slope angle)
  + [ ]  accumulate Skyview
  + [ ]  pre-stack a year's worth of 3hr Sunview grids, or just save them if they aren't saved yet?