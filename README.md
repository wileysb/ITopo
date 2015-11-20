Topographic correction of irradiance.

Trying to make the code more streamlined and flexible, rather than locked 
in extent and resolution.

Directions for use:
1. Clone ITopo project to a disk with plenty of space
2. Create 'prjName' directory within ITopo/
3. Define 'prjName' parameters in parameters.yaml
 * For best quality control, preload prjName_dem.tif in dem_derivs/. 
 * Then you can also leave blank the project's spatial parameters.
4. run $ python start_project.py prjName
5. run $ parallel -j numThreads -- < prjDir/BOG.cmds

TODO:
* Move cmd lists into the project folders
  + Maybe params can stay in the functions folder though?
* Produce Sunview and Skyview 
  + [x]  Binary Obstruction Grids (no obstruction when solar zenith is higher than max slope angle)
  + [ ]  accumulate Skyview
  + [ ]  pre-stack a year's worth of 3hr Sunview grids, or just save them if they aren't saved yet?