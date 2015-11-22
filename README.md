# Topographic correction of irradiance.

Trying to make the code more streamlined and flexible, rather than locked 
in extent and resolution.

### Directions for use:
1. Clone ITopo project to a disk with plenty of space
2. Create 'prjName' directory within ITopo/
3. Define 'prjName' parameters in parameters.yaml
 * For best quality control, preload prjName_dem.tif in dem_derivs/. 
 * Then you can also leave blank the project's spatial parameters.
4. run $ python start_project.py prjName
5. run $ parallel -j numThreads -- < prjDir/BOG.cmds
6. run $ python accumulate_skyview.py prjName


#### Timing
| Cols x Rows | Step 5   | Step 6 |
| ----------- | -------- | ------ |
|  504 x 504  |    ?     | ~20min |
| 5041 x 5041 |Too large!|   ...  |

Porsgrunn10m (5041x5041) BOGs is already 30G at azimuth 7/359.  Each BOG.asc is 49M.  This is unmanageably large.
Maybe more manageable as compressed tifs? for comparison, Norway at 1km (1210x1532) was ~3.6M.

Possible solution: pick a subregion within Porsgrunn at half cols, half rows? or 33% cols and rows...



### TODO:
* Move cmd lists into the project folders
  + Maybe params can stay in the functions folder though?
* Produce Sunview and Skyview 
  + [x]  Binary Obstruction Grids (no obstruction when solar zenith is higher than max slope angle)
  + [ ]  accumulate Skyview
  + [ ]  pre-stack a year's worth of 3hr Sunview grids, or just save them if they aren't saved yet?