Topographic correction of irradiance.

Trying to make the code more streamlined and flexible, rather than locked 
in extent and resolution.

TODO:
* Init Paths
 - path to DEM
 - DEM derivatives directory (default DEM directory)
 - SRB directory
 - temporary timestep and processing output directory

* Init spatial parameters 
 - define geotransforms for:
  + [ ] Source (srb, WGS84 lo res)
  + [ ] Intermediate, WGS84 high res
  + [ ] Output, from DEM (must be rectilinear/slope- & aspect-able

* Init temporal parameters?
 - months and years to cover?  default 1984-2007, all 12 months? Possibly useful to run 'just April and June' or something?  

* Prepare DEM derivatives:
 - Slope and Aspect (gdaldem slope, gdaldem aspect -zero_when_flat)
  + [ ] gdaldem slope, gdaldem aspect -zero_for_flat
 - Lat and Lon
  + [ ]  Should be a simple function based on geotransforms 
 - Sunview and Skyview 
  + [ ]  Binary Obstruction Grids (no obstruction when solar zenith is higher than max slope angle)
  + [ ]  accumulate Skyview
  + [ ]  pre-stack a year's worth of 3hr Sunview grids, or just save them if they aren't saved yet?



