# Topographic correction of irradiance.

### Summary of github contents:

####[parameters.yaml:](parameters.yaml)
* Adjust project parameters including directory paths and timeframe

####[start_from_dem.py:](start_from_dem.py)
* Script called with project name and path to DEM
* Loads project parameters
* Creates output directories for both final and temporary products
* Loads spatial parameters from DEM: projection, extent, and resolution
* Creates Slope, Aspect, Lat, and Lon raster derivatives from the elevation model
* Generates syntax for further processing commands based on project parameters, DEM spatial parameters, and directory structure
* Prints initial commands to standard out, and writes parallel-processing and sequential commands to file

####[find_horizon.py:](find_horizon.py)
* Produces a set of 360 raster grids, each giving the altitude of the last obstruction in the given azimuth direction for every cell
  (ie, the degrees above horizontal where the sky or sun is first visible instead of topography)

####[skyview_from_horizon.py:](skyview_from_horizon.py)
* Produces a single raster grid giving the proportion of unobstructed sky viewable from each point in the grid, calculated from the 360 horizon grids

####[gen_diff_days.py:](gen_diff_days.py)
* Script called with 3 arguments: projectName, year, month
* For each timestep within the specified month, produces diffuse proportion and surface irradiance grids at the DEM projection, extent, and resolution:
    * Derives diffuse proportion from 1-degree lat-lon irradiance at surface and TOA, loaded from SRB files
    * Resamples in memory diffuse proportion and surface irradiance to finer resolution lat-lon grid over project area
    * Resamples diffuse proportion and surface irradiance to DEM spatial parameters (saved to disk)
    
####[gen_itopo_days.py:](gen_itopo_days.py)
* Script called with 3 arguments: projectName, year, month
* For each timestep within the specified month, performs and saves topographic correction of surface irradiance:
    * loads surface irradiance, diffuse proportion, skyview, and shade (sunview)
    * calculates topographic correction for diffuse and direct irradiance
    * combines diffuse and direct correction based on diffuse proportion
    * writes out surface irradiance for that timestep adjusted by topography

####[mean_itopo_month.py:](mean_itopo_month.py)
* Script called with 3 arguments: projectName, year, month
* Produces monthly mean grids from all timesteps in the specified month for:
    * diffuse proportion
    * surface irradiance
    * topographically adjusted irradiance

####[itopo_1month.sh:](itopo_1month.sh)
* Script called with 3 arguments: projectName, year, month
* For the specified month, creates an output directory and calls the following python scripts (described above) in the proper order:
    * gen_diff_days.py
    * gen_itopo_days.py
    * mean_itopo_month.py

####[run_project.sh:](run_project.sh)
* Script called with 2 arguments: projectName and number of threads to use
    * Generates 360 horizon grids in parallel
    * Derives skyview from horizon grids
    * Iterates in parallel over months within timerange specified by 'projectName_parameters.yaml':
        * Calls itopo_1month.sh to:
            * create temporary output dir
            * perform topographic calculations for every timestep in specified month
            * generate monthly mean summaries, saved to permanent results folder
        * Deletes temporary folder holding timestep calculations for specified month

####[topocorr_funcs.py:](topocorr_funcs.py)
* Contains core functions called by other scripts:
    * Direct/diffuse proportion
    * Topographic correction
    * Solar angle as a function of time and position
    * Reprojections and file reading/writing/management

