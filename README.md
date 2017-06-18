# ITopo
# Topographic correction of irradiance.

##### [Description of Project Files](TableOfContents.md)
##### [Dependencies](Dependencies.md)


### Directions for use:
1. Prepare a dem in the desired output projection and alignment
2. Place irradiance (3-hour SRB) data in accessible directory
3. Adjust parameters in parameters.yaml 
    - rename to projectName_parameters.yaml
4. Run $ python start_from_dem.py projectName /path/to/dem.tif
5. Run $ chmod 744 itopo_1month.sh
6. Choose a number of processors to tie up, and run the commands prompted at the completion of [4].


### WORKFLOW SUMMARY
1. Derive time-independent grids (DEM derivatives)
    - Slope, aspect, latitude, and longitude
    - Binary Obstruction Grids
    - sum over hemisphere, weighted by cos zenith, to derive skyview grid
2. for each 3 hour UTC timestep in 1 arbitrary year:
    - compute solar position grids (azimuth and zenith) in local time
    - For each pixel, calculate binary 'obstruction' value from local solar position and horizon grids
3. for every month from January 1984 - December 2007:
    - 3 hour UTC timestep within the given month:
        - [ ] compute solar position grids (azimuth and zenith) in local time
        - [ ] derive diffuse fraction of global radiation, after Skartveit et al \cite{skartveit98}
        - [ ] resample 'diffuse fraction' and 'downwelling shortwave at the surface' grids from 1x1\degree to 1x1km UTM-33N
        - [ ] topographic correction factor, direct irradiance
        - [ ] diffuse correction factor: diffuse fraction*skyview
        - [ ] topographic correction = diffuse correction + direct correction
    - average all timesteps for monthly averages
4. average all 23 years of each month for 'climatological monthly means'
