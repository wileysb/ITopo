#!/usr/bin/env bash

# $1 (first argument) is project/directory name

# DEM exists?
if [[ -n $1/dem_derivs/$1_dem.tif ]] && [[ -r $1/dem_derivs/$1_dem.tif ]]
then
    echo "DEM in place"
else
    echo "**Start with the DEM . . ."
fi

# If BOGS are complete, there are 5760 ascii grid files in the directory
numBOGS=$(ls -lh $1/dem_derivs/BOGs/*.asc | wc -l)
slices=$(awk "BEGIN {print $numBOGS/360}")

re='^[0-9]+$'
if [[ $slices =~ $re ]]
then
    echo "BOGS: $slices degrees steepest slope"
else
    echo "** BOGS not a multiple of 360; $numBOGS files"
fi

# skyview
if [[ -n $1/dem_derivs/$1_skyview.asc ]] && [[ -r $1/dem_derivs/$1_skyview.asc ]]
then
    echo "Skyview exists"
else
    echo "**Skyview missing"
fi

# sunview
if [ $(ls -lh $1/dem_derivs/sunview/*.tif | wc -l) -eq 2928]
then
    echo "Sunview cache is complete; turbo activated!"
else
    echo "**Sunview cache incomplete; itopo will have to dynamically cast shade at each timestep"
fi




108K    ./porsgrunn1km/monthly_means
BOGS complete
sunview complete

46M     ./porsgrunn1km/dem_derivs/BOGs
12M     ./porsgrunn1km/dem_derivs/sunview
57M     ./porsgrunn1km/dem_derivs
4,0K    ./porsgrunn1km/tmp
58M     ./porsgrunn1km   

7,8M    ./porsgrunn100m/monthly_means
8,3G    ./porsgrunn100m/dem_derivs/BOGs
721M    ./porsgrunn100m/dem_derivs/sunview
9,0G    ./porsgrunn100m/dem_derivs
24K     ./porsgrunn100m/tmp
9,0G    ./porsgrunn100m  

2,2G    ./porsgrunn10mctr/monthly_means
128G    ./porsgrunn10mctr/dem_derivs/BOGs
7,1G    ./porsgrunn10mctr/dem_derivs/sunview
135G    ./porsgrunn10mctr/dem_derivs
411M    ./porsgrunn10mctr/tmp/200606
411M    ./porsgrunn10mctr/tmp
137G    ./porsgrunn10mctr

4,0K    ./gjende1km/monthly_means
4,0K    ./gjende1km/dem_derivs/BOGs
4,0K    ./gjende1km/dem_derivs/sunview
88K     ./gjende1km/dem_derivs
4,0K    ./gjende1km/tmp  
1,5M    ./gjende1km

4,0K    ./gjende100m/monthly_means
11G     ./gjende100m/dem_derivs/BOGs
721M    ./gjende100m/dem_derivs/sunview
12G     ./gjende100m/dem_derivs
4,0K    ./gjende100m/tmp 
12G     ./gjende100m

4,0K    ./gjende10mctr/monthly_means
4,0K    ./gjende10mctr/dem_derivs/BOGs
4,0K    ./gjende10mctr/dem_derivs/sunview
69M     ./gjende10mctr/dem_derivs
4,0K    ./gjende10mctr/tmp
73M     ./gjende10mctr   

4,0K    ./gtopo30norway1km/dem_derivs/BOGs
50M     ./gtopo30norway1km/dem_derivs
4,0K    ./gtopo30norway1km/tmp
50M     ./gtopo30norway1km

