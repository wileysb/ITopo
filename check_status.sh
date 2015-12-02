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
if [[ -n $1/dem_derivs/$1_skyview.tif ]] && [[ -r $1/dem_derivs/$1_skyview.tif ]]
then
    echo "Skyview exists"
else
    echo "**Skyview missing"
fi

# sunview
sunview_cache_size=$(ls -lh $1/dem_derivs/sunview/*.tif | wc -l)
if [[ $sunview_cache_size -eq 2928 ]]
then
    echo "Sunview cache is complete; turbo activated"
else
    echo "**Sunview cache incomplete; itopo will have to dynamically cast shade at each timestep"
fi



