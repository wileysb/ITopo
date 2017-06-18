#!/usr/bin/env bash

# * Script called with 2 arguments: projectName and number of threads to use
#     * Generates 360 horizon grids in parallel
#     * Derives skyview from horizon grids
#     * Iterates in parallel over months within timerange specified by 'projectName_parameters.yaml':
#         * Calls itopo_1month.sh to:
#             * create temporary output dir
#             * perform topographic calculations for every timestep in specified month
#             * generate monthly mean summaries, saved to permanent results folder
#         * Deletes temporary folder holding timestep calculations for specified month


parallel -j $2 python find_horizon.py $1 {} ::: {0..359}
python skyview_from_horizon.py $1
parallel -j $2 -- < /home/sl_wib/ITopo/$1/itopo_months_$1.cmds
