#!/usr/bin/env bash

parallel -j $2 python find_horizon.py $1 {} ::: {0..359}
python skyview_from_horizon.py $1
# parallel -j $2 -- < /home/sl_wib/ITopo/$1/sunview_$1.cmds
parallel -j $2 -- < /home/sl_wib/ITopo/$1/itopo_months_$1.cmds
