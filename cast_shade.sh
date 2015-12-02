#!/usr/bin/env bash

parallel -j $2 -- < $1/BOG.cmds
python accumulate_skyview.py $1
parallel -j $2 -- < /home/sl_wib/ITopo/$1/sunview_$1.cmds
