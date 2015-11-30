#!/usr/bin/env bash

python gen_diff_days.py $1 $2 $3
python gen_itopo_days.py $1 $2 $3
python mean_itopo_month.py $1 $2 $3
rm -rf $1/tmp/$2$3