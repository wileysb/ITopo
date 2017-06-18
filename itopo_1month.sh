#!/usr/bin/env bash

# * Script called with 3 arguments: projectName, year, month
# * For the specified month, creates an output directory and calls
#   the following python scripts (described above) in the proper order:
#     * gen_diff_days.py
#     * gen_itopo_days.py
#     * mean_itopo_month.py

mkdir $1/tmp/$2$3
python gen_diff_days.py $1 $2 $3
python gen_itopo_days.py $1 $2 $3
python mean_itopo_month.py $1 $2 $3
